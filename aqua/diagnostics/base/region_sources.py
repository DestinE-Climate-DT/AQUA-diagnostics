"""Polygon collections listed in the ``sources`` of the regions file."""

import glob
import os
import shutil
import tempfile
from functools import cache

import geopandas as gpd
import numpy as np
import pooch
import regionmask
import shapely
from filelock import SoftFileLock

from aqua.core.util import load_yaml


@cache
def load_region_source(name: str, url: str, names: str, simplify: float, folder: str | None = None) -> regionmask.Regions:
    """
    Read the polygons from ``folder`` if a copy is there, otherwise download the
    zipped shapefile once, simplify it and cache it next to the regionmask data.
    """
    filename = f"{name}_{simplify}.gpkg"
    shipped = os.path.join(folder, filename) if folder else ""
    if os.path.exists(shipped):
        return regionmask.from_geopandas(gpd.read_file(shipped), names=names, name=name, overlap=False)

    cache_dir = os.path.join(regionmask.get_options()["cache_dir"] or pooch.os_cache("regionmask"), "aqua")
    os.makedirs(cache_dir, exist_ok=True)
    cached = os.path.join(cache_dir, filename)
    # Checked before the lock too, which cannot be created in a read-only cache
    if not os.path.exists(cached):
        with SoftFileLock(cached + ".lock", timeout=900):  # parallel runs download it only once
            if not os.path.exists(cached):
                raw = pooch.retrieve(url, known_hash=None, fname=f"{name}.zip", path=cache_dir)
                data = gpd.read_file(f"zip://{raw}")
                geometry = shapely.simplify(data.geometry.values, simplify, preserve_topology=False)
                # Some polygons stop a hair short of ±180°: snap them to it, or the cells centred there are lost
                geometry = shapely.transform(geometry, lambda xy: np.where(np.abs(xy) > 179.99, np.sign(xy) * 180, xy))
                data["geometry"] = shapely.make_valid(geometry, method="structure", keep_collapsed=False)
                data.to_file(cached)
                os.remove(raw)
    return regionmask.from_geopandas(gpd.read_file(cached), names=names, name=name, overlap=False)


def fetch_region_sources(regions_file: str):
    """Download all the sources of a regions file into the regions/ folder next to it, as the container build does."""
    folder = os.path.join(os.path.dirname(regions_file), "regions")
    os.makedirs(folder, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, regionmask.set_options(cache_dir=tmp):
        for name, spec in load_yaml(regions_file)["sources"].items():
            load_region_source(name, **spec)
        for gpkg in glob.glob(os.path.join(tmp, "aqua", "*.gpkg")):
            shutil.move(gpkg, os.path.join(folder, os.path.basename(gpkg)))
