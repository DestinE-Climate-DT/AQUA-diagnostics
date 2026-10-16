"""Polygon collections not shipped with regionmask, listed in the ``sources`` of the regions file."""

import os
from functools import cache

import geopandas as gpd
import numpy as np
import pooch
import regionmask
import shapely

from aqua.core.lock import SafeFileLock


@cache
def load_region_source(name: str, url: str, names: str, simplify: float) -> regionmask.Regions:
    """Download a zipped shapefile once, simplify it and cache it next to the regionmask data."""
    cache_dir = os.path.join(regionmask.get_options()["cache_dir"] or pooch.os_cache("regionmask"), "aqua")
    os.makedirs(cache_dir, exist_ok=True)
    cached = os.path.join(cache_dir, f"{name}_{simplify}.gpkg")
    with SafeFileLock(cached + ".lock", timeout=900):  # parallel runs download it only once
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
