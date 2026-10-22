"""Tests for the centralized region lookup in the ``Diagnostic`` base class."""

import shutil
from textwrap import dedent

import geopandas as gpd
import pytest
import shapely

from aqua.diagnostics.base import Diagnostic, region_sources
from tests.shared_constants import LOGLEVEL

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]


def _make_diag():
    """Minimal Diagnostic instance for invoking region-helper methods."""
    return Diagnostic(model="m", exp="e", source="s", loglevel=LOGLEVEL)


def _write_yaml(path, body):
    path.write_text(dedent(body).lstrip())
    return str(path)


# _load_regions_from_file
def test_load_regions_from_file_returns_flat_dict(tmp_path):
    path = _write_yaml(
        tmp_path / "r.yaml",
        """
        regions:
          mybox: {longname: My Box, lon_limits: [0, 10], lat_limits: [0, 10]}
          other: {longname: Other, lon_limits: [20, 30], lat_limits: [20, 30]}
    """,
    )
    diag = _make_diag()
    result = diag._load_regions_from_file(regions_file_path=path)
    assert set(result) == {"mybox", "other"}
    assert result["mybox"]["longname"] == "My Box"


def test_load_regions_from_file_centralized_has_known_entries():
    """Sanity-check that the shipped centralized regions file is reachable and well-formed."""
    diag = _make_diag()
    regions = diag._load_regions_from_file()
    # A handful of entries that should always be present
    for name in ("nh", "tropics", "arctic", "antarctic", "io"):
        assert name in regions, f"missing '{name}' in centralized regions file"
        # Each region is either a lon/lat box or a regionmask-based region (e.g. 'io')
        is_box = "lon_limits" in regions[name] and "lat_limits" in regions[name]
        assert is_box or "regionmask" in regions[name]


# _set_region
def test_set_region_none_passes_through_limits():
    diag = _make_diag()
    name, lon, lat = diag._set_region(region=None, lon_limits=[0, 10], lat_limits=[20, 30])
    assert name is None
    assert lon == [0, 10]
    assert lat == [20, 30]


@pytest.mark.parametrize(
    "region, expected",
    [
        ("nh", ("North Midlatitudes", [-180, 180], [30, 90])),
        ("arctic", ("Arctic", [0, 360], [0, 90])),  # any diagnostic can pick sea-ice regions too
        ("io", ("Indian Ocean", None, None)),  # polygons (IHO) have no lon/lat limits
    ],
)
def test_set_region_resolves_against_centralized_file(region, expected):
    assert _make_diag()._set_region(region=region) == expected


def test_set_region_unknown_name_raises_with_list():
    diag = _make_diag()
    with pytest.raises(ValueError, match="not found"):
        diag._set_region(region="atlantis")


def test_set_region_user_override_file(tmp_path):
    """A user-supplied regions file overrides the centralized lookup."""
    path = _write_yaml(
        tmp_path / "user.yaml",
        """
        regions:
          mybox: {longname: My Box, lon_limits: [1, 2], lat_limits: [3, 4]}
    """,
    )
    diag = _make_diag()
    name, lon, lat = diag._set_region(region="mybox", regions_file_path=path)
    assert name == "My Box"
    assert lon == [1, 2]
    assert lat == [3, 4]


def test_set_region_user_file_does_not_see_centralized_entries(tmp_path):
    """Override files fully replace the centralized lookup, not merge with it."""
    path = _write_yaml(
        tmp_path / "user.yaml",
        """
        regions:
          mybox: {longname: My Box, lon_limits: [1, 2], lat_limits: [3, 4]}
    """,
    )
    diag = _make_diag()
    with pytest.raises(ValueError, match="not found"):
        diag._set_region(region="nh", regions_file_path=path)


# _resolve_regionmask_object
def test_resolve_regionmask_object_reads_shipped_copy(tmp_path):
    """Polygons in regions/ next to the regions file (as in the container) are read without downloading."""
    path = _write_yaml(
        tmp_path / "user.yaml",
        """
        sources:
          box: {url: "https://invalid.invalid/box.zip", names: name, simplify: 0}
        regions:
          mybox: {longname: My Box, regionmask: box, regionmask_names: [Box], frac_threshold: 0.0}
    """,
    )
    (tmp_path / "regions").mkdir()
    box = gpd.GeoDataFrame({"name": ["Box"]}, geometry=[shapely.box(0, 0, 10, 10)], crs="EPSG:4326")
    box.to_file(tmp_path / "regions" / "box_0.gpkg")
    region_object, region_sel, frac_threshold = _make_diag()._resolve_regionmask_object("mybox", regions_file_path=path)
    assert region_object.names == ["Box"]
    assert (region_sel, frac_threshold) == (["Box"], 0.0)


# fetch_region_sources
def test_fetch_region_sources_fills_regions_folder(tmp_path, monkeypatch):
    """The container build step: every source is downloaded, simplified and stored in regions/ next to the regions file."""
    (tmp_path / "shp").mkdir()
    box = gpd.GeoDataFrame({"name": ["Box"]}, geometry=[shapely.box(0, 0, 10, 10)], crs="EPSG:4326")
    box.to_file(tmp_path / "shp" / "box.shp")
    zipped = shutil.make_archive(str(tmp_path / "box"), "zip", tmp_path / "shp")
    monkeypatch.setattr(region_sources.pooch, "retrieve", lambda *args, **kwargs: zipped)  # no network
    (tmp_path / "definitions").mkdir()
    path = _write_yaml(
        tmp_path / "definitions" / "regions.yaml",
        """
        sources:
          fetchbox: {url: "https://invalid.invalid/box.zip", names: name, simplify: 0}
    """,
    )
    region_sources.fetch_region_sources(path)
    assert gpd.read_file(tmp_path / "definitions" / "regions" / "fetchbox_0.gpkg")["name"].tolist() == ["Box"]
