"""Tests for the centralized region lookup in the ``Diagnostic`` base class."""

from textwrap import dedent

import pytest

from aqua.diagnostics.base import Diagnostic
from tests.shared_constants import LOGLEVEL

pytestmark = [pytest.mark.aqua]


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
        assert "lon_limits" in regions[name] and "lat_limits" in regions[name]


# _set_region
def test_set_region_none_passes_through_limits():
    diag = _make_diag()
    name, lon, lat = diag._set_region(region=None, lon_limits=[0, 10], lat_limits=[20, 30])
    assert name is None
    assert lon == [0, 10]
    assert lat == [20, 30]


def test_set_region_resolves_against_centralized_file():
    diag = _make_diag()
    name, lon, lat = diag._set_region(region="nh")
    assert name == "North Midlatitudes"
    assert lat == [30, 90]
    assert lon == [-180, 180]


def test_set_region_resolves_seaice_entry_too():
    """Any diagnostic can pick any region — sea-ice geometries included."""
    diag = _make_diag()
    name, _, lat = diag._set_region(region="arctic")
    assert name == "Arctic"
    assert lat == [0, 90]


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
