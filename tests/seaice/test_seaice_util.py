"""Unit tests for aqua.diagnostics.seaice.util."""

import logging
from collections import defaultdict
from datetime import datetime

import pytest
import xarray as xr

from aqua.diagnostics.seaice.util import (
    _check_list_regions_type,
    defaultdict_to_dict,
    ensure_istype,
    extract_dates,
    filter_region_list,
)

pytestmark = [pytest.mark.diagnostics]


# ======================================================================
# defaultdict_to_dict
# ======================================================================
class TestDefaultdictToDict:
    def test_plain_value_is_returned_unchanged(self):
        assert defaultdict_to_dict(42) == 42
        assert defaultdict_to_dict({"a": 1}) == {"a": 1}

    def test_flat_defaultdict_is_converted(self):
        dd = defaultdict(list)
        dd["a"].append(1)
        out = defaultdict_to_dict(dd)
        assert isinstance(out, dict) and not isinstance(out, defaultdict)
        assert out == {"a": [1]}

    def test_nested_defaultdict_is_converted_recursively(self):
        dd = defaultdict(lambda: defaultdict(list))
        dd["outer"]["inner"].append("x")
        out = defaultdict_to_dict(dd)
        assert isinstance(out, dict) and not isinstance(out, defaultdict)
        assert isinstance(out["outer"], dict) and not isinstance(out["outer"], defaultdict)
        assert out == {"outer": {"inner": ["x"]}}


# ======================================================================
# filter_region_list
# ======================================================================
class TestFilterRegionList:
    @pytest.fixture(scope="module")
    def regions_dict(self):
        return {
            "regions": {
                "arctic": {"lat_limits": [60, 90]},
                "antarctic": {"lat_limits": [-90, -60]},
                "no_limits": {"lat_limits": []},
            }
        }

    @pytest.fixture(scope="module")
    def logger(self):
        return logging.getLogger("test_filter_region_list")

    def test_nh_keeps_northern_regions(self, regions_dict, logger):
        out = filter_region_list(regions_dict, ["arctic", "antarctic"], "nh", logger)
        assert out == ["arctic"]

    def test_sh_keeps_southern_regions(self, regions_dict, logger):
        out = filter_region_list(regions_dict, ["arctic", "antarctic"], "sh", logger)
        assert out == ["antarctic"]

    def test_unknown_region_is_skipped_and_logged(self, regions_dict, caplog):
        logger = logging.getLogger("test_unknown")
        with caplog.at_level(logging.ERROR, logger=logger.name):
            out = filter_region_list(regions_dict, ["arctic", "unknown"], "nh", logger)
        assert out == ["arctic"]
        assert any("No region 'unknown'" in rec.message for rec in caplog.records)

    def test_invalid_domain_raises(self, regions_dict, logger):
        with pytest.raises(ValueError, match="Invalid domain"):
            filter_region_list(regions_dict, ["arctic"], "bogus", logger)

    def test_custom_valid_domains_are_accepted(self, regions_dict, logger):
        # When a custom domain list is provided, it replaces the default nh/sh.
        # With domain='nh' still allowed, arctic should come through.
        out = filter_region_list(regions_dict, ["arctic"], "nh", logger, valid_domains=["nh", "sh", "global"])
        assert out == ["arctic"]


# ======================================================================
# ensure_istype
# ======================================================================
class TestEnsureIstype:
    def test_single_type_passes(self):
        ensure_istype(42, int)  # no exception

    def test_single_type_raises(self):
        with pytest.raises(ValueError, match="Expected type int"):
            ensure_istype("foo", int)

    def test_tuple_of_types_passes(self):
        ensure_istype("foo", (int, str))

    def test_tuple_of_types_raises_includes_all_names(self):
        with pytest.raises(ValueError, match="int, str"):
            ensure_istype(1.2, (int, str))


# ======================================================================
# extract_dates
# ======================================================================
class TestExtractDates:
    def test_datetime_like_values_are_formatted(self):
        da = xr.DataArray(
            [0.0],
            attrs={
                "AQUA_startdate": datetime(2020, 3, 1),
                "AQUA_enddate": datetime(2021, 4, 2),
            },
        )
        assert extract_dates(da) == ("2020-03-01", "2021-04-02")

    def test_iso_string_with_time_part_is_trimmed(self):
        da = xr.DataArray(
            [0.0],
            attrs={
                "AQUA_startdate": "2020-01-01T00:00:00",
                "AQUA_enddate": "2020-12-31T23:59:59",
            },
        )
        assert extract_dates(da) == ("2020-01-01", "2020-12-31")

    def test_plain_string_is_returned_unchanged(self):
        da = xr.DataArray([0.0], attrs={"AQUA_startdate": "2020", "AQUA_enddate": "2021"})
        assert extract_dates(da) == ("2020", "2021")

    def test_missing_attributes_return_placeholder(self):
        da = xr.DataArray([0.0])
        start, end = extract_dates(da)
        assert "No AQUA_startdate" in start
        assert "No AQUA_enddate" in end


# ======================================================================
# _check_list_regions_type
# ======================================================================
class TestCheckListRegionsType:
    @pytest.fixture(scope="module")
    def logger(self):
        return logging.getLogger("test_check_list_regions_type")

    def test_none_returns_none_and_warns(self, logger, caplog):
        with caplog.at_level(logging.WARNING, logger=logger.name):
            out = _check_list_regions_type(None, logger=logger)
        assert out is None
        assert any("Expected regions_to_plot to be a list" in rec.message for rec in caplog.records)

    def test_non_list_raises_typeerror(self, logger):
        with pytest.raises(TypeError, match="to be a list"):
            _check_list_regions_type("arctic", logger=logger)

    def test_list_with_non_string_raises(self, logger):
        with pytest.raises(TypeError, match="list of strings"):
            _check_list_regions_type(["arctic", 1], logger=logger)

    def test_valid_list_is_returned_as_is(self, logger):
        assert _check_list_regions_type(["arctic", "antarctic"], logger=logger) == ["arctic", "antarctic"]
