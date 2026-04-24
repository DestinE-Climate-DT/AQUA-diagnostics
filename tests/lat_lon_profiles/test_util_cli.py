"""Unit tests for aqua.diagnostics.lat_lon_profiles.util_cli."""

import pytest

from aqua.diagnostics.lat_lon_profiles.util_cli import load_var_config

pytestmark = [pytest.mark.diagnostics]


def test_string_variable_uses_per_var_params():
    """A string var picks its configuration from params.<var_name> when available."""
    config = {
        "diagnostics": {
            "lat_lon_profiles": {
                "params": {
                    "2t": {"regions": ["tropics", "nh_midlat"], "units": "K"},
                }
            }
        }
    }

    var_config, regions = load_var_config(config, "2t")

    assert var_config["name"] == "2t"
    assert var_config["units"] == "K"
    assert regions == ["tropics", "nh_midlat"]


def test_string_variable_without_params_falls_back_to_name_only():
    """With no params block, a string var returns {'name': var} and regions=[None]."""
    config = {"diagnostics": {"lat_lon_profiles": {}}}

    var_config, regions = load_var_config(config, "missing_var")

    assert var_config == {"name": "missing_var"}
    assert regions == [None]


def test_params_default_applies_to_string_variable():
    """Fields under params.default are merged into the config for any variable."""
    config = {
        "diagnostics": {
            "lat_lon_profiles": {
                "params": {
                    "default": {"std_startdate": "19900101", "std_enddate": "20201231"},
                }
            }
        }
    }

    var_config, _ = load_var_config(config, "t2m")

    assert var_config["name"] == "t2m"
    assert var_config["std_startdate"] == "19900101"
    assert var_config["std_enddate"] == "20201231"


def test_dict_variable_merges_params_default_per_var_and_inline():
    """Merge precedence: params.default < params.<name> < inline var dict."""
    config = {
        "diagnostics": {
            "lat_lon_profiles": {
                "params": {
                    "default": {"std_startdate": "19900101", "units": "base"},
                    "custom_var": {"units": "from_params", "long_name": "From params"},
                }
            }
        }
    }
    var = {"name": "custom_var", "regions": ["global"], "long_name": "Inline"}

    var_config, regions = load_var_config(config, var)

    # Inline wins over per-var params
    assert var_config["long_name"] == "Inline"
    # Per-var params wins over default
    assert var_config["units"] == "from_params"
    # params.default carries through when not overridden
    assert var_config["std_startdate"] == "19900101"
    assert regions == ["global"]


def test_custom_diagnostic_key_is_supported():
    """The diagnostic parameter allows loading params from a custom section."""
    config = {
        "diagnostics": {
            "custom_diag": {
                "params": {
                    "sos": {"regions": ["go"], "units": "psu"},
                }
            }
        }
    }

    var_config, regions = load_var_config(config, "sos", diagnostic="custom_diag")

    assert var_config["name"] == "sos"
    assert var_config["units"] == "psu"
    assert regions == ["go"]
