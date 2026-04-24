"""Unit tests for aqua.diagnostics.lat_lon_profiles.util_cli."""

import pytest

from aqua.diagnostics.lat_lon_profiles.util_cli import load_var_config

pytestmark = [pytest.mark.diagnostics]


def test_string_variable_uses_default_variable_config():
    """A string var picks its configuration from default_variables when available."""
    config = {
        "diagnostics": {
            "lat_lon_profiles": {
                "default_variables": {
                    "2t": {"name": "t2m", "regions": ["tropics", "nh_midlat"], "units": "K"},
                }
            }
        }
    }

    var_config, regions = load_var_config(config, "2t")

    assert var_config["name"] == "t2m"
    assert var_config["units"] == "K"
    assert regions == ["tropics", "nh_midlat"]


def test_string_variable_without_default_falls_back_to_name_only():
    """A missing default variable returns {'name': var} and regions=[None]."""
    config = {"diagnostics": {"lat_lon_profiles": {"default_variables": {}}}}

    var_config, regions = load_var_config(config, "missing_var")

    assert var_config == {"name": "missing_var"}
    assert regions == [None]


def test_dict_variable_is_used_directly_with_regions():
    """A dict var is returned as config, and explicit regions are preserved."""
    var = {"name": "custom_var", "regions": ["global"], "long_name": "Custom"}
    config = {"diagnostics": {"lat_lon_profiles": {"default_variables": {}}}}

    var_config, regions = load_var_config(config, var)

    assert var_config is var
    assert var_config["long_name"] == "Custom"
    assert regions == ["global"]


def test_dict_variable_without_name_gets_name_from_var_argument():
    """If dict var misses 'name', util inserts name using the var argument itself."""
    var = {"regions": ["nh"]}
    config = {"diagnostics": {"lat_lon_profiles": {"default_variables": {}}}}

    var_config, regions = load_var_config(config, var)

    # Current function behavior: "name" is set to the raw `var` object.
    assert var_config["name"] is var
    assert regions == ["nh"]


def test_custom_diagnostic_key_is_supported():
    """The diagnostic parameter allows loading defaults from a custom section."""
    config = {
        "diagnostics": {
            "custom_diag": {
                "default_variables": {
                    "sos": {"name": "sea_surface_salinity", "regions": ["go"]},
                }
            }
        }
    }

    var_config, regions = load_var_config(config, "sos", diagnostic="custom_diag")

    assert var_config["name"] == "sea_surface_salinity"
    assert regions == ["go"]
