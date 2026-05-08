"""Unit tests for aqua.diagnostics.base.util.load_var_config."""

import pytest

from aqua.diagnostics.base import load_var_config

pytestmark = [pytest.mark.diagnostics]

_FREQ_KEYS = ("hourly", "daily", "monthly", "annual")


def test_merge_precedence_default_per_var_inline():
    """params.default < params.<name> < inline dict, with regions popped out."""
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

    var_config, regions = load_var_config(config, var, diagnostic="lat_lon_profiles")

    assert var_config["long_name"] == "Inline"  # inline wins
    assert var_config["units"] == "from_params"  # per-var wins over default
    assert var_config["std_startdate"] == "19900101"  # default carries through
    assert regions == ["global"]
    assert "regions" not in var_config


def test_string_var_with_empty_diagnostic_block_raises():
    """A string var with an empty diagnostics.<name> block has nothing to merge -> error."""
    with pytest.raises(ValueError, match="missing or empty"):
        load_var_config(
            {"diagnostics": {"lat_lon_profiles": {}}},
            "missing_var",
            diagnostic="lat_lon_profiles",
        )


def test_inline_dict_without_params_block():
    """An inline var dict works on its own (histogram template style)."""
    config = {"diagnostics": {"histogram": {}}}
    var = {"name": "2t", "regions": [None, "tropics"], "units": "K"}

    var_config, regions = load_var_config(config, var, diagnostic="histogram")

    assert var_config == {"name": "2t", "units": "K"}
    assert regions == [None, "tropics"]


def test_collapse_freq_keys_builds_freq_list_and_pops_flags():
    """Boolean frequency flags are collapsed into a freq list and removed."""
    config = {
        "diagnostics": {
            "timeseries": {
                "params": {"default": {"hourly": False, "daily": False, "monthly": True, "annual": True, "std": True}}
            }
        }
    }

    var_config, _ = load_var_config(
        config,
        "2t",
        diagnostic="timeseries",
        collapse_freq_keys=_FREQ_KEYS,
    )

    assert var_config["freq"] == ["monthly", "annual"]
    assert var_config["std"] is True
    for key in _FREQ_KEYS:
        assert key not in var_config


def test_prepend_global_prepends_none_and_dedupes():
    """prepend_global=True puts a single None at the head of the regions list."""
    config = {"diagnostics": {"timeseries": {"params": {"default": {"regions": [None, "global", "nh"]}}}}}

    _, regions = load_var_config(
        config,
        "2t",
        diagnostic="timeseries",
        prepend_global=True,
    )

    assert regions == [None, "global", "nh"]
