"""Unit tests for aqua.diagnostics.timeseries.util_cli."""

import pytest

from aqua.diagnostics.timeseries.util_cli import load_var_config

pytestmark = [pytest.mark.diagnostics]


def test_timeseries_uses_default_when_variable_specific_missing():
    """If var is not configured, defaults are used and converted to freq list."""
    config = {
        "diagnostics": {
            "timeseries": {
                "params": {
                    "default": {
                        "hourly": False,
                        "daily": False,
                        "monthly": True,
                        "annual": True,
                        "std": True,
                    }
                }
            }
        }
    }

    var_config, regions = load_var_config(config, "2t", diagnostic="timeseries")

    assert var_config["freq"] == ["monthly", "annual"]
    assert var_config["std"] is True
    for key in ["hourly", "daily", "monthly", "annual"]:
        assert key not in var_config
    assert regions == [None]


def test_timeseries_variable_specific_overrides_default_values():
    """Var-specific params take precedence over defaults before frequency extraction."""
    config = {
        "diagnostics": {
            "timeseries": {
                "params": {
                    "default": {
                        "hourly": False,
                        "daily": False,
                        "monthly": True,
                        "annual": False,
                        "std": False,
                        "line_style": "solid",
                    },
                    "2t": {
                        "daily": True,
                        "monthly": False,
                        "annual": True,
                        "std": True,
                        "line_style": "dashed",
                    },
                }
            }
        }
    }

    var_config, _ = load_var_config(config, "2t", diagnostic="timeseries")

    # From merged hourly/daily/monthly/annual flags.
    assert var_config["freq"] == ["daily", "annual"]
    assert var_config["std"] is True
    assert var_config["line_style"] == "dashed"


def test_regions_are_extracted_and_none_values_are_filtered():
    """regions key is removed from var_config and returned as [None] + valid regions."""
    config = {
        "diagnostics": {
            "timeseries": {
                "params": {
                    "default": {
                        "hourly": False,
                        "daily": False,
                        "monthly": True,
                        "annual": False,
                        "regions": [None, "global", "nh"],
                    }
                }
            }
        }
    }

    var_config, regions = load_var_config(config, "2t", diagnostic="timeseries")

    assert regions == [None, "global", "nh"]
    assert "regions" not in var_config


def test_non_timeseries_diagnostic_keeps_time_keys():
    """For diagnostics other than 'timeseries', hourly/daily/monthly/annual keys are untouched."""
    config = {
        "diagnostics": {
            "seasonalcycles": {
                "params": {
                    "default": {
                        "hourly": False,
                        "daily": False,
                        "monthly": True,
                        "annual": True,
                        "custom": 7,
                    },
                    "2t": {"annual": False},
                }
            }
        }
    }

    var_config, regions = load_var_config(config, "2t", diagnostic="seasonalcycles")

    assert var_config["monthly"] is True
    assert var_config["annual"] is False
    assert "freq" not in var_config
    assert var_config["custom"] == 7
    assert regions == [None]
