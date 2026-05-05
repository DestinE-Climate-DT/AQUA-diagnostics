"""Unit tests for aqua.diagnostics.global_biases.util.handle_pressure_level."""

import numpy as np
import pytest
import xarray as xr

from aqua.core.exceptions import NoDataError
from aqua.diagnostics.global_biases.util import handle_pressure_level

pytestmark = [pytest.mark.diagnostics]


def _make_dataset_with_plev(plev_values):
    """Build a minimal Dataset where variable `var` has a `plev` coordinate."""
    plev = np.atleast_1d(plev_values).astype(float)
    data = np.arange(plev.size * 2 * 2, dtype=float).reshape(plev.size, 2, 2)
    return xr.Dataset(
        {"var": (("plev", "lat", "lon"), data)},
        coords={"plev": plev, "lat": [0.0, 10.0], "lon": [0.0, 10.0]},
    )


def _make_dataset_without_plev():
    """Build a minimal Dataset where variable `var` has no `plev` coordinate."""
    data = np.arange(4, dtype=float).reshape(2, 2)
    return xr.Dataset(
        {"var": (("lat", "lon"), data)},
        coords={"lat": [0.0, 10.0], "lon": [0.0, 10.0]},
    )


class TestHandlePressureLevel:
    def test_missing_variable_raises_nodataerror(self):
        ds = _make_dataset_without_plev()
        with pytest.raises(NoDataError, match="Variable 'missing' not found"):
            handle_pressure_level(ds, "missing", plev=None)

    def test_variable_without_plev_and_no_plev_arg_returns_data(self):
        ds = _make_dataset_without_plev()
        out = handle_pressure_level(ds, "var", plev=None)
        assert out is ds

    def test_variable_without_plev_but_plev_requested_raises(self):
        ds = _make_dataset_without_plev()
        with pytest.raises(ValueError, match="does not have a 'plev' dimension"):
            handle_pressure_level(ds, "var", plev=500.0)

    def test_multiple_levels_without_plev_request_returns_full_data(self):
        ds = _make_dataset_with_plev([500.0, 850.0])
        out = handle_pressure_level(ds, "var", plev=None)
        assert out is ds

    def test_multiple_levels_selects_nearest(self):
        ds = _make_dataset_with_plev([500.0, 850.0])
        out = handle_pressure_level(ds, "var", plev=850.0)
        assert float(out["var"].coords["plev"].values) == 850.0
        # The plev dimension collapses when a single level is selected.
        assert "plev" not in out["var"].dims
