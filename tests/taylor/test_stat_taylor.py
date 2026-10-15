"""Tests for the Taylor statistics, on synthetic fields with known results."""

import numpy as np
import pytest
import xarray as xr

from aqua.diagnostics.taylor import compute_taylor_statistics
from tests.shared_constants import APPROX_REL, LOGLEVEL

pytestmark = [pytest.mark.diagnostics]


@pytest.fixture
def field():
    """A smooth lat-lon field with some structure."""
    lat = np.linspace(-89.5, 89.5, 30)
    lon = np.linspace(0, 354, 60)
    values = np.cos(np.deg2rad(lat))[:, None] * 30 + np.sin(np.deg2rad(lon))[None, :] * 5
    return xr.DataArray(values, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"], attrs={"units": "K"})


@pytest.fixture
def weights(field):
    """Area-like weights."""
    return np.cos(np.deg2rad(field.lat)) * xr.ones_like(field)


class TestComputeTaylorStatistics:
    """Statistics on synthetic fields."""

    def test_identical_fields(self, field, weights):
        """A field compared with itself is the reference point."""
        stats = compute_taylor_statistics(field, field, weights=weights, loglevel=LOGLEVEL)
        assert float(stats["corr"]) == pytest.approx(1.0, rel=APPROX_REL)
        assert float(stats["std_norm"]) == pytest.approx(1.0, rel=APPROX_REL)
        assert float(stats["crmse"]) == pytest.approx(0.0, abs=1e-10)
        assert float(stats["bias"]) == pytest.approx(0.0, abs=1e-10)
        assert stats["std"].attrs["units"] == "K"
        assert stats["corr"].attrs["units"] == "1"

    def test_scaled_and_shifted_field(self, field, weights):
        """Scaling changes only the std, shifting only the bias."""
        stats = compute_taylor_statistics(2 * field + 3, field, weights=weights)
        assert float(stats["corr"]) == pytest.approx(1.0, rel=APPROX_REL)
        assert float(stats["std_norm"]) == pytest.approx(2.0, rel=APPROX_REL)
        assert float(stats["crmse_norm"]) == pytest.approx(1.0, rel=APPROX_REL)
        assert float(stats["bias"]) == pytest.approx(float((field + 3).weighted(weights).mean()), rel=APPROX_REL)

    def test_anticorrelated_field(self, field):
        """The opposite anomaly has correlation -1."""
        stats = compute_taylor_statistics(-field, field)
        assert float(stats["corr"]) == pytest.approx(-1.0, rel=APPROX_REL)
        assert float(stats["crmse_norm"]) == pytest.approx(2.0, rel=APPROX_REL)

    def test_taylor_geometry(self, field, weights):
        """The law of cosines of the Taylor diagram holds, as well as rmse^2 = bias^2 + crmse^2."""
        rng = np.random.default_rng(42)
        other = field * 0.7 + rng.normal(0, 5, field.shape) + 1.5
        stats = compute_taylor_statistics(other, field, weights=weights)
        std, std_ref, corr, crmse = (float(stats[v]) for v in ("std", "std_ref", "corr", "crmse"))
        assert crmse**2 == pytest.approx(std**2 + std_ref**2 - 2 * std * std_ref * corr, rel=APPROX_REL)
        assert float(stats["rmse"]) ** 2 == pytest.approx(float(stats["bias"]) ** 2 + crmse**2, rel=APPROX_REL)
        assert -1 < corr < 1

    def test_weights_matter(self, field, weights):
        """Weighted and unweighted statistics differ for a latitude dependent difference."""
        other = field + np.abs(field.lat)
        weighted = compute_taylor_statistics(other, field, weights=weights)
        unweighted = compute_taylor_statistics(other, field)
        assert float(weighted["bias"]) != pytest.approx(float(unweighted["bias"]), rel=APPROX_REL)

    def test_joint_mask(self, field):
        """A point missing in one field is excluded from both."""
        masked = field.where(field.lat > 0)
        stats = compute_taylor_statistics(masked, field)
        expected = compute_taylor_statistics(masked, field.where(field.lat > 0))
        assert float(stats["std_ref"]) == pytest.approx(float(expected["std_ref"]), rel=APPROX_REL)
        assert float(stats["corr"]) == pytest.approx(1.0, rel=APPROX_REL)

    def test_kept_dimension(self, field):
        """Dimensions not in dims are kept, one set of statistics each."""
        stacked = xr.concat([field, 2 * field], dim="period").assign_coords(period=["annual", "DJF"])
        stats = compute_taylor_statistics(stacked, stacked.sel(period="annual"), dims=["lat", "lon"])
        assert stats["std_norm"].dims == ("period",)
        np.testing.assert_allclose(stats["std_norm"].values, [1.0, 2.0])

    def test_grid_tolerance(self, field):
        """Coordinates differing by floating point noise are the same grid."""
        noisy = field.assign_coords(lat=field.lat + 1e-10)
        stats = compute_taylor_statistics(field, noisy)
        assert float(stats["corr"]) == pytest.approx(1.0, rel=APPROX_REL)

    @pytest.mark.parametrize(
        "ref_transform",
        [
            lambda f: f.isel(lat=slice(1, None)),
            lambda f: f.assign_coords(lon=f.lon + 1),
            lambda f: f.rename(lat="y"),
        ],
        ids=["size", "coords", "dims"],
    )
    def test_different_grids_raise(self, field, ref_transform):
        """Data and reference on different grids raise a ValueError."""
        with pytest.raises(ValueError, match="same grid"):
            compute_taylor_statistics(field, ref_transform(field), dims=["lat", "lon"])
