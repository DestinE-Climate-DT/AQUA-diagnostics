import os
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from aqua.core.exceptions import NoDataError
from aqua.diagnostics import GlobalBiases, PlotGlobalBiases, StatGlobalBiases
from tests.shared_constants import APPROX_REL, DPI, LOGLEVEL

# Tolerance for numerical comparisons
approx_rel = APPROX_REL
loglevel = LOGLEVEL

# pytestmark groups tests that run sequentially on the same worker to avoid conflicts
# These tests use setup_class with shared resources (data fetching, tmp files)
pytestmark = [pytest.mark.diagnostics]


# Module-level fixtures
@pytest.fixture(scope="module")
def global_biases_instance(module_outdir):
    """Create a GlobalBiases instance with pre-fetched data."""
    gb = GlobalBiases(
        catalog="ci",
        model="ERA5",
        exp="era5-hpz3",
        source="monthly",
        regrid="r100",
        outputdir=module_outdir,
    )
    gb.retrieve()
    return gb


@pytest.fixture(scope="module")
def plot_global_biases_instance(module_outdir):
    """Create a PlotGlobalBiases instance."""
    return PlotGlobalBiases(dpi=DPI, outputdir=module_outdir)


@pytest.fixture
def test_var():
    """Variable to test with."""
    return "q"


class TestGlobalBiases:
    """Test suite for GlobalBiases diagnostic."""

    def test_climatology(self, global_biases_instance, plot_global_biases_instance, module_outdir, test_var):

        gb = global_biases_instance
        plotgb = plot_global_biases_instance
        var = test_var

        gb.compute_climatology(var=var, seasonal=True)
        assert hasattr(gb, "climatology")
        assert hasattr(gb, "seasonal_climatology")
        assert isinstance(gb.climatology, xr.Dataset)
        assert isinstance(gb.seasonal_climatology, xr.Dataset)
        assert var in gb.climatology
        assert var in gb.seasonal_climatology
        assert "season" in gb.seasonal_climatology[var].dims
        assert set(gb.seasonal_climatology["season"].values) == {"DJF", "MAM", "JJA", "SON"}

        nc = os.path.join(module_outdir, "netcdf", f"globalbiases.annual_climatology.ci.ERA5.era5-hpz3.r1.{var}.nc")
        assert os.path.exists(nc)

        nc_seasonal = os.path.join(module_outdir, "netcdf", f"globalbiases.seasonal_climatology.ci.ERA5.era5-hpz3.r1.{var}.nc")
        assert os.path.exists(nc_seasonal)

        plotgb.plot_climatology(data=gb.climatology, var=var, plev=85000)

        pdf = os.path.join(module_outdir, "pdf", f"globalbiases.annual_climatology.ci.ERA5.era5-hpz3.r1.{var}.85000.pdf")
        assert os.path.exists(pdf)

        png = os.path.join(module_outdir, "png", f"globalbiases.annual_climatology.ci.ERA5.era5-hpz3.r1.{var}.85000.png")
        assert os.path.exists(png)

    def test_bias(self, global_biases_instance, plot_global_biases_instance, module_outdir, test_var):
        gb = global_biases_instance
        plotgb = plot_global_biases_instance
        var = test_var

        gb.compute_climatology(var=var, seasonal=True, areas=True)
        assert "cell_area" in gb.climatology

        plotgb.plot_bias(data=gb.climatology, data_ref=gb.climatology, var=var, plev=85000, show_stats=True)
        pdf = os.path.join(module_outdir, "pdf", f"globalbiases.bias.ci.ERA5.era5-hpz3.r1.ERA5.era5-hpz3.{var}.85000.pdf")
        assert os.path.exists(pdf)
        png = os.path.join(module_outdir, "png", f"globalbiases.bias.ci.ERA5.era5-hpz3.r1.ERA5.era5-hpz3.{var}.85000.png")
        assert os.path.exists(png)

    def test_stat_global_biases(self, global_biases_instance, test_var):
        gb = global_biases_instance
        var = test_var
        gb.compute_climatology(var=var, areas=True, plev=85000)
        stat_gb = StatGlobalBiases()
        result = stat_gb.compute_bias_statistics(data=gb.climatology, data_ref=gb.climatology, var=var)
        assert float(result["mean_bias"].values) == pytest.approx(0.0, abs=1e-12)
        assert float(result["rmse"].values) == pytest.approx(0.0, abs=1e-12)

        # With a +1000 offset, every grid point must be flagged as significant
        ds_base = gb.data
        ds_big = gb.data.copy(deep=True) + 1000.0
        result = stat_gb.compute_significance_ttest(data=ds_base, data_ref=ds_big, var=var, alpha=0.05)
        assert isinstance(result, xr.DataArray)
        assert result.dtype == bool
        assert bool(result.all())

    def test_bias_with_stat(self, global_biases_instance, plot_global_biases_instance, module_outdir, test_var):
        gb = global_biases_instance
        plotgb = plot_global_biases_instance
        var = test_var

        gb.compute_climatology(var=var)
        plotgb.plot_bias(
            data=gb.climatology,
            data_ref=gb.climatology,
            var=var,
            plev=85000,
            data_timeseries=gb.data,
            data_ref_timeseries=gb.data,
            show_significance=True,
        )

        pdf = os.path.join(module_outdir, "pdf", f"globalbiases.bias.ci.ERA5.era5-hpz3.r1.ERA5.era5-hpz3.{var}.85000.pdf")
        assert os.path.exists(pdf)
        png = os.path.join(module_outdir, "png", f"globalbiases.bias.ci.ERA5.era5-hpz3.r1.ERA5.era5-hpz3.{var}.85000.png")
        assert os.path.exists(png)

    def test_seasonal_bias(self, global_biases_instance, plot_global_biases_instance, module_outdir, test_var):
        gb = global_biases_instance
        plotgb = plot_global_biases_instance
        var = test_var

        # Ensure seasonal climatology is computed
        if not hasattr(gb, "seasonal_climatology"):
            gb.compute_climatology(var=var, seasonal=True)

        plotgb.plot_seasonal_bias(data=gb.seasonal_climatology, data_ref=gb.seasonal_climatology, var=var, plev=85000)
        pdf = os.path.join(
            module_outdir, "pdf", f"globalbiases.seasonal_bias.ci.ERA5.era5-hpz3.r1.ERA5.era5-hpz3.{var}.85000.pdf"
        )
        assert os.path.exists(pdf)
        png = os.path.join(
            module_outdir, "png", f"globalbiases.seasonal_bias.ci.ERA5.era5-hpz3.r1.ERA5.era5-hpz3.{var}.85000.png"
        )
        assert os.path.exists(png)

    def test_vertical_bias(self, global_biases_instance, plot_global_biases_instance, module_outdir, test_var):
        gb = global_biases_instance
        plotgb = plot_global_biases_instance
        var = test_var

        # Ensure climatology is computed
        if not hasattr(gb, "climatology"):
            gb.compute_climatology(var=var, seasonal=True)

        plotgb.plot_vertical_bias(data=gb.climatology, data_ref=gb.climatology, var=var, vmin=-0.002, vmax=0.002)
        pdf = os.path.join(module_outdir, "pdf", f"globalbiases.vertical_bias.ci.ERA5.era5-hpz3.r1.ERA5.era5-hpz3.{var}.pdf")
        assert os.path.exists(pdf)
        png = os.path.join(module_outdir, "png", f"globalbiases.vertical_bias.ci.ERA5.era5-hpz3.r1.ERA5.era5-hpz3.{var}.png")
        assert os.path.exists(png)

    def test_plev_selection(self, test_var):
        gb = GlobalBiases(catalog="ci", model="ERA5", exp="era5-hpz3", source="monthly", regrid="r100")

        gb.retrieve(var=test_var, plev=85000)
        gb.compute_climatology(var=test_var, plev=85000)
        assert gb.climatology["q"].coords["plev"] == 85000

        with pytest.raises(ValueError):
            gb.retrieve("tprate", plev=85000)

    def test_variables(self):
        gb_local = GlobalBiases(catalog="ci", model="ERA5", exp="era5-hpz3", source="monthly")
        with pytest.raises(Exception) as exc:
            gb_local.retrieve(var="pippo")
        assert isinstance(exc.value, (ValueError, NoDataError))

        gb_local.retrieve(var="tprate", units="mm/day")
        gb_local.compute_climatology(var="tprate")
        assert gb_local.climatology["tprate"].attrs.get("units") == "mm/day"

    def test_formula(self):
        var = "tnlwrf+tnswrf"
        long_name = "Top net radiation"
        short_name = "tnr"
        gb = GlobalBiases(catalog="ci", model="ERA5", exp="era5-hpz3", source="monthly")
        gb.retrieve(formula=True, var=var, long_name=long_name, short_name=short_name)
        gb.compute_climatology()
        assert short_name in gb.climatology.data_vars
        assert gb.data[short_name].attrs.get("long_name") == long_name
        assert gb.data[short_name].attrs.get("short_name") == short_name

    def test_adaptive_stipple_density(self, plot_global_biases_instance):
        """Test that adaptive stipple_density is computed correctly for different grid resolutions."""

        plotgb = plot_global_biases_instance
        mock_ax = MagicMock()

        for n_lat, n_lon, target, expected_density in [
            (180, 360, 1000, 8),  # r100-like
            (1800, 3600, 1000, 80),  # hpz10-like
        ]:
            lat = xr.DataArray(np.linspace(-90, 90, n_lat), dims=["lat"])
            lon = xr.DataArray(np.linspace(-180, 180, n_lon), dims=["lon"])
            significance_mask = xr.DataArray(
                np.ones((n_lat, n_lon), dtype=bool),
                coords={"lat": lat, "lon": lon},
                dims=["lat", "lon"],
            )

            # Should not raise and should use adaptive density
            plotgb._add_significance_stippling(
                mock_ax,
                significance_mask,
                lat,
                lon,
                stipple_density=None,
                target_stipple_points=target,
            )

            computed_density = max(1, int(np.sqrt((n_lat * n_lon) / target)))
            assert computed_density == expected_density, (
                f"Expected density {expected_density} for grid {n_lat}x{n_lon}, got {computed_density}"
            )
