import glob
import os

import pytest
import xarray as xr

from aqua.diagnostics import PlotVariabilityMap, VariabilityMap

pytestmark = [
    pytest.mark.diagnostics,
    pytest.mark.xdist_group(name="diagnostic_setup_class"),
]

@pytest.fixture
def variability_map_instance(tmp_path):
    """Create a fresh VariabilityMap instance for each test."""
    outputdir = str(tmp_path)

    return VariabilityMap(
        catalog="ci",
        model="ERA5",
        exp="era5-hpz3",
        source="monthly",
        regrid="r100",
        outputdir=outputdir,
        save_netcdf=True,
        var="2t",
    )

@pytest.fixture
def plot_variability_map_instance(variability_map_instance):
    """Create a PlotVariabilityMap instance."""
    plotvm = PlotVariabilityMap(
        outputdir=variability_map_instance.outputdir,
    )
    # Assign attributes directly to satisfy the OutputSaver
    plotvm.catalog = variability_map_instance.catalog
    plotvm.model = variability_map_instance.model
    plotvm.exp = variability_map_instance.exp
    return plotvm

@pytest.fixture
def test_var():
    return "2t"


class TestVariabilityMap:

    def test_variability_map(self, variability_map_instance):
        """Test the variability map calculation."""
        vm = variability_map_instance
        outdir = vm.outputdir

        vm.run()

        assert isinstance(vm.data_std, xr.DataArray)
        assert vm.data_std.name == "2t"

        assert "time" not in vm.data_std.dims
        assert "lat" in vm.data_std.dims
        assert "lon" in vm.data_std.dims

        # Standard deviation must be non-negative.
        # NaN values are allowed.
        valid = vm.data_std.dropna(dim="lat", how="all")
        valid = valid.dropna(dim="lon", how="all")

        assert float(valid.min()) >= 0.0

        # Check that the NetCDF was generated
        nc_files = glob.glob(os.path.join(outdir, "netcdf", "*.nc"))
        assert len(nc_files) > 0, "NetCDF file was not saved."

    def test_variability_map_values(self, variability_map_instance):
        """Test that the calculated standard deviation is correct."""
        vm = variability_map_instance

        expected_vm = VariabilityMap(
            catalog="ci",
            model="ERA5",
            exp="era5-hpz3",
            source="monthly",
            regrid="r100",
            var="2t",
        )

        expected_vm.retrieve()

        expected = expected_vm.data.std(
            dim="time",
            skipna=True,
        ).compute()

        vm.run()

        xr.testing.assert_allclose(
            vm.data_std,
            expected,
        )

    def test_variability_map_metadata(self, variability_map_instance):
        """Test metadata of the variability map."""
        vm = variability_map_instance

        vm.run()

        assert vm.data_std.attrs.get("short_name") == "2t"
        assert "long_name" in vm.data_std.attrs
        assert "units" in vm.data_std.attrs

    def test_variables(self, test_var):
        """Test variable selection during retrieval."""
        vm = VariabilityMap(
            catalog="ci",
            model="ERA5",
            exp="era5-hpz3",
            source="monthly",
            var=test_var,
        )

        vm.retrieve()

        assert isinstance(vm.data, xr.DataArray)
        assert vm.data.name == test_var


class TestPlotVariabilityMap:

    def test_plot(
        self,
        variability_map_instance,
        plot_variability_map_instance,
        test_var,
    ):
        """Test the variability map plot."""
        vm = variability_map_instance
        plotvm = plot_variability_map_instance
        outdir = vm.outputdir

        vm.run()

        plotvm.plot(
            dataset_std=vm.data_std,
            var=test_var,
            catalog="ci",
            model="ERA5",
            exp="era5-hpz3",
            startdate=vm.startdate,
            enddate=vm.enddate,
            tgt_grid_name=None,
            proj="plate_carree",
        )

        pdf_files = glob.glob(os.path.join(outdir, "pdf", "*.pdf"))
        png_files = glob.glob(os.path.join(outdir, "png", "*.png"))

        assert len(pdf_files) > 0, "PDF plot was not saved."
        assert len(png_files) > 0, "PNG plot was not saved."

    def test_plot_with_limits(
        self,
        variability_map_instance,
        plot_variability_map_instance,
        test_var,
    ):
        """Test the variability map plot with fixed limits."""
        vm = variability_map_instance
        plotvm = plot_variability_map_instance
        outdir = vm.outputdir

        vm.run()

        plotvm.plot(
            dataset_std=vm.data_std,
            var=test_var,
            catalog="ci",
            model="ERA5",
            exp="era5-hpz3",
            startdate=vm.startdate,
            enddate=vm.enddate,
            vmin=0.0,
            vmax=10.0,
            tgt_grid_name=None,
            proj="plate_carree",
        )

        pdf_files = glob.glob(os.path.join(outdir, "pdf", "*.pdf"))
        png_files = glob.glob(os.path.join(outdir, "png", "*.png"))

        assert len(pdf_files) > 0, "PDF plot was not saved."
        assert len(png_files) > 0, "PNG plot was not saved."

    def test_plot_diff(
        self,
        variability_map_instance,
        plot_variability_map_instance,
        test_var,
    ):
        """Test the variability map difference plot."""
        vm = variability_map_instance
        plotvm = plot_variability_map_instance
        outdir = vm.outputdir

        vm.run()

        plotvm.plot_diff(
            dataset_std=vm.data_std,
            dataset_std_ref=vm.data_std,
            var=test_var,
            catalog="ci",
            model="ERA5",
            exp="era5-hpz3",
            catalog_ref="ci",
            model_ref="ERA5",
            exp_ref="era5-hpz3",
            startdate=vm.startdate,
            enddate=vm.enddate,
            startdate_ref=vm.startdate,
            enddate_ref=vm.enddate,
            tgt_grid_name=None,
            proj="plate_carree",
        )

        pdf_files = glob.glob(os.path.join(outdir, "pdf", "*.pdf"))
        png_files = glob.glob(os.path.join(outdir, "png", "*.png"))

        assert len(pdf_files) > 0, "PDF diff plot was not saved."
        assert len(png_files) > 0, "PNG diff plot was not saved."
