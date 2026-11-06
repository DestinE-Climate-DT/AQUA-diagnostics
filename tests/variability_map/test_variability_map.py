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
def variability_map_instance():
    """Create a fresh VariabilityMap instance for each test."""
    vm = VariabilityMap(
        catalog="ci",
        model="ERA5",
        exp="era5-hpz3",
        source="monthly",
        regrid="r100",
        var="2t",
    )
    return vm


@pytest.fixture
def plot_variability_map_instance(tmp_path):
    """Create a PlotVariabilityMap instance."""
    return PlotVariabilityMap(outputdir=str(tmp_path))


@pytest.fixture
def test_var():
    return "2t"


class TestVariabilityMap:
    def test_variability_map(self, variability_map_instance, tmp_path):
        """Test the variability map calculation."""
        vm = variability_map_instance
        outdir = str(tmp_path)

        vm.run(outputdir=str(tmp_path))

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

    def test_variability_map_values(self, variability_map_instance, tmp_path):
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

        vm.run(outputdir=str(tmp_path))

        xr.testing.assert_allclose(
            vm.data_std,
            expected,
        )

    def test_variability_map_metadata(self, variability_map_instance, tmp_path):
        """Test metadata of the variability map."""
        vm = variability_map_instance

        vm.run(outputdir=str(tmp_path))

        assert vm.data_std.attrs.get("short_name") == "2t"
        assert "long_name" in vm.data_std.attrs
        assert "units" in vm.data_std.attrs

    def test_units_conversion(self):
        """Units given to the class are applied at retrieve."""
        vm = VariabilityMap(catalog="ci", model="ERA5", exp="era5-hpz3", source="monthly", var="2t", units="degC")
        vm.retrieve()
        assert vm.data.attrs["units"] == "degC"
        assert float(vm.data.max()) < 100

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
        tmp_path,
    ):
        """Test the variability map plot."""
        vm = variability_map_instance
        plotvm = plot_variability_map_instance
        outdir = str(tmp_path)

        vm.run(outputdir=str(tmp_path))

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
        tmp_path,
    ):
        """Test the variability map plot with fixed limits."""
        vm = variability_map_instance
        plotvm = plot_variability_map_instance
        outdir = str(tmp_path)

        vm.run(outputdir=str(tmp_path))

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
        tmp_path,
    ):
        """Test the variability map difference plot."""
        vm = variability_map_instance
        plotvm = plot_variability_map_instance
        outdir = str(tmp_path)

        vm.run(outputdir=str(tmp_path))

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
