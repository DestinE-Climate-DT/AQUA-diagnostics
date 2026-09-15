import pytest
import xarray as xr

from aqua.diagnostics.ensemble.ensembleMaps import EnsembleMaps
from aqua.diagnostics.ensemble.plot_ensemble_maps import PlotEnsembleMaps


@pytest.fixture(scope="module")
def ensemble_config():
    return {
        "var": "2t",
        "catalog_list": ["ci", "ci"],
        "model_list": ["IFS", "IFS"],
        "exp_list": ["test-tco79", "test-tco79"],
        "source_list": ["short", "short"],
    }


@pytest.fixture
def tmp_path_str(tmp_path):
    return str(tmp_path)


@pytest.fixture
def dataset_instance(ifs_tco79_short_data_2t):
    """Create a two-member ensemble from the existing IFS test dataset."""
    return xr.concat(
        [ifs_tco79_short_data_2t, ifs_tco79_short_data_2t],
        dim="ensemble",
    ).assign_coords(ensemble=[0, 1])


@pytest.fixture
def ensemble_maps_instance(ensemble_config, dataset_instance):
    """Create an EnsembleMaps instance using a real ensemble dimension."""
    return EnsembleMaps(
        var=ensemble_config["var"],
        dataset=dataset_instance,
        catalog_list=ensemble_config["catalog_list"],
        model_list=ensemble_config["model_list"],
        exp_list=ensemble_config["exp_list"],
        source_list=ensemble_config["source_list"],
        ensemble_dimension_name="ensemble",
    )


@pytest.fixture
def plot_ensemble_instance():
    return PlotEnsembleMaps()


class TestEnsembleMaps:

    def test_initialization(self, dataset_instance):
        assert dataset_instance is not None
        assert isinstance(dataset_instance, xr.Dataset)
        assert "ensemble" in dataset_instance.dims

    def test_run(self, ensemble_maps_instance, ensemble_config, tmp_path_str):
        """Test the computation and NetCDF output generation."""
        ens = ensemble_maps_instance
        ens.outputdir = tmp_path_str
        conf = ensemble_config

        ens.run()

        assert ens.dataset_mean is not None
        assert ens.dataset_std is not None

        expected_mean = (
            f"{tmp_path_str}/ensemble_mean_{conf['var']}.nc"
        )
        expected_std = (
            f"{tmp_path_str}/ensemble_std_{conf['var']}.nc"
        )

        assert expected_mean
        assert expected_std

    def test_statistics(self, ensemble_maps_instance):
        """Test the statistical correctness of the ensemble."""
        ens = ensemble_maps_instance

        ens.run()

        assert ens.dataset_mean is not None
        assert ens.dataset_std is not None

        # Both ensemble members contain identical data, so std should be zero.
        assert ens.dataset_std.all() == 0

    def test_plotting(
        self,
        ensemble_maps_instance,
        plot_ensemble_instance,
        ensemble_config,
        tmp_path_str,
    ):
        """Test the plotting functionality."""
        ens = ensemble_maps_instance
        plot_ens = plot_ensemble_instance
        plot_ens.outputdir = tmp_path_str
        conf = ensemble_config

        ens.run()

        fig_m, ax_m = plot_ens.plot(
            var=conf["var"],
            dpi=50,
            save_format=("png", "pdf"),
            title="Test data Mean",
            cbar_label="Test Label",
            dataset=ens.dataset_mean,
            data_name="mean",
        )

        assert fig_m is not None
        assert ax_m is not None
