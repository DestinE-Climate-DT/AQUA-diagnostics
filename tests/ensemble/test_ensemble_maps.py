import numpy as np
import pytest
import xarray as xr

from aqua.diagnostics.ensemble.ensembleMaps import EnsembleMaps
from aqua.diagnostics.ensemble.plot_ensemble_maps import SEASONS, PlotEnsembleMaps

pytestmark = [pytest.mark.diagnostics, pytest.mark.ensemble]


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
def ensemble_maps_instance(ensemble_config, dataset_instance, tmp_path):
    """Create an EnsembleMaps instance using a real ensemble dimension."""
    return EnsembleMaps(
        var=ensemble_config["var"],
        dataset=dataset_instance,
        catalog_list=ensemble_config["catalog_list"],
        model_list=ensemble_config["model_list"],
        exp_list=ensemble_config["exp_list"],
        source_list=ensemble_config["source_list"],
        ensemble_dimension_name="ensemble",
        outputdir=str(tmp_path),
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

        expected_mean = f"{tmp_path_str}/ensemble_mean_{conf['var']}.nc"
        expected_std = f"{tmp_path_str}/ensemble_std_{conf['var']}.nc"

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


@pytest.fixture
def seasonal_dataset():
    """Three-member ensemble of a small seasonal climatology, members differing by 1 K."""
    lat = np.linspace(-85, 85, 18)
    lon = np.linspace(0, 350, 36)
    base = np.full((4, lat.size, lon.size), 280.0) + np.arange(4)[:, None, None]
    data = np.stack([base + member for member in range(3)])
    return xr.Dataset(
        {"2t": (("ensemble", "season", "lat", "lon"), data, {"units": "K"})},
        coords={"ensemble": ["r1", "r2", "r3"], "season": SEASONS, "lat": lat, "lon": lon},
    )


class TestEnsembleSeasonalMaps:
    def _run(self, dataset, outputdir):
        ens = EnsembleMaps(
            var="2t",
            dataset=dataset,
            catalog_list="ci",
            model_list="IFS",
            exp_list="test",
            source_list="short",
            diagnostic_product="EnsembleSeasonalMaps",
            outputdir=outputdir,
        )
        ens.run()
        return ens

    def test_statistics_keep_season(self, seasonal_dataset, tmp_path):
        ens = self._run(seasonal_dataset, str(tmp_path))

        assert ens.dataset_mean.dims == ("season", "lat", "lon")
        assert ens.dataset_mean.sel(season="JJA").values == pytest.approx(283.0)
        assert ens.dataset_std.values == pytest.approx(np.std([0, 1, 2]))

    def test_output_does_not_overwrite_annual(self, seasonal_dataset, tmp_path):
        self._run(seasonal_dataset, str(tmp_path))
        EnsembleMaps(
            var="2t",
            dataset=seasonal_dataset.isel(season=0, drop=True),
            catalog_list="ci",
            model_list="IFS",
            exp_list="test",
            source_list="short",
            outputdir=str(tmp_path),
        ).run()

        names = [path.name for path in tmp_path.rglob("*.nc")]
        assert len(names) == 4
        assert sum("ensembleseasonalmaps" in name for name in names) == 2

    @pytest.mark.parametrize("with_reference", [False, True])
    def test_plot_seasonal(self, seasonal_dataset, tmp_path, with_reference):
        ens = self._run(seasonal_dataset, str(tmp_path))
        reference = ens.dataset_mean - 1.0 if with_reference else None

        plot = PlotEnsembleMaps(
            diagnostic_product="EnsembleSeasonalMaps",
            catalog_list="ci",
            model_list="IFS",
            exp_list="test",
            source_list="short",
            outputdir=str(tmp_path),
        )
        fig = plot.plot_seasonal(
            var="2t",
            dataset=ens.dataset_mean,
            ref_dataset=reference,
            data_name="ensemble_mean",
            contour=False,
            save_format="png",
            dpi=50,
        )

        assert fig is not None
        assert len(list(tmp_path.rglob("*.png"))) == 1

    def test_plot_seasonal_without_data(self, tmp_path):
        assert PlotEnsembleMaps(outputdir=str(tmp_path)).plot_seasonal(var="2t", dataset=None) is None
