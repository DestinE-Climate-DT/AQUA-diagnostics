"""Tests for the Ocean3D CLI (parse_arguments + section orchestration)."""

import pytest

from aqua.diagnostics.ocean3d.cli_ocean3d import main, parse_arguments

CLI_MODULE = "aqua.diagnostics.ocean3d.cli_ocean3d"

BASE_DRIFT = {
    "hovmoller": {
        "run": True,
        "regions": ["global_ocean"],
        "diagnostic_name": "ocean_drift",
        "var": ["thetao"],
        "dim_mean": ["lat", "lon"],
        "vert_coord": "lev",
    }
}

BASE_OT = {
    "multilevel": {
        "run": True,
        "regions": ["global_ocean", "atlantic_ocean"],
        "diagnostic_name": "ocean_trends",
        "var": ["thetao"],
        "dim_mean": ["lat", "lon"],
        "vert_coord": "lev",
    }
}

BASE_STRAT = {
    "stratification": {
        "run": True,
        "regions": ["global_ocean"],
        "climatology": ["annual"],
        "diagnostic_name": "ocean_stratification",
        "var": ["thetao", "so"],
        "vert_coord": "lev",
    }
}

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]


def test_parse_arguments_cli_options():
    """Verify parse_arguments parses CLI options."""
    args = parse_arguments(["--model", "IFS", "--nworkers", "2"])
    assert args.model == "IFS"
    assert args.nworkers == 2
    assert args.catalog is None

    with pytest.raises(SystemExit):
        parse_arguments(["--help"])


class TestHovmoller:
    """Test main() for ocean_drift.hovmoller with mocked Hovmoller / PlotHovmoller."""

    @pytest.fixture
    def mock_od(self, mocker):
        mock_hov_cls = mocker.patch(f"{CLI_MODULE}.Hovmoller")
        mock_plot_cls = mocker.patch(f"{CLI_MODULE}.PlotHovmoller")
        return mock_hov_cls, mock_plot_cls

    def test_hovmoller_disabled_skips_processing(self, build_config, mock_cluster, mock_od):
        """When run=False, diagnostic and plot classes are not instantiated."""
        mock_hov_cls, mock_plot_cls = mock_od
        config_file = build_config({"ocean_drift": {"hovmoller": {**BASE_DRIFT["hovmoller"], "run": False}}})

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_hov_cls.assert_not_called()
        mock_plot_cls.assert_not_called()

    def test_hovmoller_full_pipeline(self, build_config, mock_cluster, mock_od):
        """With run=True, Hovmoller.run and both plotting methods are called."""
        mock_hov_cls, mock_plot_cls = mock_od
        mock_hov_instance = mock_hov_cls.return_value
        mock_hov_instance.processed_data = {"global_ocean": [object()]}
        config_file = build_config({"ocean_drift": BASE_DRIFT})

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_hov_cls.assert_called_once()
        mock_hov_instance.run.assert_called_once()
        run_call = mock_hov_instance.run.call_args
        assert run_call.kwargs["regions"] == ["global_ocean"]
        assert run_call.kwargs["var"] == ["thetao"]

        mock_plot_cls.assert_called_once()
        mock_plot_cls.return_value.plot_hovmoller.assert_called_once()
        mock_plot_cls.return_value.plot_timeseries.assert_called_once()


class TestTrends:
    """Test main() for ocean_trends.multilevel with mocked Trends / PlotTrends."""

    @pytest.fixture
    def mock_ot(self, mocker):
        mock_trends_cls = mocker.patch(f"{CLI_MODULE}.Trends")
        mock_plot_cls = mocker.patch(f"{CLI_MODULE}.PlotTrends")
        inst = mock_trends_cls.return_value
        inst.trend_coef = mocker.MagicMock()
        region_data = mocker.MagicMock()
        region_data.mean.return_value = mocker.MagicMock()
        inst.select_region.side_effect = [
            (region_data, "global_ocean"),
            (region_data, "atlantic_ocean"),
        ]
        return mock_trends_cls, mock_plot_cls

    def test_trends_disabled_skips_processing(self, build_config, mock_cluster, mock_ot):
        """When run=False, diagnostic and plot classes are not instantiated."""
        mock_trends_cls, mock_plot_cls = mock_ot
        config_file = build_config({"ocean_trends": {"multilevel": {**BASE_OT["multilevel"], "run": False}}})

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_trends_cls.assert_not_called()
        mock_plot_cls.assert_not_called()

    def test_trends_full_pipeline(self, build_config, mock_cluster, mock_ot):
        """
        With run=True and two regions:
        - Trends.run is called once on full dataset
        - select_region is called once per region
        - PlotTrends is instantiated twice per region (multilevel + zonal).
        """
        mock_trends_cls, mock_plot_cls = mock_ot
        config_file = build_config({"ocean_trends": BASE_OT})

        main(["--config", config_file, "--loglevel", "WARNING"])

        inst = mock_trends_cls.return_value
        assert inst.run.call_count == 1
        assert inst.select_region.call_count == 2

        # 2 regions * 2 PlotTrends instances each
        assert mock_plot_cls.call_count == 4
        assert mock_plot_cls.return_value.plot_multilevel.call_count == 2
        assert mock_plot_cls.return_value.plot_zonal.call_count == 2


class TestStratification:
    """Test main() for ocean_stratification with mocked Stratification / plot classes."""

    @pytest.fixture
    def mock_os(self, mocker):
        mocks = {
            "Stratification": mocker.patch(f"{CLI_MODULE}.Stratification"),
            "PlotStratification": mocker.patch(f"{CLI_MODULE}.PlotStratification"),
            "PlotMLD": mocker.patch(f"{CLI_MODULE}.PlotMLD"),
        }
        # Allow data[["thetao", "so", "rho"]] / data[["mld"]] and processed_data iteration.
        mock_data = mocker.MagicMock()
        mocks["Stratification"].return_value.data = mock_data
        mocks["Stratification"].return_value.processed_data = {"global_ocean": mock_data}
        return mocks

    def test_stratification_disabled_skips_processing(self, build_config, mock_cluster, mock_os):
        """When run=False, diagnostic and plot classes are not instantiated."""
        config_file = build_config({"ocean_stratification": {"stratification": {"run": False}}})

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_os["Stratification"].assert_not_called()
        mock_os["PlotStratification"].assert_not_called()
        mock_os["PlotMLD"].assert_not_called()

    def test_stratification_full_pipeline(self, build_config, mock_cluster, mock_os):
        """
        With run=True and one reference, per-region flow creates:
        - 4 Stratification runs (model+obs for stratification and MLD)
        - 1 PlotStratification and 1 PlotMLD call.
        """
        config_file = build_config(
            {"ocean_stratification": {"stratification": BASE_STRAT["stratification"], "mld": BASE_STRAT["stratification"]}}
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        strat_cls = mock_os["Stratification"]
        assert strat_cls.call_count == 4
        assert strat_cls.return_value.run.call_count == 4
        # Stratification NetCDF path passes region and climatology lists together.
        strat_run_calls = [c for c in strat_cls.return_value.run.call_args_list if c.kwargs.get("mld") is False]
        assert strat_run_calls
        assert strat_run_calls[0].kwargs["regions"] == ["global_ocean"]
        assert strat_run_calls[0].kwargs["climatology"] == ["annual"]
        mock_os["PlotStratification"].return_value.plot_stratification.assert_called_once()
        mock_os["PlotMLD"].return_value.plot_mld.assert_called_once()
