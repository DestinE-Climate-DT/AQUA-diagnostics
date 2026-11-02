"""Tests for the Ocean Stratification CLI (parse_arguments + main orchestration)."""

import pytest

from aqua.diagnostics.ocean_stratification.cli_ocean_stratification import main, parse_arguments

CLI_MODULE = "aqua.diagnostics.ocean_stratification.cli_ocean_stratification"

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


class TestMainExecutionFlow:
    """Test main() execution flow with mocked Stratification and plot classes."""

    @pytest.fixture
    def mock_os(self, mocker):
        mocks = {
            "Stratification": mocker.patch(f"{CLI_MODULE}.Stratification"),
            "PlotStratification": mocker.patch(f"{CLI_MODULE}.PlotStratification"),
            "PlotMLD": mocker.patch(f"{CLI_MODULE}.PlotMLD"),
        }
        # Allow data[["thetao", "so", "rho"]] and data[["mld"]] in CLI.
        data = mocker.MagicMock()
        mocks["Stratification"].return_value.data = data
        mocks["Stratification"].return_value.processed_data = {("global_ocean", "annual"): data}
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
        mock_os["PlotStratification"].return_value.plot_stratification.assert_called_once()
        mock_os["PlotMLD"].return_value.plot_mld.assert_called_once()

    @pytest.mark.parametrize(
        "output, runs, saves, loads",
        [
            ({}, True, True, True),  # compute, save and plot from the files just written
            ({"save_netcdf": False}, True, False, False),  # compute and plot from memory only
            ({"plot_only": True}, False, None, True),  # no evaluation, plot from previous files
        ],
    )
    def test_plot_only_and_save_netcdf(self, build_config, mock_cluster, mock_os, output, runs, saves, loads):
        """plot_only skips the evaluation, save_netcdf only decides whether the results are written."""
        config_file = build_config(
            {"ocean_stratification": {"stratification": BASE_STRAT["stratification"], "mld": BASE_STRAT["stratification"]}},
            output_overrides=output,
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        inst = mock_os["Stratification"].return_value
        # model + reference, for both the stratification and the MLD sections
        assert inst.run.call_count == (4 if runs else 0)
        for call in inst.run.call_args_list:
            assert call.kwargs["save_netcdf"] is saves
        # Loading unsaved results would replace them with older files on disk
        assert inst.load.call_count == (4 if loads else 0)
        # load needs the climatologies too, to address the files run writes
        for call in inst.load.call_args_list:
            assert call.kwargs["climatology"] == ["annual"]
        mock_os["PlotStratification"].return_value.plot_stratification.assert_called_once()
        mock_os["PlotMLD"].return_value.plot_mld.assert_called_once()

    def test_plot_only_without_reference_results(self, build_config, mock_cluster, mock_os, mocker):
        """A region with no reference file is plotted without the reference, instead of stopping the CLI."""
        model, obs = mocker.MagicMock(), mocker.MagicMock()
        model.processed_data = {("global_ocean", "annual"): mocker.MagicMock()}
        obs.processed_data = {}
        mock_os["Stratification"].side_effect = [model, obs]
        config_file = build_config(
            {"ocean_stratification": {"stratification": BASE_STRAT["stratification"]}},
            output_overrides={"plot_only": True},
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        assert mock_os["PlotStratification"].call_args.kwargs["obs"] is None
        mock_os["PlotStratification"].return_value.plot_stratification.assert_called_once()
