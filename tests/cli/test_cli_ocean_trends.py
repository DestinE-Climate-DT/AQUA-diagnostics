"""Tests for the Ocean Trends CLI (parse_arguments + main orchestration)."""

import pytest

from aqua.diagnostics.ocean_trends.cli_ocean_trends import main, parse_arguments

CLI_MODULE = "aqua.diagnostics.ocean_trends.cli_ocean_trends"

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
    """Test main() execution flow with mocked Trends and PlotTrends."""

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

    @pytest.mark.parametrize(
        "output, runs, saves, loads",
        [
            ({}, True, True, True),  # compute, save and plot from the file just written
            ({"save_netcdf": False}, True, False, False),  # compute and plot from memory only
            ({"plot_only": True}, False, None, True),  # no evaluation, plot from a previous file
        ],
    )
    def test_plot_only_and_save_netcdf(self, build_config, mock_cluster, mock_ot, output, runs, saves, loads):
        """plot_only skips the evaluation, save_netcdf only decides whether the results are written."""
        mock_trends_cls, mock_plot_cls = mock_ot
        inst = mock_trends_cls.return_value
        config_file = build_config({"ocean_trends": BASE_OT}, output_overrides=output)

        main(["--config", config_file, "--loglevel", "WARNING"])

        assert inst.run.call_count == (1 if runs else 0)
        for call in inst.run.call_args_list:
            assert call.kwargs["save_netcdf"] is saves
        # Loading unsaved results would replace them with an older file on disk
        assert inst.load.call_count == (1 if loads else 0)
        # 2 regions * 2 PlotTrends instances each, whatever the source of the results
        assert mock_plot_cls.call_count == 4

    def test_plot_only_without_results(self, build_config, mock_cluster, mock_ot):
        """With no file to plot from, the plots are skipped instead of failing region by region."""
        mock_trends_cls, mock_plot_cls = mock_ot
        mock_trends_cls.return_value.trend_coef = None
        config_file = build_config({"ocean_trends": BASE_OT}, output_overrides={"plot_only": True})

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_trends_cls.return_value.select_region.assert_not_called()
        mock_plot_cls.assert_not_called()
