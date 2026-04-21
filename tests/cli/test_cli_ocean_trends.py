"""Tests for the Ocean Trends CLI (parse_arguments + main orchestration)."""

import pytest

from aqua.diagnostics.ocean_trends.cli_ocean_trends import main, parse_arguments

CLI_MODULE = "aqua.diagnostics.ocean_trends.cli_ocean_trends"

BASE_BLOCK = {
    "run": True,
    "regions": ["global_ocean", "atlantic_ocean"],
    "diagnostic_name": "ocean_trends",
    "var": ["thetao"],
    "dim_mean": ["lat", "lon"],
    "vert_coord": "lev",
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
        config_file = build_config({"ocean_trends": {"multilevel": {"run": False}}})

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
        config_file = build_config({"ocean_trends": {"multilevel": BASE_BLOCK}})

        main(["--config", config_file, "--loglevel", "WARNING"])

        inst = mock_trends_cls.return_value
        inst.run.assert_called_once()
        assert inst.select_region.call_count == 2

        # 2 regions * 2 PlotTrends instances each
        assert mock_plot_cls.call_count == 4
        assert mock_plot_cls.return_value.plot_multilevel.call_count == 2
        assert mock_plot_cls.return_value.plot_zonal.call_count == 2
