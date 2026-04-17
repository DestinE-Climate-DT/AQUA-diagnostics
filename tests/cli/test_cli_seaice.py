"""Tests for the SeaIce CLI (parse_arguments + main orchestration)."""

import pytest

from aqua.diagnostics.seaice.cli_seaice import main, parse_arguments

CLI_MODULE = "aqua.diagnostics.seaice.cli_seaice"

# Minimal block used by the timeseries / seasonal-cycle branches.
# References are intentionally omitted to keep the test surface small.
BASE_SET = {
    "run": True,
    "methods": ["fraction"],
    "regions": ["arctic"],
    "varname": {"fraction": "siconc"},
}

BASE_2D_SET = {
    "run": True,
    "methods": ["fraction"],
    "regions": ["arctic"],
    "months": [3, 9],
    "varname": {"fraction": "siconc"},
    "projections": {"orthographic": {"central_longitude": 0.0, "central_latitude": 90.0}},
}

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]


# ======================================================================
# Argument parsing
# ======================================================================
def test_parse_arguments_cli_options():
    """
    Verify parse_arguments parses CLI options, including the seaice-specific --proj.
    Detailed flag coverage lives in test_base_util.py
    """
    args = parse_arguments(["--model", "IFS", "--proj", "azimuthal_equidistant"])
    assert args.model == "IFS"
    assert args.proj == "azimuthal_equidistant"
    assert args.catalog is None

    with pytest.raises(SystemExit):
        parse_arguments(["--help"])


# ======================================================================
# CLI execution flow (main)
# ======================================================================
class TestMainExecutionFlow:
    """Test main() execution flow with mocked SeaIce, PlotSeaIce and Plot2DSeaIce."""

    @pytest.fixture
    def mock_si(self, mocker):
        """
        Patch SeaIce, PlotSeaIce and Plot2DSeaIce at the CLI module path.
        Returns (mock_seaice_cls, mock_plot_cls, mock_plot_2d_cls).

        Note: cli_seaice.main() always instantiates SeaIce once at startup to
        call _load_regions_from_file, regardless of whether any diagnostic
        block is enabled. Tests should therefore assert on the plot classes
        to check whether a specific block actually ran.
        """
        mock_seaice_cls = mocker.patch(f"{CLI_MODULE}.SeaIce")
        mock_plot_cls = mocker.patch(f"{CLI_MODULE}.PlotSeaIce")
        mock_plot_2d_cls = mocker.patch(f"{CLI_MODULE}.Plot2DSeaIce")
        return mock_seaice_cls, mock_plot_cls, mock_plot_2d_cls

    def test_all_diagnostics_disabled_skip_processing(self, build_config, mock_cluster, mock_si):
        """When every seaice block has run=False, no plotting class is instantiated."""
        mock_seaice_cls, mock_plot_cls, mock_plot_2d_cls = mock_si
        config_file = build_config(
            {
                "seaice_timeseries": {"run": False},
                "seaice_seasonal_cycle": {"run": False},
                "seaice_2d_bias": {"run": False},
            }
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        # SeaIce is still called once for the top-level _load_regions_from_file.
        assert mock_seaice_cls.call_count == 1
        mock_plot_cls.assert_not_called()
        mock_plot_2d_cls.assert_not_called()

    def test_timeseries_full_pipeline(self, build_config, mock_cluster, mock_si):
        """
        With seaice_timeseries enabled, verify the full execution flow:
        SeaIce.compute_seaice + save_netcdf are called for the dataset,
        PlotSeaIce is created and plot_seaice('timeseries') is called.
        """
        mock_seaice_cls, mock_plot_cls, _ = mock_si
        mock_seaice_instance = mock_seaice_cls.return_value
        config_file = build_config({"seaice_timeseries": BASE_SET})

        main(["--config", config_file, "--loglevel", "WARNING"])

        # compute_seaice called once per (method, dataset) = 1 * 1 = 1
        assert mock_seaice_instance.compute_seaice.call_count == 1
        compute_call = mock_seaice_instance.compute_seaice.call_args
        assert compute_call.kwargs["method"] == "fraction"
        assert compute_call.kwargs["var"] == "siconc"

        mock_seaice_instance.save_netcdf.assert_called()
        mock_plot_cls.assert_called_once()
        mock_plot_cls.return_value.plot_seaice.assert_called_once()
        plot_call = mock_plot_cls.return_value.plot_seaice.call_args
        assert plot_call.kwargs["plot_type"] == "timeseries"

    def test_seasonal_cycle_full_pipeline(self, build_config, mock_cluster, mock_si):
        """When seaice_seasonal_cycle is enabled, plot_seaice is called with 'seasonalcycle'."""
        mock_seaice_cls, mock_plot_cls, _ = mock_si
        mock_seaice_instance = mock_seaice_cls.return_value
        config_file = build_config({"seaice_seasonal_cycle": BASE_SET})

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_seaice_instance.compute_seaice.assert_called_once()
        compute_call = mock_seaice_instance.compute_seaice.call_args
        assert compute_call.kwargs["get_seasonal_cycle"] is True

        mock_plot_cls.assert_called_once()
        plot_call = mock_plot_cls.return_value.plot_seaice.call_args
        assert plot_call.kwargs["plot_type"] == "seasonalcycle"

    def test_2d_bias_full_pipeline(self, build_config, mock_cluster, mock_si):
        """When seaice_2d_bias is enabled, Plot2DSeaIce.plot_2d_seaice is called."""
        mock_seaice_cls, _, mock_plot_2d_cls = mock_si
        mock_seaice_instance = mock_seaice_cls.return_value
        config_file = build_config({"seaice_2d_bias": BASE_2D_SET})

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_seaice_instance.compute_seaice.assert_called_once()
        mock_plot_2d_cls.assert_called_once()
        plot_call = mock_plot_2d_cls.return_value.plot_2d_seaice.call_args
        assert plot_call.kwargs["plot_type"] == "bias"
        assert plot_call.kwargs["method"] == "fraction"
        assert plot_call.kwargs["months"] == [3, 9]
