"""Tests for the SeaIce CLI (parse_arguments + main orchestration)."""

from unittest.mock import MagicMock

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

# Two references: the first matches the method, the second is tagged for a
# different method so the CLI 'use_for_method' skip branch is exercised.
REFERENCES_TWO = [
    {"catalog": "ref1", "model": "R1", "exp": "e1", "source": "s1", "use_for_method": "fraction", "domain": "nh"},
    {"catalog": "ref2", "model": "R2", "exp": "e2", "source": "s2", "use_for_method": "thickness", "domain": "nh"},
]

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

    def test_timeseries_with_references_and_realization(self, build_config, mock_cluster, mock_si, mocker):
        """
        Enable the timeseries block with references + calc_ref_std and pass
        --realization on the CLI. This covers the references loop (including
        the use_for_method skip branch) and the realization reader_kwargs branch.
        """
        mock_seaice_cls, mock_plot_cls, _ = mock_si
        # calc_ref_std=True makes compute_seaice return (data, std) for refs
        mock_seaice_cls.return_value.compute_seaice.return_value = (MagicMock(), MagicMock())
        mocker.patch(f"{CLI_MODULE}.filter_region_list", return_value=["arctic"])

        config_file = build_config(
            {
                "seaice_timeseries": {
                    **BASE_SET,
                    "calc_ref_std": True,
                    "ref_std_freq": "monthly",
                    "references": REFERENCES_TWO,
                },
            }
        )

        main(["--config", config_file, "--loglevel", "WARNING", "--realization", "r1"])

        mock_plot_cls.return_value.plot_seaice.assert_called_once()
        # Only the matching reference (use_for_method='fraction') gets its
        # compute_seaice invoked; the thickness one is skipped.
        # Calls: 1 dataset + 1 matching ref = 2
        assert mock_seaice_cls.return_value.compute_seaice.call_count == 2

    def test_seasonal_cycle_with_references(self, build_config, mock_cluster, mock_si, mocker):
        """Enable the seasonal-cycle block with references + calc_ref_std."""
        mock_seaice_cls, mock_plot_cls, _ = mock_si
        mock_seaice_cls.return_value.compute_seaice.return_value = (MagicMock(), MagicMock())
        mocker.patch(f"{CLI_MODULE}.filter_region_list", return_value=["arctic"])

        config_file = build_config(
            {
                "seaice_seasonal_cycle": {
                    **BASE_SET,
                    "calc_ref_std": True,
                    "ref_std_freq": "monthly",
                    "references": REFERENCES_TWO,
                },
            }
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_plot_cls.return_value.plot_seaice.assert_called_once()
        plot_call = mock_plot_cls.return_value.plot_seaice.call_args
        assert plot_call.kwargs["plot_type"] == "seasonalcycle"

    def test_2d_bias_with_references(self, build_config, mock_cluster, mock_si, mocker):
        """Enable the 2D bias block with references (no calc_ref_std in this block)."""
        mock_seaice_cls, _, mock_plot_2d_cls = mock_si
        mocker.patch(f"{CLI_MODULE}.filter_region_list", return_value=["arctic"])

        config_file = build_config(
            {
                "seaice_2d_bias": {**BASE_2D_SET, "references": REFERENCES_TWO},
            }
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_plot_2d_cls.return_value.plot_2d_seaice.assert_called_once()
        # 1 dataset + 1 matching reference = 2 compute_seaice calls
        assert mock_seaice_cls.return_value.compute_seaice.call_count == 2
