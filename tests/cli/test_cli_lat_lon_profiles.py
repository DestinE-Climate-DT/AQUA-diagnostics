"""Tests for the LatLonProfiles CLI (parse_arguments + main orchestration)."""

import pytest

from aqua.diagnostics.lat_lon_profiles.cli_lat_lon_profiles import main, parse_arguments

CLI_MODULE = "aqua.diagnostics.lat_lon_profiles.cli_lat_lon_profiles"

# Enabling both longterm and seasonal exercises the two branches of the
# internal _create_plot helper (one PlotLatLonProfiles call per mode).
BASE_DICT = {
    "run": True,
    "mean_type": "zonal",
    "longterm": True,
    "seasonal": True,
    "compute_std": False,
    "variables": ["2t"],
    "formulae": [],
}

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]


# ======================================================================
# Argument parsing
# ======================================================================
def test_parse_arguments_cli_options():
    """
    Verify parse_arguments parses CLI options.
    Detailed flag coverage lives in test_base_util.py
    """
    args = parse_arguments(["--model", "IFS", "--nworkers", "2"])
    assert args.model == "IFS"
    assert args.nworkers == 2
    assert args.catalog is None

    with pytest.raises(SystemExit):
        parse_arguments(["--help"])


# ======================================================================
# CLI execution flow (main)
# ======================================================================
class TestMainExecutionFlow:
    """Test main() execution flow with mocked LatLonProfiles and PlotLatLonProfiles."""

    @pytest.fixture
    def mock_llp(self, mocker):
        """
        Patch LatLonProfiles and PlotLatLonProfiles at the CLI module path.
        Returns (mock_llp_cls, mock_plot_cls).
        """
        mock_llp_cls = mocker.patch(f"{CLI_MODULE}.LatLonProfiles")
        mock_plot_cls = mocker.patch(f"{CLI_MODULE}.PlotLatLonProfiles")
        return mock_llp_cls, mock_plot_cls

    def test_diagnostic_disabled_skips_processing(self, build_config, mock_cluster, mock_llp):
        """When run=False, no LatLonProfiles instance is created."""
        mock_llp_cls, mock_plot_cls = mock_llp
        config_file = build_config({"lat_lon_profiles": {"run": False}})

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_llp_cls.assert_not_called()
        mock_plot_cls.assert_not_called()

    def test_full_pipeline_longterm_and_seasonal(self, build_config, mock_cluster, mock_llp):
        """
        With longterm + seasonal enabled, verify LatLonProfiles is created
        per dataset and reference, run() is called, and PlotLatLonProfiles
        is invoked once for longterm and once for seasonal.
        """
        mock_llp_cls, mock_plot_cls = mock_llp
        mock_llp_instance = mock_llp_cls.return_value
        config_file = build_config({"lat_lon_profiles": BASE_DICT})

        main(["--config", config_file, "--loglevel", "WARNING"])

        # 1 variable * (1 dataset + 1 reference) = 2 LatLonProfiles instantiations
        assert mock_llp_cls.call_count == 2
        assert mock_llp_instance.run.call_count == 2

        # For the reference run, std=True is always forced
        reference_run = mock_llp_instance.run.call_args_list[1]
        assert reference_run.kwargs["std"] is True

        # _create_plot called once per freq type: longterm + seasonal = 2 PlotLatLonProfiles
        assert mock_plot_cls.call_count == 2
        assert mock_plot_cls.return_value.run.call_count == 2

        plot_types = {call.kwargs["data_type"] for call in mock_plot_cls.call_args_list}
        assert plot_types == {"longterm", "seasonal"}

    def test_longterm_only_skips_seasonal_plot(self, build_config, mock_cluster, mock_llp):
        """With seasonal=False, only the longterm plot is produced."""
        mock_llp_cls, mock_plot_cls = mock_llp
        config_file = build_config(
            {
                "lat_lon_profiles": {**BASE_DICT, "seasonal": False},
            }
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_plot_cls.assert_called_once()
        assert mock_plot_cls.call_args.kwargs["data_type"] == "longterm"

    def test_formula_flag_forwarded_to_profile_run(self, build_config, mock_cluster, mock_llp):
        """A formula in the 'formulae' list propagates formula=True to LatLonProfiles.run."""
        mock_llp_cls, _ = mock_llp
        config_file = build_config(
            {
                "lat_lon_profiles": {**BASE_DICT, "variables": [], "formulae": ["net_toa"]},
            }
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        first_run = mock_llp_cls.return_value.run.call_args_list[0]
        assert first_run.kwargs["var"] == "net_toa"
        assert first_run.kwargs["formula"] is True
