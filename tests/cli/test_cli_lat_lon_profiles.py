"""Tests for the LatLonProfiles CLI (parse_arguments + main orchestration)."""

import pytest

from aqua.core.exceptions import NotEnoughDataError
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
        is invoked once for longterm and once for seasonal (in that order).
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

        # _create_plot called once per freq type, in order: longterm then seasonal
        assert mock_plot_cls.call_count == 2
        plot_types = [call.kwargs["data_type"] for call in mock_plot_cls.call_args_list]
        assert plot_types == ["longterm", "seasonal"]
        assert mock_plot_cls.return_value.run.call_count == 2

    def test_longterm_only_skips_seasonal_plot(self, build_config, mock_cluster, mock_llp):
        """With seasonal=False, only the longterm plot is produced."""
        _, mock_plot_cls = mock_llp
        config_file = build_config({"lat_lon_profiles": {**BASE_DICT, "seasonal": False}})

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_plot_cls.assert_called_once()
        assert mock_plot_cls.call_args.kwargs["data_type"] == "longterm"

    def test_seasonal_only_skips_longterm_plot(self, build_config, mock_cluster, mock_llp):
        """With longterm=False, only the seasonal plot is produced."""
        _, mock_plot_cls = mock_llp
        config_file = build_config({"lat_lon_profiles": {**BASE_DICT, "longterm": False}})

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_plot_cls.assert_called_once()
        assert mock_plot_cls.call_args.kwargs["data_type"] == "seasonal"

    def test_formula_flag_forwarded_to_profile_run(self, build_config, mock_cluster, mock_llp):
        """A formula in the 'formulae' list propagates formula=True to LatLonProfiles.run."""
        mock_llp_cls, _ = mock_llp
        config_file = build_config({"lat_lon_profiles": {**BASE_DICT, "variables": [], "formulae": ["net_toa"]}})

        main(["--config", config_file, "--loglevel", "WARNING"])

        first_run = mock_llp_cls.return_value.run.call_args_list[0]
        assert first_run.kwargs["var"] == "net_toa"
        assert first_run.kwargs["formula"] is True

    def test_no_references_passes_none_ref_data(self, build_config, mock_cluster, mock_llp):
        """With references=[], PlotLatLonProfiles is called with ref_data=None."""
        mock_llp_cls, mock_plot_cls = mock_llp
        config_file = build_config({"lat_lon_profiles": BASE_DICT}, references=[])

        main(["--config", config_file, "--loglevel", "WARNING"])

        # Only the dataset is instantiated (no reference).
        assert mock_llp_cls.call_count == 1
        assert mock_plot_cls.call_count == 2  # longterm + seasonal still produced
        for call in mock_plot_cls.call_args_list:
            assert call.kwargs["ref_data"] is None
            assert call.kwargs["ref_std_data"] is None

    def test_multiple_datasets_collected_in_profiles(self, build_config, mock_cluster, mock_llp):
        """All datasets are instantiated in order and their profiles bundled into one plot."""
        mock_llp_cls, mock_plot_cls = mock_llp
        config_file = build_config(
            {"lat_lon_profiles": {**BASE_DICT, "seasonal": False}},
            datasets=[
                {"catalog": "c1", "model": "M1", "exp": "e1", "source": "s1"},
                {"catalog": "c2", "model": "M2", "exp": "e2", "source": "s2"},
            ],
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        # 2 datasets + 1 reference = 3 constructors, in order.
        assert mock_llp_cls.call_count == 3
        models = [c.kwargs["model"] for c in mock_llp_cls.call_args_list]
        assert models == ["M1", "M2", "RefModel"]

        # Plot gets one list entry per dataset.
        mock_plot_cls.assert_called_once()
        assert len(mock_plot_cls.call_args.kwargs["data"]) == 2

    def test_multiple_regions_processed_independently(self, build_config, mock_cluster, mock_llp):
        """Each region in default_variables triggers its own dataset/ref pair and plot."""
        mock_llp_cls, mock_plot_cls = mock_llp
        config_file = build_config(
            {
                "lat_lon_profiles": {
                    **BASE_DICT,
                    "seasonal": False,
                    "default_variables": {
                        "2t": {"name": "2t", "regions": ["Global", "Tropics"]},
                    },
                }
            }
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        # 2 regions * (1 dataset + 1 reference) = 4 constructors.
        assert mock_llp_cls.call_count == 4
        regions = [c.kwargs["region"] for c in mock_llp_cls.call_args_list]
        assert regions == ["Global", "Global", "Tropics", "Tropics"]

        # One plot per region.
        assert mock_plot_cls.call_count == 2

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------
    def test_not_enough_data_on_dataset_skips_region(self, build_config, mock_cluster, mock_llp):
        """
        If the only dataset raises NotEnoughDataError, the region is skipped:
        the reference is never instantiated and no plot is produced.
        """
        mock_llp_cls, mock_plot_cls = mock_llp
        mock_llp_cls.return_value.run.side_effect = NotEnoughDataError("no data")
        config_file = build_config({"lat_lon_profiles": BASE_DICT})

        main(["--config", config_file, "--loglevel", "WARNING"])

        # Only the dataset constructor ran; reference skipped because profiles is empty.
        assert mock_llp_cls.call_count == 1
        mock_plot_cls.assert_not_called()

    def test_not_enough_data_on_reference_still_plots_without_ref(self, build_config, mock_cluster, mock_llp):
        """
        If the reference raises NotEnoughDataError, profile_ref is reset to None
        and the plot still runs with ref_data=None.
        """
        mock_llp_cls, mock_plot_cls = mock_llp
        # Dataset run ok, reference run fails.
        mock_llp_cls.return_value.run.side_effect = [None, NotEnoughDataError("no data")]
        config_file = build_config({"lat_lon_profiles": {**BASE_DICT, "seasonal": False}})

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_plot_cls.assert_called_once()
        assert mock_plot_cls.call_args.kwargs["ref_data"] is None
        assert mock_plot_cls.call_args.kwargs["ref_std_data"] is None

    # ------------------------------------------------------------------
    # Data plumbing / core transformations
    # ------------------------------------------------------------------
    def test_seasonal_data_transposition(self, build_config, mock_cluster, mocker):
        """
        _create_plot transposes [[DJF,MAM,JJA,SON], ...] (one per profile) into
        [DJF_list, MAM_list, JJA_list, SON_list] (one list per season).
        This is the only non-trivial data reshape in the CLI and deserves a direct check.
        """
        ds1 = mocker.MagicMock(seasonal=["d1_djf", "d1_mam", "d1_jja", "d1_son"])
        ds2 = mocker.MagicMock(seasonal=["d2_djf", "d2_mam", "d2_jja", "d2_son"])
        ref = mocker.MagicMock(
            seasonal=["r_djf", "r_mam", "r_jja", "r_son"],
            std_seasonal=["s_djf", "s_mam", "s_jja", "s_son"],
        )
        mock_llp_cls = mocker.patch(f"{CLI_MODULE}.LatLonProfiles", side_effect=[ds1, ds2, ref])
        mock_plot_cls = mocker.patch(f"{CLI_MODULE}.PlotLatLonProfiles")

        config_file = build_config(
            {"lat_lon_profiles": {**BASE_DICT, "longterm": False}},
            datasets=[
                {"catalog": "c1", "model": "M1", "exp": "e1", "source": "s1"},
                {"catalog": "c2", "model": "M2", "exp": "e2", "source": "s2"},
            ],
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        assert mock_llp_cls.call_count == 3  # 2 datasets + 1 reference
        mock_plot_cls.assert_called_once()
        kwargs = mock_plot_cls.call_args.kwargs
        assert kwargs["data_type"] == "seasonal"
        assert kwargs["data"] == [
            ["d1_djf", "d2_djf"],
            ["d1_mam", "d2_mam"],
            ["d1_jja", "d2_jja"],
            ["d1_son", "d2_son"],
        ]
        assert kwargs["ref_data"] == ["r_djf", "r_mam", "r_jja", "r_son"]
        assert kwargs["ref_std_data"] == ["s_djf", "s_mam", "s_jja", "s_son"]

    def test_meridional_mean_type_forwarded(self, build_config, mock_cluster, mock_llp):
        """mean_type from config is forwarded to LatLonProfiles."""
        mock_llp_cls, _ = mock_llp
        config_file = build_config({"lat_lon_profiles": {**BASE_DICT, "mean_type": "meridional"}})

        main(["--config", config_file, "--loglevel", "WARNING"])

        for call in mock_llp_cls.call_args_list:
            assert call.kwargs["mean_type"] == "meridional"
