"""Tests for the SeaIce CLI (parse_arguments + main orchestration)."""

import pytest

from aqua.core.util import load_yaml
from aqua.diagnostics.base import load_var_config
from aqua.diagnostics.seaice.cli_seaice import main, parse_arguments
from tests.cli.conftest import CONFIG_COLLECTIONS_DIR

CLI_MODULE = "aqua.diagnostics.seaice.cli_seaice"

# Minimal block of the timeseries / seasonal-cycle tools: one method, one northern region.
BASE_SET = {
    "run": True,
    "diagnostic_name": "seaice",
    "methods": ["extent"],
    "params": {"default": {"regions": ["arctic"], "std": True}},
}

BASE_2D_SET = {
    "run": True,
    "diagnostic_name": "seaice",
    "methods": ["fraction"],
    "params": {"default": {"regions": ["arctic"]}},
    "plot_params": {"default": {"months": [3, 9]}},
}

# The first reference is meant for the methods of the blocks above and covers the northern hemisphere;
# the second is for another method and the third covers the other hemisphere: both must be skipped.
REFERENCES = [
    {"catalog": "ref1", "model": "R1", "exp": "e1", "source": "s1", "methods": ["extent", "fraction"], "domain": "nh"},
    {"catalog": "ref2", "model": "R2", "exp": "e2", "source": "s2", "methods": ["thickness"], "domain": "nh"},
    {"catalog": "ref3", "model": "R3", "exp": "e3", "source": "s3", "domain": "sh"},
]

# No reference applies to the blocks above: the only one is for another method. An empty list is not an
# option, the configuration built on the commented repository template is not written correctly with one.
NO_REFERENCE = [REFERENCES[1]]

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

        The SeaIce instances expose the resolved regions and their definition, which the CLI
        uses to pick the regions of each reference.
        """
        mock_seaice_cls = mocker.patch(f"{CLI_MODULE}.SeaIce")
        mock_seaice_cls.return_value.regions = ["arctic"]
        mock_seaice_cls.return_value.regions_definition = {"arctic": {"lat_limits": [0, 90]}}
        mock_plot_cls = mocker.patch(f"{CLI_MODULE}.PlotSeaIce")
        mock_plot_2d_cls = mocker.patch(f"{CLI_MODULE}.Plot2DSeaIce")
        return mock_seaice_cls, mock_plot_cls, mock_plot_2d_cls

    def test_all_diagnostics_disabled_skip_processing(self, build_config, mock_cluster, mock_si):
        """When every seaice block has run=False, nothing is computed or plotted."""
        mock_seaice_cls, mock_plot_cls, mock_plot_2d_cls = mock_si
        config_file = build_config(
            {
                "seaice_timeseries": {"run": False},
                "seaice_seasonal_cycle": {"run": False},
                "seaice_2d_bias": {"run": False},
            }
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_seaice_cls.assert_not_called()
        mock_plot_cls.assert_not_called()
        mock_plot_2d_cls.assert_not_called()

    def test_timeseries_full_pipeline(self, build_config, mock_cluster, mock_si):
        """The dataset is run with the options of the block and its monthly series is plotted as a time series."""
        mock_seaice_cls, mock_plot_cls, _ = mock_si
        seaice = mock_seaice_cls.return_value
        config_file = build_config({"seaice_timeseries": BASE_SET}, references=NO_REFERENCE)

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_seaice_cls.assert_called_once()
        assert mock_seaice_cls.call_args.kwargs["regions"] == ["arctic"]
        assert mock_seaice_cls.call_args.kwargs["diagnostic_name"] == "seaice"
        seaice.run.assert_called_once()
        run = seaice.run.call_args.kwargs
        assert (run["method"], run["var"], run["seasonal_cycle"], run["annual"]) == ("extent", None, False, False)
        assert run["save_netcdf"] is True and run["rebuild"] is False  # from the output block
        assert "std" not in run  # the std is for the references

        mock_plot_cls.assert_called_once()
        assert mock_plot_cls.call_args.kwargs["monthly_models"] == [seaice.monthly]
        assert mock_plot_cls.call_args.kwargs["monthly_ref"] == []
        assert mock_plot_cls.call_args.kwargs["dpi"] == 50
        mock_plot_cls.return_value.plot_seaice.assert_called_once_with(plot_type="timeseries", save_format=["pdf"])

    def test_seasonal_cycle_full_pipeline(self, build_config, mock_cluster, mock_si):
        """The seasonal cycle block asks SeaIce for the cycle and plots it."""
        mock_seaice_cls, mock_plot_cls, _ = mock_si
        seaice = mock_seaice_cls.return_value
        config_file = build_config({"seaice_seasonal_cycle": BASE_SET}, references=NO_REFERENCE)

        main(["--config", config_file, "--loglevel", "WARNING"])

        seaice.run.assert_called_once()
        assert seaice.run.call_args.kwargs["seasonal_cycle"] is True
        assert mock_plot_cls.call_args.kwargs["monthly_models"] == [seaice.seasonalcycle]
        assert mock_plot_cls.return_value.plot_seaice.call_args.kwargs["plot_type"] == "seasonalcycle"

    def test_references_are_selected_by_method_and_domain(self, build_config, mock_cluster, mock_si):
        """
        Of three references only the one for this method and hemisphere is run, with the std of the block
        and its own analysis window; --realization goes to the Reader of the dataset.
        """
        mock_seaice_cls, mock_plot_cls, _ = mock_si
        seaice = mock_seaice_cls.return_value
        block = {**BASE_SET, "params": {"default": {**BASE_SET["params"]["default"], "ref_startdate": "2000-01-01"}}}
        config_file = build_config({"seaice_timeseries": block}, references=REFERENCES)

        main(["--config", config_file, "--loglevel", "WARNING", "--realization", "r1"])

        # 1 dataset + 1 matching reference
        assert mock_seaice_cls.call_count == 2
        reference = mock_seaice_cls.call_args_list[1].kwargs
        assert (reference["model"], reference["regions"]) == ("R1", ["arctic"])
        assert reference["startdate"] == "2000-01-01"
        dataset_run, reference_run = (call.kwargs for call in seaice.run.call_args_list)
        assert dataset_run["reader_kwargs"] == {"realization": "r1"}
        assert reference_run["std"] is True
        assert mock_plot_cls.call_args.kwargs["monthly_std_ref"] == [seaice.std_monthly]

    def test_reference_without_domain_and_methods_is_always_used(self, build_config, mock_cluster, mock_si):
        """The default reference of the test configuration has neither key: it is compared with all the regions."""
        mock_seaice_cls, _, _ = mock_si
        config_file = build_config({"seaice_seasonal_cycle": BASE_SET})

        main(["--config", config_file, "--loglevel", "WARNING"])

        assert mock_seaice_cls.call_count == 2
        assert mock_seaice_cls.call_args_list[1].kwargs["regions"] == ["arctic"]

    def test_2d_bias_full_pipeline(self, build_config, mock_cluster, mock_si):
        """The climatologies of dataset and reference are plotted with the months and the projection requested."""
        mock_seaice_cls, _, mock_plot_2d_cls = mock_si
        seaice = mock_seaice_cls.return_value
        config_file = build_config({"seaice_2d_bias": BASE_2D_SET}, references=REFERENCES)

        main(["--config", config_file, "--loglevel", "WARNING", "--proj", "azimuthal_equidistant"])

        assert seaice.run.call_count == 2  # 1 dataset + 1 matching reference
        assert seaice.run.call_args.kwargs["method"] == "fraction"
        mock_plot_2d_cls.assert_called_once()
        assert mock_plot_2d_cls.call_args.kwargs["models"] == [seaice.climatology]
        assert mock_plot_2d_cls.call_args.kwargs["ref"] == [seaice.climatology]
        mock_plot_2d_cls.return_value.plot_2d_seaice.assert_called_once_with(
            method="fraction",
            months=[3, 9],
            projection="azimuthal_equidistant",
            plot_ref_contour=True,
            save_format=["pdf"],
        )

    def test_2d_bias_without_reference_is_skipped(self, build_config, mock_cluster, mock_si):
        """The difference maps need a reference: without one the method is skipped instead of failing."""
        _, _, mock_plot_2d_cls = mock_si
        config_file = build_config({"seaice_2d_bias": BASE_2D_SET}, references=NO_REFERENCE)

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_plot_2d_cls.assert_not_called()


@pytest.mark.parametrize("config_file", ["legacy/seaice/config-seaice-osi.yaml", "jinja/seaice/config-seaice-psc.j2"])
def test_collection_configs_follow_the_layout_read_by_the_cli(config_file):
    """Regions under params.default and hemisphere on each reference: elsewhere the CLI would not see them."""
    kinds = load_yaml(str(CONFIG_COLLECTIONS_DIR.parent / "analysis" / "climatedt-experiment-kind.yaml"))
    config = load_yaml(str(CONFIG_COLLECTIONS_DIR / config_file), definitions=kinds["default"], strict=True)

    for tool, block in config["diagnostics"].items():
        for method in block["methods"]:
            assert load_var_config(config, method, diagnostic=tool)[1] == ["arctic", "antarctic"], tool
    assert [reference["domain"] for reference in config["references"]] == ["nh", "sh"]
    if config_file.endswith(".j2"):
        # the time series compare with the references over the timeseries window of the experiment kind
        params = load_var_config(config, "volume", diagnostic="seaice_timeseries")[0]
        assert params["ref_startdate"] == kinds["default"]["timeseries"]["startdate"]
