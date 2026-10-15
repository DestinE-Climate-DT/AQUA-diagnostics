"""Tests for the Taylor CLI (parse_arguments + main orchestration)."""

import pytest

from aqua.diagnostics.taylor.cli_taylor import main, parse_arguments

CLI_MODULE = "aqua.diagnostics.taylor.cli_taylor"

BASE_TAYLOR = {
    "run": True,
    "diagnostic_name": "atmosphere2d",
    "seasons": True,
    "variables": ["2t", "msl"],
    "formulae": ["(10u^2+10v^2)^0.5"],
    "params": {
        "default": {"regions": [None]},
        "(10u^2+10v^2)^0.5": {"short_name": "10si", "long_name": "Wind speed at 10m"},
    },
    "plot_params": {"std_max": 1.5, "annotate": True},
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
    args = parse_arguments(["--model", "IFS", "--loglevel", "WARNING"])
    assert args.model == "IFS"
    assert args.loglevel == "WARNING"
    assert args.catalog is None

    with pytest.raises(SystemExit):
        parse_arguments(["--help"])


# ======================================================================
# CLI execution flow (main)
# ======================================================================
class TestMainExecutionFlow:
    """Test main() execution flow with mocked Taylor and PlotTaylor."""

    @pytest.fixture
    def mock_taylor(self, mocker):
        """
        Patch Taylor and PlotTaylor at the CLI module path.
        Returns (mock_taylor_cls, mock_plot_cls).
        """
        mock_taylor_cls = mocker.patch(f"{CLI_MODULE}.Taylor")
        mock_plot_cls = mocker.patch(f"{CLI_MODULE}.PlotTaylor")
        return mock_taylor_cls, mock_plot_cls

    def test_diagnostic_disabled_skips_processing(self, build_config, mock_cluster, mock_taylor):
        """When run=False, neither Taylor nor PlotTaylor is instantiated."""
        mock_taylor_cls, mock_plot_cls = mock_taylor
        config_file = build_config({"taylor": {"run": False}})

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_taylor_cls.assert_not_called()
        mock_plot_cls.assert_not_called()

    def test_full_pipeline(self, build_config, mock_cluster, mock_taylor):
        """
        For each of the 3 variables the reference and the dataset are run, the statistics computed
        and loaded back, and a seasonal diagram plotted. A single annual diagram has all the variables.
        """
        mock_taylor_cls, mock_plot_cls = mock_taylor
        inst = mock_taylor_cls.return_value
        config_file = build_config({"taylor": BASE_TAYLOR})

        main(["--config", config_file, "--loglevel", "WARNING"])

        # 3 variables * (1 reference + 1 dataset)
        assert mock_taylor_cls.call_count == 6
        assert inst.run.call_count == 6
        assert inst.compute_statistics.call_count == 3
        assert inst.load.call_count == 3
        assert all(call.kwargs["diagnostic_name"] == "atmosphere2d" for call in mock_taylor_cls.call_args_list)

        formula_run = inst.run.call_args_list[-1].kwargs
        assert formula_run["var"] == "(10u^2+10v^2)^0.5"
        assert formula_run["formula"] is True
        assert formula_run["short_name"] == "10si"
        # Files are named after the short name
        assert inst.load.call_args_list[-1].kwargs["var"] == "10si"

        plot_calls = mock_plot_cls.return_value.plot_taylor.call_args_list
        # 3 seasonal diagrams + 1 annual one
        assert len(plot_calls) == 4
        assert [c.kwargs["period"] for c in plot_calls] == ["seasonal"] * 3 + ["annual"]
        annual = plot_calls[-1].kwargs
        assert len(annual["statistics"]) == 3
        assert annual["color_by"] == "var"
        assert annual["std_max"] == 1.5
        assert annual["annotate"] is True

    def test_plot_only_skips_computation(self, build_config, mock_cluster, mock_taylor):
        """With plot_only, nothing is run and the statistics are loaded."""
        mock_taylor_cls, mock_plot_cls = mock_taylor
        inst = mock_taylor_cls.return_value
        config_file = build_config(
            {"taylor": {**BASE_TAYLOR, "seasons": False}}, output_overrides={"plot_only": True, "save_netcdf": False}
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        inst.run.assert_not_called()
        inst.compute_statistics.assert_not_called()
        assert inst.load.call_count == 3
        # No seasonal diagrams
        mock_plot_cls.return_value.plot_taylor.assert_called_once()

    def test_reference_failure_skips_variable(self, build_config, mock_cluster, mock_taylor, mocker):
        """When the reference has not enough data, the variable is skipped."""
        from aqua.core.exceptions import NotEnoughDataError

        mock_taylor_cls, mock_plot_cls = mock_taylor
        inst = mock_taylor_cls.return_value
        # Reference of the first variable fails, then reference and dataset of the second succeed
        inst.run.side_effect = [NotEnoughDataError("no data"), mocker.DEFAULT, mocker.DEFAULT]
        config_file = build_config({"taylor": {**BASE_TAYLOR, "variables": ["2t", "msl"], "formulae": []}})

        main(["--config", config_file, "--loglevel", "WARNING"])

        assert inst.compute_statistics.call_count == 1
        annual = mock_plot_cls.return_value.plot_taylor.call_args_list[-1].kwargs
        assert len(annual["statistics"]) == 1

    def test_reader_kwargs_are_forwarded_per_entry(self, build_config, mock_cluster, mock_taylor):
        """Dataset and reference Reader kwargs reach run(), not the constructor."""
        mock_taylor_cls, _ = mock_taylor
        datasets = [
            {
                "catalog": "test-catalog",
                "model": "TestModel",
                "exp": "test-exp",
                "source": "test-source",
                "reader_kwargs": {"realization": "r2"},
            }
        ]
        references = [
            {
                "catalog": "ref-catalog",
                "model": "RefModel",
                "exp": "ref-exp",
                "source": "ref-source",
                "reader_kwargs": {"zoom": 4},
            }
        ]
        config_file = build_config(
            {"taylor": {**BASE_TAYLOR, "variables": ["2t"], "formulae": []}}, datasets=datasets, references=references
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        assert all("reader_kwargs" not in call.kwargs for call in mock_taylor_cls.call_args_list)
        run_calls = mock_taylor_cls.return_value.run.call_args_list
        assert run_calls[0].kwargs["reader_kwargs"] == {"zoom": 4}
        assert run_calls[1].kwargs["reader_kwargs"] == {"realization": "r2"}
        load_call = mock_taylor_cls.return_value.load.call_args
        assert load_call.kwargs["reader_kwargs"] == {"realization": "r2"}

    def test_no_reference_raises(self, build_config, mock_cluster, mock_taylor):
        """The diagnostic needs a reference."""
        config_file = build_config({"taylor": BASE_TAYLOR}, references=[])

        with pytest.raises(ValueError, match="reference"):
            main(["--config", config_file, "--loglevel", "WARNING"])
