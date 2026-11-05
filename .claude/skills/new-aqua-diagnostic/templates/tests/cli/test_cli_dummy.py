"""Tests for the Dummy CLI (parse_arguments + main orchestration), with everything mocked."""

import pytest

from aqua.core.exceptions import NoDataError
from aqua.diagnostics.dummy.cli_dummy import main, parse_arguments

CLI_MODULE = "aqua.diagnostics.dummy.cli_dummy"

BASE_DUMMY = {"run": True, "diagnostic_name": "dummy", "variables": ["2t"]}

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]


def test_parse_arguments_cli_options():
    """Detailed flag coverage lives in test_base_util.py."""
    args = parse_arguments(["--model", "IFS", "--nworkers", "4"])
    assert args.model == "IFS"
    assert args.nworkers == 4

    with pytest.raises(SystemExit):
        parse_arguments(["--help"])


class TestMain:
    """main() with Dummy and PlotDummy patched at the CLI module path."""

    @pytest.fixture
    def mocks(self, mocker):
        return mocker.patch(f"{CLI_MODULE}.Dummy"), mocker.patch(f"{CLI_MODULE}.PlotDummy")

    def test_disabled_skips_processing(self, build_config, mock_cluster, mocks):
        mock_diag_cls, mock_plot_cls = mocks
        main(["--config", build_config({"dummy": {"run": False}})])
        mock_diag_cls.assert_not_called()
        mock_plot_cls.assert_not_called()

    def test_full_pipeline(self, build_config, mock_cluster, mocks):
        mock_diag_cls, mock_plot_cls = mocks
        main(["--config", build_config({"dummy": BASE_DUMMY}, references=[])])
        mock_diag_cls.return_value.run.assert_called_once()
        mock_plot_cls.assert_called_once()
        mock_plot_cls.return_value.plot.assert_called_once()

    def test_reference_is_processed_and_plotted(self, build_config, mock_cluster, mocks):
        """The reference gets its own instance, run like the dataset, and is handed to the plot class."""
        mock_diag_cls, mock_plot_cls = mocks
        main(["--config", build_config({"dummy": BASE_DUMMY})])  # build_config adds a reference
        assert mock_diag_cls.call_count == 2
        assert mock_diag_cls.call_args_list[1].kwargs["model"] == "RefModel"
        assert mock_plot_cls.call_args.kwargs["ref_data"] is mock_diag_cls.return_value.result

    def test_without_reference(self, build_config, mock_cluster, mocks):
        """No `references` block: one instance, and the plot gets ref_data=None."""
        mock_diag_cls, mock_plot_cls = mocks
        main(["--config", build_config({"dummy": BASE_DUMMY}, references=[])])
        assert mock_diag_cls.call_count == 1
        assert mock_plot_cls.call_args.kwargs["ref_data"] is None

    def test_missing_reference_still_plots(self, build_config, mock_cluster, mocks):
        """A reference without data must not cost the figure of the dataset."""
        mock_diag_cls, mock_plot_cls = mocks
        mock_diag_cls.return_value.run.side_effect = [None, NoDataError("no reference")]
        main(["--config", build_config({"dummy": BASE_DUMMY})])
        assert mock_plot_cls.call_args.kwargs["ref_data"] is None

    def test_plot_only_does_not_run(self, build_config, mock_cluster, mocks):
        mock_diag_cls, mock_plot_cls = mocks
        config_file = build_config({"dummy": BASE_DUMMY}, references=[], output_overrides={"plot_only": True})
        main(["--config", config_file])
        mock_diag_cls.return_value.run.assert_not_called()
        mock_diag_cls.return_value.load.assert_called_once()

    def test_failure_skips_the_variable(self, build_config, mock_cluster, mocks):
        """One missing variable must not stop the run, nor produce a plot."""
        mock_diag_cls, mock_plot_cls = mocks
        mock_diag_cls.return_value.run.side_effect = NoDataError("missing")
        main(["--config", build_config({"dummy": BASE_DUMMY}, references=[])])
        mock_plot_cls.assert_not_called()
