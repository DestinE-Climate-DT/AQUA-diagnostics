"""Tests for the AQUA diagnostics setup checker CLI."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aqua.core.exceptions import NoDataError
from aqua.core.util import dump_yaml as write_yaml
from aqua.core.util import load_yaml
from aqua.diagnostics.dummy.cli_checker import _checker_build_config, main, parse_arguments

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]


def test_parse_arguments_uses_common_and_checker_options():
    """The checker parser combines shared diagnostic and checker flags."""
    args = parse_arguments(
        [
            "--model",
            "IFS",
            "--exp",
            "test-tco79",
            "--source",
            "short",
            "--realization",
            "r1",
            "--regrid",
            "r200",
            "--yaml",
            "/tmp/output",
            "--no-rebuild",
        ]
    )

    assert args.model == "IFS"
    assert args.exp == "test-tco79"
    assert args.source == "short"
    assert args.realization == "r1"
    assert args.regrid == "r200"
    assert args.yaml == "/tmp/output"
    assert args.rebuild is False


def test_checker_config_uses_effective_operational_arguments():
    """The temporary config mirrors the effective values passed by the user."""
    args = parse_arguments(
        [
            "--catalog",
            "ci",
            "--model",
            "IFS",
            "--exp",
            "test-tco79",
            "--source",
            "short",
            "--realization",
            "r2",
            "--startdate",
            "2000-01-01",
            "--enddate",
            "2001-12-31",
            "--outputdir",
            "/tmp/output",
            "--no-rebuild",
        ]
    )

    config = _checker_build_config(args)

    assert config["datasets"] == [
        {
            "catalog": "ci",
            "model": "IFS",
            "exp": "test-tco79",
            "source": "short",
            "regrid": "r100",
            "startdate": "2000-01-01",
            "enddate": "2001-12-31",
            "reader_kwargs": {"realization": "r2"},
        }
    ]
    assert config["output"] == {"outputdir": "/tmp/output", "rebuild": False}


def test_main_removes_temporary_config():
    """The generated operational configuration is removed after preparation."""
    config_paths = []

    def capture_config(outfile, cfg):
        config_paths.append(outfile)
        write_yaml(outfile=outfile, cfg=cfg)

    with (
        patch("aqua.diagnostics.dummy.cli_checker.dump_yaml", side_effect=capture_config),
        patch("aqua.diagnostics.dummy.cli_checker.Diagnostic", return_value=MagicMock()),
    ):
        main(["--model", "IFS", "--exp", "test-tco79", "--source", "short"])

    assert len(config_paths) == 1
    assert not Path(config_paths[0]).exists()


def test_main_retrieves_with_diagnostic_and_writes_metadata(tmp_path):
    """The checker uses Diagnostic retrieval and writes resolved catalog metadata."""
    experiment_catalog = SimpleNamespace(
        metadata={"catalog_dir": "/catalog", "machine": "levante"},
        description="Test experiment",
    )
    diagnostic = MagicMock()
    diagnostic.catalog = "resolved-catalog"
    diagnostic.model = "IFS"
    diagnostic.exp = "test-tco79"
    diagnostic.reader.backend.expcat = experiment_catalog

    with patch("aqua.diagnostics.dummy.cli_checker.Diagnostic", return_value=diagnostic) as diagnostic_class:
        main(
            [
                "--catalog",
                "ci",
                "--model",
                "IFS",
                "--exp",
                "test-tco79",
                "--source",
                "short",
                "--realization",
                "r1",
                "--regrid",
                "r200",
                "--yaml",
                str(tmp_path),
                "--no-rebuild",
            ]
        )

    diagnostic_class.assert_called_once_with(
        catalog="ci",
        model="IFS",
        exp="test-tco79",
        source="short",
        regrid="r200",
        startdate=None,
        enddate=None,
        loglevel="WARNING",
    )
    diagnostic.retrieve.assert_called_once_with(reader_kwargs={"realization": "r1", "rebuild": False})

    metadata = load_yaml(str(tmp_path / "experiment.yaml"))
    assert metadata == {
        "machine": "levante",
        "description": "Test experiment",
        "catalog": "resolved-catalog",
        "model": "IFS",
        "experiment": "test-tco79",
    }


def test_main_uses_default_regrid_and_wraps_retrieval_errors():
    """Retrieval failures become the core NoDataError expected by callers."""
    diagnostic = MagicMock()
    diagnostic.retrieve.side_effect = RuntimeError("catalog unavailable")

    with (
        patch("aqua.diagnostics.dummy.cli_checker.Diagnostic", return_value=diagnostic) as diagnostic_class,
        pytest.raises(NoDataError, match="catalog unavailable"),
    ):
        main(
            [
                "--model",
                "IFS",
                "--exp",
                "test-tco79",
                "--source",
                "short",
            ]
        )

    assert diagnostic_class.call_args.kwargs["regrid"] == "r100"
    diagnostic.retrieve.assert_called_once_with(reader_kwargs={"rebuild": True})


def test_main_requires_dataset_identity():
    """A checker run requires a complete model, experiment, and source triplet."""
    with pytest.raises(ValueError, match="model, exp and source are required"):
        main(["--exp", "test-tco79", "--source", "short"])
