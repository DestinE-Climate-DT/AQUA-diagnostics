#!/usr/bin/env python3
"""Command-line setup checker for AQUA diagnostics.

The checker validates that a configured dataset can be retrieved and can
optionally write its catalog metadata to ``experiment.yaml``.
"""

import argparse
import os
import sys
from tempfile import TemporaryDirectory

from aqua.core.exceptions import NoDataError
from aqua.core.util import dump_yaml
from aqua.diagnostics.base import Diagnostic, DiagnosticCLI, template_parse_arguments


def parse_arguments(arguments):
    """Parse command-line arguments for the setup checker.

    Args:
        arguments (list): Command-line arguments to parse.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Check the AQUA diagnostics setup")
    parser = template_parse_arguments(parser)
    parser.add_argument("--yaml", help="write experiment.yaml to this directory")
    parser.add_argument(
        "--no-rebuild",
        action="store_false",
        dest="rebuild",
        default=None,
        help="reuse existing areas and regridding weights",
    )
    return parser.parse_args(arguments)


def _write_experiment_yaml(diagnostic, outputdir):
    """Write the checked experiment metadata to YAML."""
    experiment_catalog = diagnostic.reader.backend.expcat
    metadata = experiment_catalog.metadata.copy()
    metadata.pop("catalog_dir", None)
    metadata.update(
        description=getattr(experiment_catalog, "description", ""),
        catalog=diagnostic.catalog,
        model=diagnostic.model,
        experiment=diagnostic.exp,
    )
    dump_yaml(outfile=os.path.join(outputdir, "experiment.yaml"), cfg=metadata)


def _checker_build_config(args):
    """Build a diagnostic configuration from the effective CLI arguments."""
    reader_kwargs = {"realization": args.realization} if args.realization else None
    return {
        "setup": {"loglevel": args.loglevel or "WARNING"},
        "datasets": [
            {
                "catalog": args.catalog,
                "model": args.model,
                "exp": args.exp,
                "source": args.source,
                "regrid": args.regrid or "r100",
                "startdate": args.startdate,
                "enddate": args.enddate,
                "reader_kwargs": reader_kwargs,
            }
        ],
        "output": {
            "outputdir": args.outputdir or "./",
            "rebuild": True if args.rebuild is None else args.rebuild,
        },
    }


def main(argv=None):
    """Run the AQUA diagnostics setup checker.

    Args:
        argv (list, optional): Command-line arguments. Defaults to sys.argv[1:].

    Raises:
        ValueError: If model, experiment, or source is not configured.
        NoDataError: If the configured dataset cannot be retrieved.
    """
    args = parse_arguments(argv if argv is not None else sys.argv[1:])
    with TemporaryDirectory(prefix="aqua-checker-") as config_dir:
        args.config = os.path.join(config_dir, "config-checker.yaml")
        dump_yaml(outfile=args.config, cfg=_checker_build_config(args))
        cli = DiagnosticCLI(
            args=args,
            diagnostic_name="checker",
            default_config=None,
            log_name="Setup Checker CLI",
        ).prepare()

    dataset = cli.config_dict["datasets"][0]
    if any(dataset.get(key) is None for key in ("model", "exp", "source")):
        raise ValueError("model, exp and source are required arguments")
    if dataset.get("catalog") is None:
        cli.logger.warning("No catalog provided, determining the catalog with the Reader")

    reader_kwargs = dict(dataset.get("reader_kwargs") or {})
    reader_kwargs["rebuild"] = cli.rebuild
    diagnostic = Diagnostic(**cli.dataset_args(dataset), loglevel=cli.loglevel)

    try:
        diagnostic.retrieve(reader_kwargs=reader_kwargs)
    except Exception as error:
        cli.logger.error("Failed to retrieve data: %s", error)
        cli.logger.error("Check that the model is available in the Reader catalog.")
        raise NoDataError(f"Failed to retrieve data: {error}") from error

    if args.yaml:
        cli.logger.info("Creating experiment.yaml")
        _write_experiment_yaml(diagnostic, args.yaml)

    cli.logger.info("Check is terminated, diagnostics can run!")


if __name__ == "__main__":
    main()
