#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Command-line interface for Tropical Cyclones diagnostic (flat config)."""

import argparse
import sys

import pandas as pd
import xarray as xr

from aqua.core.logger import log_configure
from aqua.diagnostics.base import DiagnosticCLI, template_parse_arguments
from tropical_cyclones import TCs

TOOLNAME = "TropicalCyclones"


def parse_arguments(args):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=f"{TOOLNAME} CLI")
    parser = template_parse_arguments(parser)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--detect-only",
        action="store_true",
        default=False,
        help="Run only the DetectNodes step (skip StitchNodes).",
    )
    mode.add_argument(
        "--stitch-only",
        action="store_true",
        default=False,
        help="Run only the StitchNodes step (skip DetectNodes). Assumes DetectNodes output files are already present on disk.",
    )
    parser.add_argument(
        "--override-tmpdir",
        type=str,
        default=None,
        help="Override tmpdir from config (used for parallel monthly jobs).",
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])

    run_detect = not args.stitch_only
    run_stitch = not args.detect_only

    cli = DiagnosticCLI(
        args,
        diagnostic_name="tcs",
        default_config="config_tcs.yaml",
    )

    cli.prepare()
    cli.open_dask_cluster()

    config = cli.config_dict

    # loglevel: CLI arg takes precedence, then config, then default WARNING
    cli.loglevel = args.loglevel or config.get("setup", {}).get("loglevel", "WARNING")
    cli.logger = log_configure(log_level=cli.loglevel, log_name="Tcs CLI")

    cli.logger.info(f"Running {TOOLNAME} diagnostic")
    cli.logger.info("Mode → detect: %s | stitch: %s", run_detect, run_stitch)

    # override paths and dates from CLI args if provided
    if args.override_tmpdir:
        config["paths"]["tmpdir"] = args.override_tmpdir
        cli.logger.info("Overriding tmpdir with: %s", args.override_tmpdir)

    if args.startdate:
        config["time"]["startdate"] = args.startdate
        cli.logger.info("Overriding startdate with: %s", args.startdate)

    if args.enddate:
        config["time"]["enddate"] = args.enddate
        cli.logger.info("Overriding enddate with: %s", args.enddate)

    # execution params
    streaming = True
    stream_step = config.get("stream", {}).get("streamstep")
    startdate = config.get("time", {}).get("startdate")
    paths = config.get("paths", {})
    orography = config.get("orography") is not None
    nproc = 1

    cli.logger.debug("Initializing Tropical Cyclones diagnostic")

    tropical = TCs(
        tdict=config,
        streaming=streaming,
        stream_step=stream_step,
        stream_startdate=startdate,
        paths=paths,
        loglevel=cli.loglevel,
        orography=orography,
        nproc=nproc,
    )

    cli.logger.info("Starting Tropical Cyclones pipeline")

    if run_detect and run_stitch:
        tropical.loop_streaming(config)

    elif run_detect:
        while tropical.data_retrieve():
            cli.logger.warning(
                "Streaming from %s to %s",
                tropical.stream_startdate,
                tropical.stream_enddate,
            )
            tropical.detect_nodes_zoomin()

    elif run_stitch:
        tropical.lowres2d = xr.Dataset()
        startdate_stitch = pd.to_datetime(config.get("time", {}).get("startdate"))
        enddate_stitch = pd.to_datetime(config.get("time", {}).get("enddate"))
        n_days_freq = config.get("stitch", {}).get("n_days_freq", 30)
        n_days_ext = config.get("stitch", {}).get("n_days_ext", 10)

        cli.logger.info("Running StitchNodes from %s to %s", startdate_stitch, enddate_stitch)
        tropical.stitch_nodes_zoomin(
            startdate=startdate_stitch,
            enddate=enddate_stitch,
            n_days_freq=n_days_freq,
            n_days_ext=n_days_ext,
        )

    cli.close_dask_cluster()
    cli.logger.info("Tropical Cyclones diagnostic completed.")
