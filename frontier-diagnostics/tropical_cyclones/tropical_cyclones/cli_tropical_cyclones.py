#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Command-line interface for Tropical Cyclones diagnostic (flat config)."""

import argparse
import sys

from aqua.diagnostics.base import template_parse_arguments, DiagnosticCLI
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
        help="Run only the StitchNodes step (skip DetectNodes). "
             "Assumes DetectNodes output files are already present on disk.",
    )
 
    return parser.parse_args(args)


if __name__ == "__main__":

    args = parse_arguments(sys.argv[1:])

    # Derive boolean flags; default (neither flag set) → run both steps.
    run_detect = not args.stitch_only
    run_stitch = not args.detect_only
 
    cli = DiagnosticCLI(
        args,
        diagnostic_name="tcs",
        default_config="config_tcs_cli.yaml",
    )

    cli.prepare()
    cli.open_dask_cluster()

    config = cli.config_dict

    cli.logger.info(f"Running {TOOLNAME} diagnostic")
    cli.logger.info(
        "Detect: %s | Stitch: %s", run_detect, run_stitch
    )

    # dataset (flat config)
    dataset_cfg = config.get("dataset", {})
    dataset_args = cli.dataset_args(dataset_cfg)

    # execution params (flat config)
    streaming = True
    stream_step = config.get("stream", {}).get("streamstep")
    startdate = config.get("time", {}).get("startdate")

    paths = config.get("paths", {})
    orography = config.get("orography", {}).get("file_path") is not None or True
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
        # default: full pipeline
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
        # only StitchNodes over the full date range from config
        import pandas as pd
 
        startdate_stitch = pd.to_datetime(config.get("time", {}).get("startdate"))
        enddate_stitch = pd.to_datetime(config.get("time", {}).get("enddate"))
        n_days_freq = config.get("stitch", {}).get("n_days_freq", 30)
        n_days_ext = config.get("stitch", {}).get("n_days_ext", 10)
 
        cli.logger.info(
            "Running StitchNodes from %s to %s", startdate_stitch, enddate_stitch
        )
        tropical.stitch_nodes_zoomin(
            startdate=startdate_stitch,
            enddate=enddate_stitch,
            n_days_freq=n_days_freq,
            n_days_ext=n_days_ext,
        )
 
    cli.close_dask_cluster()

    cli.logger.info("Tropical Cyclones diagnostic completed.")