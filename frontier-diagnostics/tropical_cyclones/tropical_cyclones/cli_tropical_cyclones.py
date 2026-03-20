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
    return parser.parse_args(args)


if __name__ == "__main__":

    args = parse_arguments(sys.argv[1:])

    cli = DiagnosticCLI(
        args,
        diagnostic_name="tcs",  # arbitrary label, not used for config lookup
        default_config="config_tcs_cli.yaml",
    )

    cli.prepare()
    cli.open_dask_cluster()

    config = cli.config_dict

    cli.logger.info(f"Running {TOOLNAME} diagnostic")

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

    tropical.loop_streaming(config)

    cli.close_dask_cluster()

    cli.logger.info("Tropical Cyclones diagnostic completed.")