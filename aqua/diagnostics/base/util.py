"""
Utility functions for the CLI
"""

import argparse
import os
import uuid

import dask
from dask.base import tokenize
from dask.distributed import Client, LocalCluster

from aqua.core.configurer import ConfigPath
from aqua.core.logger import log_configure
from aqua.core.util import get_arg, load_yaml

# This creates a unique job token for this instance of the module
# so that all dask keys generated during this run are unique
_job_token = uuid.uuid4().hex
_original_tokenize = tokenize


def _unique_tokenize(*args, **kwargs):
    """Tokenize function that includes job token for uniqueness."""
    return _original_tokenize(_job_token, *args, **kwargs)


def template_parse_arguments(parser: argparse.ArgumentParser):
    """
    Add the default arguments to the parser.

    Args:
        parser: argparse.ArgumentParser

    Returns:
        argparse.ArgumentParser
    """
    parser.add_argument("--loglevel", "-l", type=str, required=False, help="loglevel")
    parser.add_argument("--catalog", type=str, required=False, help="catalog name")
    parser.add_argument("--model", type=str, required=False, help="model name")
    parser.add_argument("--exp", type=str, required=False, help="experiment name")
    parser.add_argument("--source", type=str, required=False, help="source name")
    parser.add_argument("--realization", type=str, default=None, help="realization name (default: None)")
    parser.add_argument("--config", "-c", type=str, default=None, help="yaml configuration file")
    parser.add_argument("--nworkers", "-n", type=int, required=False, help="number of workers")
    parser.add_argument("--cluster", type=str, required=False, help="cluster address")
    parser.add_argument("--regrid", type=str, required=False, help="target regrid resolution")
    parser.add_argument("--outputdir", type=str, required=False, help="output directory")
    parser.add_argument("--startdate", type=str, required=False, help="start date (YYYY-MM-DD)")
    parser.add_argument("--enddate", type=str, required=False, help="end date (YYYY-MM-DD)")

    return parser


def open_cluster(nworkers, cluster, loglevel: str = "WARNING"):
    """
    Open a dask cluster if nworkers is provided, otherwise connect to an existing cluster.

    Args:
        nworkers (int): number of workers
        cluster (str): cluster address
        loglevel (str): logging level

    Returns:
        client (dask.distributed.Client): dask client
        cluster (dask.distributed.LocalCluster): dask cluster
        private_cluster (bool): whether the cluster is private
    """

    logger = log_configure(log_name="Cluster", log_level=loglevel)

    private_cluster = False
    if nworkers or cluster:
        if not cluster:
            cluster = LocalCluster(n_workers=nworkers, threads_per_worker=1)
            logger.info(f"Initializing private cluster {cluster.scheduler_address} with {nworkers} workers.")
            private_cluster = True
        else:
            logger.info(f"Connecting to cluster {cluster} with client ID {_job_token}.")
            dask.base.tokenize = _unique_tokenize

        client = Client(cluster)
    else:
        client = None

    return client, cluster, private_cluster


def close_cluster(client, cluster, private_cluster, loglevel: str = "WARNING"):
    """
    Close the dask cluster and client.

    Args:
        client (dask.distributed.Client): dask client
        cluster (dask.distributed.LocalCluster): dask cluster
        private_cluster (bool): whether the cluster is private
        loglevel (str): logging level
    """
    logger = log_configure(log_name="Cluster", log_level=loglevel)

    if client:
        client.close()
        logger.debug("Dask client closed.")

    if private_cluster:
        cluster.close()
        logger.debug("Dask cluster closed.")


def get_diagnostic_configpath(diagnostic: str, folder="collections", loglevel="WARNING") -> str:
    """
    Get the path to the diagnostic configuration directory.

    Args:
        diagnostic (str): diagnostic name
        folder (str): folder name. Default is "collections". Can be "tools" as well.
        loglevel (str): logging level. Default is 'WARNING'.

    Returns:
        str: path to the diagnostic configuration directory
    """
    configdir = ConfigPath(loglevel=loglevel).configdir
    if folder == "templates":
        return os.path.join(configdir, folder)
    if folder in ["collections", "tools"]:
        return os.path.join(configdir, folder, diagnostic)
    raise ValueError(f"Invalid folder name: {folder}. Must be 'collections', 'tools', or 'templates'.")


def load_diagnostic_config(
    diagnostic: str, config: str = None, default_config: str = None, folder="collections", loglevel: str = "WARNING"
):
    """
    Load the diagnostic configuration file and return the configuration dictionary.

    Args:
        diagnostic (str): diagnostic name
        config (str): config argument can modify the default configuration file.
        folder (str): folder name. Default is "collections". Can be "tools" or "templates" as well.
        loglevel (str): logging level. Default is 'WARNING'.

    Returns:
        dict: configuration dictionary
    """
    if config:
        return load_yaml(config)

    if not default_config:
        default_config = f"config-{diagnostic}.yaml"

    filename = os.path.join(get_diagnostic_configpath(diagnostic, folder=folder, loglevel=loglevel), default_config)

    return load_yaml(filename)


def load_var_config(
    config_dict: dict,
    var,
    diagnostic: str,
    prepend_global: bool = False,
    collapse_freq_keys: tuple | None = None,
) -> tuple:
    """
    Build the variable-specific configuration for a diagnostic CLI.
    Merge precedence: params.default < params.<var_name> < inline dict.

    Args:
        config_dict (dict): full configuration dictionary.
        var (str | dict): variable name, or inline dict (any keys the diagnostic expects, e.g. ``units``,
            ``long_name``, ``regions``); the optional ``name`` key is the only one read here, to look up ``params.<name>``.
        diagnostic (str): key under ``config_dict["diagnostics"]`` to read from.
        prepend_global (bool): if True, always prepend ``None`` to the regions list (deduped), so the diagnostic
            is also run on the full (global) domain in addition to any configured regions.
        collapse_freq_keys (tuple | None): if set, pop these boolean flags and combine the True ones into a ``freq`` list.

    Returns:
        tuple: ``(var_config, regions)``; ``regions`` is always popped from ``var_config``.

    Raises:
        ValueError: if ``var`` is a string but the ``diagnostics.<diagnostic>`` block is missing or empty.
    """
    # Read the diagnostic-specific config block. Per-variable settings live
    # under params.<var_name>; shared defaults under params.default.
    diag_block = config_dict.get("diagnostics", {}).get(diagnostic, {}) or {}
    params = diag_block.get("params", {}) or {}
    default_params = params.get("default", {}) or {}

    # Merge precedence: params.default < params.<var_name> < inline dict.
    # Rightmost dict wins, so an inline override beats per-var which beats default.
    if isinstance(var, dict):
        var_name = var.get("name")
        var_specific = (params.get(var_name, {}) or {}) if var_name else {}
        var_config = {**default_params, **var_specific, **var}
    else:
        if not diag_block:
            raise ValueError(
                f"Cannot resolve config for variable '{var}': section 'diagnostics.{diagnostic}' is missing or empty."
            )
        var_name = var
        var_specific = params.get(var_name, {}) or {}
        var_config = {**default_params, **var_specific}
        var_config.setdefault("name", var_name)

    # Collapse boolean frequency flags (e.g. hourly/daily/monthly/annual) into a single `freq` list,
    # because Timeseries.run() expects a list of active frequencies, not separate booleans.
    if collapse_freq_keys:
        freq = [key for key in collapse_freq_keys if var_config.get(key)]
        for key in collapse_freq_keys:
            # The flags are then popped so they do not propagate as stray kwargs downstream
            var_config.pop(key, None)
        var_config["freq"] = freq

    # Regions are returned separately and not consumed as a kwarg by the diagnostic,
    # so with prepend_global=True the global case (None) is always run first.
    regions_in_config = var_config.pop("regions", None)
    if prepend_global:
        regions = [None]
        if regions_in_config:
            regions.extend(r for r in regions_in_config if r is not None)
    else:
        regions = list(regions_in_config) if regions_in_config else [None]

    return var_config, regions


def merge_config_args(config: dict, args: argparse.Namespace, loglevel: str = "WARNING") -> dict:
    """
    Merge the configuration dictionary with the arguments of the CLI.

    Args:
        config (dict): configuration dictionary
        args (argparse.Namespace): arguments of the CLI
        loglevel (str): logging level. Default is 'WARNING'.

    Returns:
        dict: merged configuration dictionary
    """
    logger = log_configure(log_name="merge_config_args", log_level=loglevel)
    datasets = config["datasets"]

    # Override the first dataset in the config file if provided in the command line
    datasets[0]["catalog"] = get_arg(args, "catalog", datasets[0]["catalog"])
    datasets[0]["model"] = get_arg(args, "model", datasets[0]["model"])
    datasets[0]["exp"] = get_arg(args, "exp", datasets[0]["exp"])
    datasets[0]["source"] = get_arg(args, "source", datasets[0]["source"])

    config["output"]["outputdir"] = get_arg(args, "outputdir", config["output"]["outputdir"])

    logger.debug("Analyzing models:")
    for model in config["datasets"]:
        logger.debug(f"  - {model['catalog']} {model['model']} {model['exp']} {model['source']}")

    if "references" in config:
        logger.debug("Using reference data:")
        for ref in config["references"]:
            logger.debug(f"  - {ref['catalog']} {ref['model']} {ref['exp']} {ref['source']}")

    return config
