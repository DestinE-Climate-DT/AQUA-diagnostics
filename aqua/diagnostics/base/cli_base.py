"""
Base class for diagnostic CLI to centralize common operations.
"""

from aqua.core.exceptions import NoDataError, NotEnoughDataError
from aqua.core.logger import log_configure
from aqua.core.util import get_arg
from aqua.core.version import __version__ as aqua_version

from .defaults import SAVE_FORMAT
from .profiler import DiagnosticProfiler
from .util import close_cluster, load_diagnostic_config, merge_config_args, open_cluster

# Data availability problems: skipped with a warning, their message is enough to understand them
EXPECTED_ERRORS = (NoDataError, NotEnoughDataError)


class DiagnosticCLI:
    """
    Base class to centralize common CLI initialization operations.


    Usage:
        cli = DiagnosticCLI(
            args=args,
            diagnostic_name='timeseries',
            config='config_timeseries_atm.yaml'
        )
        cli.prepare()
        cli.open_dask_cluster()

        # Access prepared attributes
        logger = cli.logger
        config_dict = cli.config_dict
        outputdir = cli.outputdir
        ...

        # Each part that can fail on its own is skipped, without stopping the others
        for var in variables:
            try:
                ...
            except Exception as e:
                cli.log_failure(e, "variable %s", var)

        # At the end
        cli.close_dask_cluster()
    """

    def __init__(self, args, diagnostic_name, default_config, log_name=None):
        """
        Initialize the CLI handler.

        Args:
            args: Parsed command-line arguments
            diagnostic_name (str): Name of the diagnostic (e.g., 'timeseries', 'seaice')
            default_config (str): Default config file name
            log_name (str, optional): Logger name. Defaults to '{diagnostic_name} CLI'
        """
        self.args = args
        self.diagnostic_name = diagnostic_name
        self.default_config = default_config
        self.log_name = log_name or f"{diagnostic_name.capitalize()} CLI"

        # Attributes populated by prepare()
        self.loglevel = None
        self.logger = None
        self.client = None
        self.cluster = None
        self.private_cluster = None
        self.profiler = None
        self.config_dict = None
        self.regrid = None
        self.startdate = None
        self.enddate = None
        self.outputdir = None
        self.rebuild = None
        self.save_format = None
        self.save_netcdf = None
        self.plot_only = None
        self.dpi = None
        # DEPRECATED
        # self.create_catalog_entry = None  # Default behavior; can be overridden in prepare()

    def prepare(self, **overrides):
        """
        Execute common setup operations (excluding cluster management).

        This method:
        1. Sets up logging
        2. Loads and merges config
        3. Extracts common options (regrid, realization, output settings)

        Optional keyword arguments can be passed to override options extracted
        from configuration. Overrides are applied after extraction so they
        take precedence.

        Returns:
            self: For method chaining
        """
        self._load_config()
        self._setup_logging()
        self._extract_options()

        # option to override arguments
        if overrides:
            for key, value in overrides.items():
                setattr(self, key, value)

        return self

    def _load_config(self):
        """Load diagnostic config and merge with CLI args."""
        loglevel = get_arg(self.args, "loglevel", "WARNING")
        self.config_dict = load_diagnostic_config(
            diagnostic=self.diagnostic_name,
            config=self.args.config,
            default_config=self.default_config,
            loglevel=loglevel,
        )
        self.config_dict = merge_config_args(config=self.config_dict, args=self.args, loglevel=loglevel)

    def _setup_logging(self):
        """Setup logger.

        Precedence: ``--loglevel`` from CLI, then ``setup.loglevel`` from config YAML, then ``WARNING``.
        """
        config_loglevel = None
        if self.config_dict:
            config_loglevel = self.config_dict.get("setup", {}).get("loglevel")
        self.loglevel = get_arg(self.args, "loglevel", config_loglevel or "WARNING")
        self.logger = log_configure(log_level=self.loglevel, log_name=self.log_name)
        self.logger.info("Running %s diagnostic with AQUA version %s", self.diagnostic_name, aqua_version)

    def _extract_options(self):
        """Extract common options from config and args."""
        # Regrid option
        self.regrid = get_arg(self.args, "regrid", None)
        if self.regrid:
            self.logger.info("Regrid option is set to %s", self.regrid)

        self.startdate = get_arg(self.args, "startdate", None)
        self.enddate = get_arg(self.args, "enddate", None)
        if self.startdate:
            self.logger.info("Start date is set to %s", self.startdate)
        if self.enddate:
            self.logger.info("End date is set to %s", self.enddate)

        # Output options
        output_config = self.config_dict.get("output", {})
        self.outputdir = output_config.get("outputdir", "./")
        self.rebuild = output_config.get("rebuild", True)
        self.save_format = output_config.get("save_format", SAVE_FORMAT)
        self.save_netcdf = output_config.get("save_netcdf", True)
        self.plot_only = output_config.get("plot_only", False)
        self.dpi = output_config.get("dpi", 300)
        # DEPRECATED
        # self.create_catalog_entry = output_config.get("create_catalog_entry", False)

    def dataset_args(self, dataset):
        """
        Helper to extract dataset arguments for diagnostics.
        """

        return {
            "catalog": dataset["catalog"],
            "model": dataset["model"],
            "exp": dataset["exp"],
            "source": dataset["source"],
            "regrid": dataset.get("regrid") or self.regrid,
            "startdate": dataset.get("startdate") or self.startdate,
            "enddate": dataset.get("enddate") or self.enddate,
        }

    def reference_args(self, reference):
        """
        Helper to extract reference dataset arguments for diagnostics.
        Only difference from dataset_args is that regrid, startdate, enddate
        are not overridden by CLI options.
        """

        return {
            "catalog": reference["catalog"],
            "model": reference["model"],
            "exp": reference["exp"],
            "source": reference["source"],
            "regrid": reference.get("regrid"),
            "startdate": reference.get("startdate"),
            "enddate": reference.get("enddate"),
        }

    def log_failure(self, error, msg, *args):
        """
        Log an exception caught by the CLI, which then skips that part of the diagnostic and goes on.

        Expected data availability problems (``NoDataError``, ``NotEnoughDataError``) are logged as a warning,
        any other exception as an error with its traceback, so that the log shows where it was raised.

        Args:
            error (Exception): The caught exception.
            msg (str): What is skipped, %-style, e.g. ``"variable %s in region %s"``.
            *args: The arguments of ``msg``.
        """
        if isinstance(error, EXPECTED_ERRORS):
            self.logger.warning("Skipping " + msg + ": %s: %s", *args, type(error).__name__, error)
        else:
            self.logger.error("Skipping " + msg + ": %s: %s", *args, type(error).__name__, error, exc_info=error)

    def open_dask_cluster(self):
        """
        Open dask cluster if requested via CLI arguments.

        Returns:
            self: For method chaining
        """
        cluster_arg = get_arg(self.args, "cluster", None)
        nworkers = get_arg(self.args, "nworkers", None)

        self.client, self.cluster, self.private_cluster = open_cluster(
            nworkers=nworkers, cluster=cluster_arg, loglevel=self.loglevel
        )
        self.start_profiler()
        return self

    def start_profiler(self):
        """
        Start the profiler measuring wall time, peak memory of this process and,
        when a dask client is available, the memory of the dask workers.
        Called by open_dask_cluster(); call it directly when the CLI does not use a cluster.

        Returns:
            self: For method chaining
        """
        self.profiler = DiagnosticProfiler(
            client=self.client,
            private_cluster=bool(self.private_cluster),
            loglevel=self.loglevel,
        )
        self.profiler.start()
        return self

    def close_dask_cluster(self):
        """
        Close the dask cluster if it was opened and log the profiling summary.
        """
        if self.profiler is not None:
            self.profiler.stop()
        if self.client or self.cluster:
            close_cluster(
                client=self.client, cluster=self.cluster, private_cluster=self.private_cluster, loglevel=self.loglevel
            )
            self.logger.info("%s diagnostic completed.", self.diagnostic_name)
