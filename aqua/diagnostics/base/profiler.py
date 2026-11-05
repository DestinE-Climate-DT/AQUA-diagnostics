"""
Lightweight profiler for diagnostic CLIs.

Records wall time, the peak resident memory of the current process and,
when a dask client is given, the peak memory of the dask workers sampled
while the profiler is running. One summary line is logged at the end.
"""

import platform
import resource
import time

from aqua.core.logger import log_configure


def _ru_maxrss_bytes() -> int:
    """
    Peak resident set size of the current process in bytes.

    ``ru_maxrss`` is reported in bytes on macOS and in KiB on Linux.
    """
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return int(maxrss)
    return int(maxrss) * 1024


def format_bytes(nbytes) -> str:
    """Human readable size with binary prefixes, e.g. ``4.2GiB``."""
    if nbytes is None:
        return "n/a"
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TiB"


class DiagnosticProfiler:
    """
    Measure wall time and peak memory of a diagnostic run.

    Usage::

        profiler = DiagnosticProfiler(name="timeseries", client=client, private_cluster=False)
        profiler.start()
        ...
        profiler.stop()  # logs one "[profile] ..." line

    or as a context manager::

        with DiagnosticProfiler(name="timeseries") as profiler:
            ...

    Args:
        name (str): name of the diagnostic, used in the log line.
        client (dask.distributed.Client, optional): dask client. When given, the
            cluster-wide worker memory is sampled while the profiler runs.
        private_cluster (bool): whether the cluster belongs to this process only.
            When False, the sampled cluster memory also includes the tasks of
            other tools running on the same cluster.
        interval (float): sampling interval of the cluster memory, in seconds.
        loglevel (str): logging level.
        logger (logging.Logger, optional): logger used for the summary line.
            Defaults to a logger named ``Profiler``.
    """

    def __init__(self, name, client=None, private_cluster=False, interval=1.0, loglevel="WARNING", logger=None):
        self.name = name
        self.client = client
        self.private_cluster = private_cluster
        self.interval = interval
        self.loglevel = loglevel
        self.logger = logger or log_configure(log_level=loglevel, log_name="Profiler")

        self._t0 = None
        self._sampler = None
        self._sampler_ctx = None
        self.result = None

    def start(self):
        """Start the timer and, if a client is available, the cluster memory sampler."""
        self._t0 = time.perf_counter()
        self.result = None
        if self.client is not None:
            self._start_sampler()
        return self

    def stop(self):
        """Stop sampling, build the summary and log it. Returns the summary dict."""
        if self._t0 is None:
            self.logger.warning("Profiler stopped without being started, nothing to report.")
            return None

        wall = time.perf_counter() - self._t0
        cluster_peak, nworkers = self._stop_sampler()

        self.result = {
            "name": self.name,
            "wall_seconds": wall,
            "driver_peak_rss_bytes": _ru_maxrss_bytes(),
            "cluster_peak_bytes": cluster_peak,
            "nworkers": nworkers,
            "private_cluster": self.private_cluster if self.client is not None else None,
        }
        self.logger.info(self.summary_line())
        self._t0 = None
        return self.result

    def summary_line(self) -> str:
        """One-line summary of the last result, with a fixed ``[profile]`` prefix."""
        if self.result is None:
            return f"[profile] {self.name}: no result"
        res = self.result
        line = (
            f"[profile] {res['name']}: wall={res['wall_seconds']:.1f}s "
            f"driver_peak_rss={format_bytes(res['driver_peak_rss_bytes'])}"
        )
        if res["cluster_peak_bytes"] is not None:
            kind = "private cluster" if res["private_cluster"] else "shared cluster"
            line += f" cluster_peak_mem={format_bytes(res['cluster_peak_bytes'])} ({kind}, {res['nworkers']} workers)"
        return line

    def _start_sampler(self):
        """Start the dask MemorySampler on the client, ignoring any failure."""
        try:
            from distributed.diagnostics import MemorySampler

            self._sampler = MemorySampler()
            self._sampler_ctx = self._sampler.sample(self.name, client=self.client, interval=self.interval)
            self._sampler_ctx.__enter__()
        except Exception as e:  # pylint: disable=broad-except
            self.logger.debug("Cluster memory sampling not available: %s", e)
            self._sampler = None
            self._sampler_ctx = None

    def _stop_sampler(self):
        """Stop the sampler and return (peak bytes, number of workers), or (None, None)."""
        if self._sampler_ctx is None:
            return None, None
        try:
            self._sampler_ctx.__exit__(None, None, None)
            samples = self._sampler.samples.get(self.name, [])
            peak = max((nbytes for _, nbytes in samples), default=0)
            nworkers = len(self.client.scheduler_info().get("workers", {}))
            return int(peak), nworkers
        except Exception as e:  # pylint: disable=broad-except
            self.logger.debug("Cluster memory sampling failed: %s", e)
            return None, None
        finally:
            self._sampler = None
            self._sampler_ctx = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
