"""
Lightweight profiler for diagnostic CLIs.

Records wall time, CPU time and maximum memory of the current process and,
when a dask client is given, the maximum memory of the dask workers sampled
while the profiler is running, together with the memory limit of the cluster.
One summary line is logged at the end by a logger named ``Profiler``, e.g.::

    walltime=312.4s cputime=95.1s max_memory=4.2GiB cluster_max_memory=18.3GiB/99.2GiB (shared cluster, 32 workers)
    walltime=312.4s cputime=290.0s max_memory=4.2GiB (no dask cluster: all computation in this process)
"""

import platform
import resource
import time

from aqua.core.logger import log_configure

SAMPLER_LABEL = "profiler"


def _max_memory_bytes() -> int:
    """
    Maximum resident memory of the current process in bytes.

    ``ru_maxrss`` is reported in bytes on macOS and in KiB on Linux.
    """
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return int(maxrss)
    return int(maxrss) * 1024


def _cpu_seconds() -> float:
    """User plus system CPU time of the current process, in seconds."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


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
    Measure wall time, CPU time and maximum memory of a diagnostic run.

    Usage::

        profiler = DiagnosticProfiler(client=client, private_cluster=False, loglevel="INFO")
        profiler.start()
        ...
        profiler.stop()  # logs one line through the "Profiler" logger

    or as a context manager::

        with DiagnosticProfiler(loglevel="INFO") as profiler:
            ...

    Args:
        client (dask.distributed.Client, optional): dask client. When given, the
            cluster-wide worker memory is sampled while the profiler runs.
        private_cluster (bool): whether the cluster belongs to this process only.
            When False, the sampled cluster memory also includes the tasks of
            other tools running on the same cluster.
        interval (float): sampling interval of the cluster memory, in seconds.
        loglevel (str): logging level. The summary line is logged at INFO level.
    """

    def __init__(self, client=None, private_cluster=False, interval=1.0, loglevel="WARNING"):
        self.client = client
        self.private_cluster = private_cluster
        self.interval = interval
        self.loglevel = loglevel
        self.logger = log_configure(log_level=loglevel, log_name="Profiler")

        self._t0 = None
        self._cpu0 = None
        self._sampler = None
        self._sampler_ctx = None
        self.result = None

    def start(self):
        """Start the timers and, if a client is available, the cluster memory sampler."""
        self._t0 = time.perf_counter()
        self._cpu0 = _cpu_seconds()
        self.result = None
        if self.client is not None:
            self._start_sampler()
        return self

    def stop(self):
        """Stop sampling, build the summary and log it. Returns the summary dict."""
        if self._t0 is None:
            self.logger.warning("Profiler stopped without being started, nothing to report.")
            return None

        walltime = time.perf_counter() - self._t0
        cputime = _cpu_seconds() - self._cpu0
        cluster_max, cluster_limit, nworkers = self._stop_sampler()

        self.result = {
            "walltime_seconds": walltime,
            "cputime_seconds": cputime,
            "max_memory_bytes": _max_memory_bytes(),
            "cluster_max_memory_bytes": cluster_max,
            "cluster_memory_limit_bytes": cluster_limit,
            "nworkers": nworkers,
            "private_cluster": self.private_cluster if self.client is not None else None,
        }
        self.logger.info(self.summary_line())
        self._t0 = None
        return self.result

    def summary_line(self) -> str:
        """One-line summary of the last result."""
        if self.result is None:
            return "no result"
        res = self.result
        line = (
            f"walltime={res['walltime_seconds']:.1f}s cputime={res['cputime_seconds']:.1f}s "
            f"max_memory={format_bytes(res['max_memory_bytes'])}"
        )
        if self.client is None:
            return line + " (no dask cluster: all computation in this process)"
        if res["cluster_max_memory_bytes"] is None:
            return line + " (dask cluster memory not available)"
        cluster = format_bytes(res["cluster_max_memory_bytes"])
        if res["cluster_memory_limit_bytes"]:
            cluster += f"/{format_bytes(res['cluster_memory_limit_bytes'])}"
        kind = "private cluster" if res["private_cluster"] else "shared cluster"
        return line + f" cluster_max_memory={cluster} ({kind}, {res['nworkers']} workers)"

    def _start_sampler(self):
        """Start the dask MemorySampler on the client, ignoring any failure."""
        try:
            from distributed.diagnostics import MemorySampler

            self._sampler = MemorySampler()
            self._sampler_ctx = self._sampler.sample(SAMPLER_LABEL, client=self.client, interval=self.interval)
            self._sampler_ctx.__enter__()
        except Exception as e:  # pylint: disable=broad-except
            self.logger.debug("Cluster memory sampling not available: %s", e)
            self._sampler = None
            self._sampler_ctx = None

    def _stop_sampler(self):
        """
        Stop the sampler.

        Returns:
            tuple: (maximum cluster memory, total worker memory limit, number of workers), memory in bytes,
            or (None, None, None) if sampling was not available.
        """
        if self._sampler_ctx is None:
            return None, None, None
        try:
            self._sampler_ctx.__exit__(None, None, None)
            samples = self._sampler.samples.get(SAMPLER_LABEL, [])
            cluster_max = max((nbytes for _, nbytes in samples), default=0)
            workers = self.client.scheduler_info(n_workers=-1).get("workers", {})
            limit = sum(int(w.get("memory_limit") or 0) for w in workers.values())
            return int(cluster_max), limit or None, len(workers)
        except Exception as e:  # pylint: disable=broad-except
            self.logger.debug("Cluster memory sampling failed: %s", e)
            return None, None, None
        finally:
            self._sampler = None
            self._sampler_ctx = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
