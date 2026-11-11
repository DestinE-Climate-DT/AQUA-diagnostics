"""Tests for the DiagnosticProfiler helper."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from aqua.diagnostics.base.profiler import DiagnosticProfiler, _cpu_seconds, _max_memory_bytes, format_bytes

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]

GIB = 1024**3


def test_format_bytes():
    """Binary prefixes, one decimal above bytes."""
    assert format_bytes(None) == "n/a"
    assert format_bytes(512) == "512B"
    assert format_bytes(2048) == "2.0KiB"
    assert format_bytes(4.2 * GIB) == "4.2GiB"


def test_max_memory_platform_units():
    """ru_maxrss is bytes on macOS and KiB on Linux: both end up in bytes."""
    with patch("aqua.diagnostics.base.profiler.resource.getrusage") as mock_rusage:
        mock_rusage.return_value.ru_maxrss = 1000
        with patch("aqua.diagnostics.base.profiler.platform.system", return_value="Darwin"):
            assert _max_memory_bytes() == 1000
        with patch("aqua.diagnostics.base.profiler.platform.system", return_value="Linux"):
            assert _max_memory_bytes() == 1000 * 1024
    assert _max_memory_bytes() > 0
    assert _cpu_seconds() > 0


def test_profiler_without_client(caplog):
    """No client: no cluster field, and the line says all computation ran in this process."""
    profiler = DiagnosticProfiler(loglevel="INFO")
    assert profiler.logger.name == "Profiler"
    # the AQUA logger does not propagate to the root logger: swap in a plain one to capture the line
    profiler.logger = logging.getLogger("test-profiler")
    with caplog.at_level(logging.INFO, logger="test-profiler"):
        profiler.start()
        sum(i * i for i in range(200_000))  # burn a little CPU
        result = profiler.stop()

    assert result["walltime_seconds"] > 0
    assert result["cputime_seconds"] >= 0
    assert result["max_memory_bytes"] > 0
    assert result["cluster_max_memory_bytes"] is None
    assert result["nworkers"] is None
    assert result["private_cluster"] is None

    line = profiler.summary_line()
    assert line.startswith("walltime=")
    assert " cputime=" in line and " max_memory=" in line
    assert "cluster_max_memory" not in line
    assert line.endswith("(no dask cluster: all computation in this process)")
    assert line in caplog.text


def test_profiler_context_manager():
    """The context manager starts and stops the profiler."""
    with DiagnosticProfiler() as profiler:
        assert profiler.result is None
    assert profiler.result["walltime_seconds"] > 0


def test_profiler_stop_without_start():
    """Stopping a profiler that never started returns None and does not raise."""
    profiler = DiagnosticProfiler()
    assert profiler.stop() is None
    assert profiler.summary_line() == "no result"


def _mock_client(limits):
    client = MagicMock()
    client.scheduler_info.return_value = {"workers": {f"w{i}": {"memory_limit": lim} for i, lim in enumerate(limits)}}
    return client


@pytest.mark.parametrize("private_cluster, tag", [(False, "shared cluster"), (True, "private cluster")])
def test_profiler_with_mocked_client(private_cluster, tag):
    """With a client the cluster maximum comes from the MemorySampler, the limit from the workers."""
    client = _mock_client([4 * GIB, 4 * GIB, 4 * GIB])
    sampler = MagicMock()
    sampler.samples = {"profiler": [(0.0, 1 * GIB), (1.0, 3 * GIB), (2.0, 2 * GIB)]}

    with patch("distributed.diagnostics.MemorySampler", return_value=sampler):
        profiler = DiagnosticProfiler(client=client, private_cluster=private_cluster)
        profiler.start()
        result = profiler.stop()

    sampler.sample.assert_called_once_with("profiler", client=client, interval=1.0)
    client.scheduler_info.assert_called_once_with(n_workers=-1)
    assert result["cluster_max_memory_bytes"] == 3 * GIB
    assert result["cluster_memory_limit_bytes"] == 12 * GIB
    assert result["nworkers"] == 3
    assert result["private_cluster"] is private_cluster
    assert f"cluster_max_memory=3.0GiB/12.0GiB ({tag}, 3 workers)" in profiler.summary_line()


def test_profiler_without_memory_limit():
    """Workers without a memory limit: the limit is left out of the line."""
    client = _mock_client([0, None])
    sampler = MagicMock()
    sampler.samples = {"profiler": [(0.0, 1 * GIB)]}
    with patch("distributed.diagnostics.MemorySampler", return_value=sampler):
        profiler = DiagnosticProfiler(client=client)
        profiler.start()
        profiler.stop()
    assert "cluster_max_memory=1.0GiB (shared cluster, 2 workers)" in profiler.summary_line()


def test_profiler_sampler_failure_is_not_fatal():
    """A sampler that cannot start leaves the process numbers intact and says so."""
    client = MagicMock()
    with patch("distributed.diagnostics.MemorySampler", side_effect=RuntimeError("no scheduler")):
        profiler = DiagnosticProfiler(client=client)
        profiler.start()
        result = profiler.stop()

    assert result["max_memory_bytes"] > 0
    assert result["cluster_max_memory_bytes"] is None
    assert profiler.summary_line().endswith("(dask cluster memory not available)")


def test_local_graphs_are_counted():
    """Without a client the graphs run by the local dask scheduler are counted."""
    import dask.array as da

    profiler = DiagnosticProfiler()
    profiler.start()
    da.ones((100, 100), chunks=10).sum().compute()
    da.ones((100, 100), chunks=50).sum().compute()
    result = profiler.stop()

    assert result["dask_graphs"] == 2
    assert result["dask_tasks"] > result["max_graph_tasks"] > 0
    assert f"dask_graphs=2 dask_tasks={result['dask_tasks']} " in profiler.summary_line()

    # the callback is unregistered at stop: later computations are not counted
    da.ones(10, chunks=5).sum().compute()
    assert profiler.result["dask_graphs"] == 2


def test_graph_sizes_from_scheduler():
    """With a client the graph sizes come from the scheduler plugin, which is removed at stop."""
    client = _mock_client([4 * GIB])
    client.run_on_scheduler.return_value = [10, 30]
    profiler = DiagnosticProfiler(client=client)
    profiler.start()
    result = profiler.stop()

    client.register_plugin.assert_called_once()
    assert (result["dask_graphs"], result["dask_tasks"], result["max_graph_tasks"]) == (2, 40, 30)
    assert "dask_graphs=2 dask_tasks=40 max_graph_tasks=30" in profiler.summary_line()


def test_graph_plugin_failure_is_not_fatal():
    """A scheduler that refuses the plugin leaves the graph numbers out."""
    client = _mock_client([4 * GIB])
    client.register_plugin.side_effect = RuntimeError("no pickle")
    profiler = DiagnosticProfiler(client=client)
    profiler.start()
    result = profiler.stop()

    assert result["dask_graphs"] is None
    assert " dask_graphs=n/a " in profiler.summary_line()
    client.run_on_scheduler.assert_not_called()


def test_graph_plugin_on_real_cluster():
    """End to end on an in-process cluster: only the graphs of this client are counted."""
    import dask.array as da
    from distributed import Client, LocalCluster

    with LocalCluster(n_workers=1, threads_per_worker=1, processes=False, dashboard_address=None) as cluster:
        with Client(cluster) as client, Client(cluster) as other:
            profiler = DiagnosticProfiler(client=client, private_cluster=True)
            profiler.start()
            client.compute(da.ones((100, 100), chunks=10).sum()).result()
            other.compute(da.ones((100, 100), chunks=20).sum()).result()
            result = profiler.stop()
            plugins = client.run_on_scheduler(lambda dask_scheduler: list(dask_scheduler.plugins))

    assert result["dask_graphs"] == 1
    assert result["dask_tasks"] == result["max_graph_tasks"] > 0
    assert not any(p.startswith("aqua-profiler-") for p in plugins)
