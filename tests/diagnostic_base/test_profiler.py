"""Tests for the DiagnosticProfiler helper."""

import logging
import platform
from unittest.mock import MagicMock, patch

import pytest

from aqua.diagnostics.base.profiler import DiagnosticProfiler, _ru_maxrss_bytes, format_bytes

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]


def test_format_bytes():
    """Binary prefixes, one decimal above bytes."""
    assert format_bytes(None) == "n/a"
    assert format_bytes(512) == "512B"
    assert format_bytes(2048) == "2.0KiB"
    assert format_bytes(4.2 * 1024**3) == "4.2GiB"


def test_ru_maxrss_bytes_platform_units():
    """ru_maxrss is bytes on macOS and KiB on Linux: both end up in bytes."""
    with patch("aqua.diagnostics.base.profiler.resource.getrusage") as mock_rusage:
        mock_rusage.return_value.ru_maxrss = 1000
        with patch("aqua.diagnostics.base.profiler.platform.system", return_value="Darwin"):
            assert _ru_maxrss_bytes() == 1000
        with patch("aqua.diagnostics.base.profiler.platform.system", return_value="Linux"):
            assert _ru_maxrss_bytes() == 1000 * 1024

    # the real value of this process is positive and plausible on the running platform
    assert _ru_maxrss_bytes() > 0
    assert platform.system() in ("Darwin", "Linux", "Windows")


def test_profiler_without_client(caplog):
    """No client: wall time and driver RSS only, no cluster field in the line."""
    profiler = DiagnosticProfiler(name="dummy", loglevel="INFO")
    assert profiler.logger.name == "dummy profiler"
    # the AQUA logger does not propagate to the root logger: swap in a plain one to capture the line
    profiler.logger = logging.getLogger("test-profiler")
    with caplog.at_level(logging.INFO, logger="test-profiler"):
        profiler.start()
        result = profiler.stop()

    assert result["name"] == "dummy"
    assert result["wall_seconds"] > 0
    assert result["driver_peak_rss_bytes"] > 0
    assert result["cluster_peak_bytes"] is None
    assert result["nworkers"] is None
    assert result["private_cluster"] is None

    line = profiler.summary_line()
    assert line.startswith("[profile] dummy: wall=")
    assert "driver_peak_rss=" in line
    assert "cluster_peak_mem" not in line
    assert line in caplog.text


def test_profiler_context_manager():
    """The context manager starts and stops the profiler."""
    with DiagnosticProfiler(name="dummy") as profiler:
        assert profiler.result is None
    assert profiler.result["wall_seconds"] > 0


def test_profiler_stop_without_start():
    """Stopping a profiler that never started returns None and does not raise."""
    profiler = DiagnosticProfiler(name="dummy")
    assert profiler.stop() is None
    assert profiler.summary_line() == "[profile] dummy: no result"


@pytest.mark.parametrize("private_cluster, tag", [(False, "shared cluster"), (True, "private cluster")])
def test_profiler_with_mocked_client(private_cluster, tag):
    """With a client the cluster peak is taken from the MemorySampler samples."""
    client = MagicMock()
    client.scheduler_info.return_value = {"workers": {"w1": {}, "w2": {}, "w3": {}}}

    sampler = MagicMock()
    sampler.samples = {"timeseries": [(0.0, 1024**3), (1.0, 3 * 1024**3), (2.0, 2 * 1024**3)]}

    with patch("distributed.diagnostics.MemorySampler", return_value=sampler):
        profiler = DiagnosticProfiler(name="timeseries", client=client, private_cluster=private_cluster)
        profiler.start()
        result = profiler.stop()

    sampler.sample.assert_called_once_with("timeseries", client=client, interval=1.0)
    assert result["cluster_peak_bytes"] == 3 * 1024**3
    assert result["nworkers"] == 3
    assert result["private_cluster"] is private_cluster

    line = profiler.summary_line()
    assert "cluster_peak_mem=3.0GiB" in line
    assert f"({tag}, 3 workers)" in line


def test_profiler_sampler_failure_is_not_fatal():
    """A sampler that cannot start leaves the driver numbers intact."""
    client = MagicMock()
    with patch("distributed.diagnostics.MemorySampler", side_effect=RuntimeError("no scheduler")):
        profiler = DiagnosticProfiler(name="timeseries", client=client)
        profiler.start()
        result = profiler.stop()

    assert result["driver_peak_rss_bytes"] > 0
    assert result["cluster_peak_bytes"] is None
    assert "cluster_peak_mem" not in profiler.summary_line()
