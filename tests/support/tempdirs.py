"""
Temporary directory utilities for AQUA-diagnostics test suite.

Manages per-worker OS temp directories under pytest-xdist to avoid CDO/regrid
contention, and provides helpers for diagnostic output isolation.
"""

import os
import shutil
import tempfile

_WORKER_TMPDIR_ATTR = "_worker_tmpdir"


def configure_worker_tmpdir(config) -> None:
    """Set per-worker TMPDIR to avoid CDO/temp contention in parallel runs."""
    workerinput = getattr(config, "workerinput", None)
    if workerinput is None:
        return

    worker_id = workerinput.get("workerid", "master")
    worker_tmpdir = tempfile.mkdtemp(prefix=f"aqua_diag_pytest_{worker_id}_")
    os.environ["TMPDIR"] = worker_tmpdir
    setattr(config, _WORKER_TMPDIR_ATTR, worker_tmpdir)


def cleanup_worker_tmpdir(session) -> None:
    """Remove the per-worker TMPDIR created in pytest_configure."""
    worker_tmpdir = getattr(session.config, _WORKER_TMPDIR_ATTR, None)
    if not worker_tmpdir:
        return

    shutil.rmtree(worker_tmpdir, ignore_errors=True)
