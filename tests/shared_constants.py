"""Shared constants for the test suite."""

DPI = 50
APPROX_REL = 1e-4
LOGLEVEL = "DEBUG"


def assert_nonempty(path):
    assert path.is_file(), f"File not found: {path}"
    assert path.stat().st_size > 0
