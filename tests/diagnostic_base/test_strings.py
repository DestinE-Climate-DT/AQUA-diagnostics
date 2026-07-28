"""Tests for the string utilities in aqua.diagnostics.base.strings."""

import pytest

from aqua.diagnostics.base import collapse_era5_duplicate

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]


@pytest.mark.parametrize(
    "text,expected",
    [
        # 'ERA5 era5' collapses to 'ERA5'
        ("Bias vs ERA5 era5", "Bias vs ERA5"),
        ("ERA5 era5 for global", "ERA5 for global"),
        # No 'ERA5 era5' pattern: string returned unchanged (no-op)
        ("Bias vs IFS hist", "Bias vs IFS hist"),
        ("ERA5 only", "ERA5 only"),
        # 'ERA5 era5-hpz3' must be preserved (hyphenated experiment is not the duplicate)
        ("ERA5 era5-hpz3 r1", "ERA5 era5-hpz3 r1"),
        # Falsy inputs are returned unchanged (None-safe)
        ("", ""),
        (None, None),
    ],
)
def test_collapse_era5_duplicate(text, expected):
    assert collapse_era5_duplicate(text) == expected
