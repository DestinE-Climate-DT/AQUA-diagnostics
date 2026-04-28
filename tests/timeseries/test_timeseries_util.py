"""Unit tests for aqua.diagnostics.timeseries.util."""

import pandas as pd
import pytest
import xarray as xr

from aqua.diagnostics.timeseries.util import center_timestamp, loop_seasonalcycle

pytestmark = [pytest.mark.diagnostics]


# ======================================================================
# center_timestamp
# ======================================================================
@pytest.mark.parametrize(
    "time,freq,expected",
    [
        ("2020-03-01", "monthly", "2020-03-15 12:00:00"),
        ("2020-01-01", "annual", "2020-07-02 12:00:00"),
    ],
)
def test_center_timestamp(time, freq, expected):
    """Centering aligns monthly timestamps to day 15 12:00 and annual to Jul 2 12:00."""
    assert center_timestamp(pd.Timestamp(time), freq=freq) == pd.Timestamp(expected)


def test_center_timestamp_unsupported_freq_raises():
    with pytest.raises(ValueError, match="not supported"):
        center_timestamp(pd.Timestamp("2020-01-01"), freq="weekly")


# ======================================================================
# loop_seasonalcycle
# ======================================================================
def _monthly_data():
    return xr.DataArray(
        [10, 15, 20, 25, 30, 35, 30, 25, 20, 15, 10, 5],
        dims=["time"],
        coords={"time": pd.date_range("2020-01-01", periods=12, freq="MS")},
    )


def _annual_data():
    return xr.DataArray(
        [10.0, 20.0, 30.0],
        dims=["time"],
        coords={"time": pd.date_range("2020-01-01", periods=3, freq="YS")},
    )


class TestLoopSeasonalcycleValidation:
    """Argument validation branches."""

    def test_data_none_raises(self):
        with pytest.raises(ValueError, match="Data not provided"):
            loop_seasonalcycle(None, "2020-01-01", "2020-12-31", "monthly")

    def test_missing_dates_raise(self):
        data = _monthly_data()
        with pytest.raises(ValueError, match="Start date or end date"):
            loop_seasonalcycle(data, None, "2020-12-31", "monthly")
        with pytest.raises(ValueError, match="Start date or end date"):
            loop_seasonalcycle(data, "2020-01-01", None, "monthly")

    def test_missing_freq_raises(self):
        data = _monthly_data()
        with pytest.raises(ValueError, match="Frequency not provided"):
            loop_seasonalcycle(data, "2020-01-01", "2020-12-31", None)

    def test_unsupported_freq_raises(self):
        data = _monthly_data()
        with pytest.raises(ValueError, match="not supported"):
            loop_seasonalcycle(data, "2020-01-01", "2020-12-31", "weekly")


class TestLoopSeasonalcycleMonthly:
    def test_monthly_loop_repeats_cycle(self):
        data = _monthly_data()
        looped = loop_seasonalcycle(data, "2021-01-01", "2022-12-31", freq="monthly", center_time=False)
        assert len(looped.time) == 24
        assert looped.values[0] == looped.values[12] == 10

    def test_monthly_loop_with_center_time_shifts_to_mid_month(self):
        data = _monthly_data()
        looped = loop_seasonalcycle(data, "2021-01-01", "2021-03-01", freq="monthly", center_time=True)
        # All timestamps centered at day 15, 12:00
        assert all(pd.Timestamp(t).day == 15 and pd.Timestamp(t).hour == 12 for t in looped.time.values)


class TestLoopSeasonalcycleAnnual:
    def test_annual_loop_repeats_mean(self):
        data = _annual_data()
        looped = loop_seasonalcycle(data, "2030-01-01", "2032-01-01", freq="annual", center_time=False)
        assert len(looped.time) == 3
        # All three copies equal the annual mean of [10, 20, 30].
        assert float(looped.values[0]) == pytest.approx(20.0)
        assert float(looped.values[-1]) == pytest.approx(20.0)

    def test_annual_loop_with_center_time_shifts_to_july_2(self):
        data = _annual_data()
        looped = loop_seasonalcycle(data, "2030-01-01", "2031-01-01", freq="annual", center_time=True)
        for t in looped.time.values:
            ts = pd.Timestamp(t)
            assert (ts.month, ts.day, ts.hour) == (7, 2, 12)


class TestLoopSeasonalcycleEdgeCases:
    def test_empty_time_range_returns_empty_along_time(self):
        """A reversed date range produces no timestamps; util returns a zero-length slice."""
        data = _monthly_data()
        looped = loop_seasonalcycle(data, "2022-12-31", "2022-01-01", freq="monthly", center_time=False)
        assert looped.sizes.get("time", 0) == 0
