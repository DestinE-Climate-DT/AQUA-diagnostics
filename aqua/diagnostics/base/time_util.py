"""Time utilities for AQUA diagnostics"""

import pandas as pd

from aqua.core.util import pandas_freq_to_string, xarray_to_pandas_freq


def _end_timestamp(date):
    """Convert an end date to an inclusive timestamp."""
    if date is None:
        return None
    if isinstance(date, str):
        return pd.Period(date).end_time
    return pd.Timestamp(date)


def _period(date, freq):
    """Return the monthly or annual period containing a date."""
    period_freq = {"monthly": "M", "annual": "Y"}.get(freq)
    if period_freq is None:
        raise ValueError(f"Unsupported frequency '{freq}'. Only 'monthly' and 'annual' are supported.")
    return pd.Period(date, freq=period_freq)


def start_end_dates(startdate=None, enddate=None, start_std=None, end_std=None):
    """
    Evaluate start and end dates for data retrieve so Reader call covers
    also the std-dates when set.
    String end dates are interpreted as inclusive periods, consistently
    with ``Reader.seldate()``. For example, ``2020-01-02`` includes the
    complete day rather than stopping at midnight.

    Args:
        startdate (str): start date for the data retrieve
        enddate (str): end date for the data retrieve
        start_std (str): start date for the standard deviation data retrieve
        end_std (str): end date for the standard deviation data retrieve

    Returns:
        tuple: ``(start_retrieve, end_retrieve)`` as ``pandas.Timestamp``
        or ``None`` for open-ended bounds.

    Raises:
        ValueError: If only one standard-deviation date is provided.
    """
    startdate = pd.Timestamp(startdate) if startdate else None
    enddate = _end_timestamp(enddate)
    start_std = pd.Timestamp(start_std) if start_std else None
    end_std = _end_timestamp(end_std)

    if (start_std is None) != (end_std is None):
        raise ValueError("std_startdate and std_enddate must be provided together.")

    if start_std is None:
        return startdate, enddate

    # None represents an open analysis bound, so the union must remain open.
    start_retrieve = None if startdate is None else min(startdate, start_std)
    end_retrieve = None if enddate is None else max(enddate, end_std)

    return start_retrieve, end_retrieve


def round_startdate(startdate, freq="monthly"):
    """
    Round the start date to the start of the month or year.

    Args:
        startdate (pd.Timestamp): start date
        freq (str): frequency ('monthly' or 'annual'). Default is 'monthly'.

    Returns:
        pd.Timestamp: rounded start date
    """
    return _period(startdate, freq).start_time


def round_enddate(enddate, freq="monthly"):
    """
    Round the end date to the end of the month or year.

    Args:
        enddate (pd.Timestamp): end date
        freq (str): frequency ('monthly' or 'annual'). Default is 'monthly'.

    Returns:
        pd.Timestamp: rounded end date
    """
    return _period(enddate, freq).end_time


def available_time_bounds(data):
    """Return the logical first and last available dates.

    Monthly and annual timestamps are expanded to the start and end of
    their represented period. Other frequencies retain their exact first
    and last timestamps.

    Args:
        data (xarray.Dataset or xarray.DataArray): Time-indexed data.

    Returns:
        tuple: The first and last available dates as ``pandas.Timestamp``.
    """
    start_bound = pd.Timestamp(data.time.values[0])
    end_bound = pd.Timestamp(data.time.values[-1])
    freq = pandas_freq_to_string(xarray_to_pandas_freq(data))

    if freq in ("monthly", "annual"):
        start_bound = round_startdate(start_bound, freq=freq)
        end_bound = round_enddate(end_bound, freq=freq)

    return start_bound, end_bound
