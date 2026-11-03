"""Time utilities for AQUA diagnostics"""

import pandas as pd

from aqua.core.util import pandas_freq_to_string, xarray_to_pandas_freq


def start_end_dates(startdate=None, enddate=None, start_std=None, end_std=None):
    """
    Evaluate start and end dates for data retrieve so Reader call covers
    also the std-dates when set.
    They should be of the form 'YYYY-MM-DD' or 'YYYYMMDD'.
    The function will translate them to the form 'YYYY-MM-DD' and
    then use pandas Timestamp to evaluate the minimum and maximum
    dates.

    Args:
        startdate (str): start date for the data retrieve
        enddate (str): end date for the data retrieve
        start_std (str): start date for the standard deviation data retrieve
        end_std (str): end date for the standard deviation data retrieve

    Returns:
        tuple: ``(start_retrieve, end_retrieve)`` as ``pandas.Timestamp`` or ``None``
        for open-ended bounds.
    """
    # Convert to pandas Timestamp
    startdate = pd.Timestamp(startdate) if startdate else None
    enddate = pd.Timestamp(enddate) if enddate else None
    start_std = pd.Timestamp(start_std) if start_std else None
    end_std = pd.Timestamp(end_std) if end_std else None

    start_retrieve = min((t for t in (startdate, start_std) if t is not None), default=None)
    end_retrieve = max((t for t in (enddate, end_std) if t is not None), default=None)

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
    if freq == "annual":
        return pd.Timestamp(year=startdate.year, month=1, day=1, hour=0, minute=0, second=0)
    elif freq == "monthly":
        return pd.Timestamp(year=startdate.year, month=startdate.month, day=1, hour=0, minute=0, second=0)
    else:
        raise ValueError(f"Unsupported frequency '{freq}'. Only 'monthly' and 'annual' are supported.")


def round_enddate(enddate, freq="monthly"):
    """
    Round the end date to the end of the month or year.

    Args:
        enddate (pd.Timestamp): end date
        freq (str): frequency ('monthly' or 'annual'). Default is 'monthly'.

    Returns:
        pd.Timestamp: rounded end date
    """
    if freq == "annual":
        return pd.Timestamp(year=enddate.year, month=12, day=31, hour=23, minute=59, second=59)
    elif freq == "monthly":
        return (
            pd.Timestamp(year=enddate.year, month=enddate.month, day=1, hour=0, minute=0, second=0)
            + pd.DateOffset(months=1)
            - pd.Timedelta(seconds=1)
        )
    else:
        raise ValueError(f"Unsupported frequency '{freq}'. Only 'monthly' and 'annual' are supported.")


def effective_bounds(data):
    """
    Evaluate the time bounds of the data actually available.
    For monthly and annual data the comparison bounds are widened to the
    whole first/last period, to avoid clipping when the user asks e.g. for
    an end-of-month date while the timestamps are at the start of the month.

    Args:
        data (xarray.Dataset or xarray.DataArray): data with a time coordinate

    Returns:
        tuple: ``(eff_start, eff_end, start_bound, end_bound)``, where ``eff_*`` are
        the first and last timestamps of the data and ``*_bound`` the
        ``pandas.Timestamp`` to compare the requested dates with.
    """
    eff_start = data.time.values[0]
    eff_end = data.time.values[-1]

    freq = pandas_freq_to_string(xarray_to_pandas_freq(data))
    if freq in ("monthly", "annual"):
        start_bound = round_startdate(pd.Timestamp(eff_start), freq=freq)
        end_bound = round_enddate(pd.Timestamp(eff_end), freq=freq)
    else:
        start_bound = pd.Timestamp(eff_start)
        end_bound = pd.Timestamp(eff_end)

    return eff_start, eff_end, start_bound, end_bound
