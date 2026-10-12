"""Utility for the ocean drift plots."""

import matplotlib.dates as mdates

# Round year steps for the time axis ticks
YEAR_STEPS = (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500)


def set_year_ticks(ax, max_intervals: int = 5):
    """Label the time axis of a panel with years only (yyyy).

    Args:
        ax (matplotlib.axes.Axes): Panel with time on the x-axis.
        max_intervals (int): Maximum number of intervals between ticks. Default is 5.
    """
    start, end = sorted(mdates.num2date(limit).year for limit in ax.get_xlim())
    if start == end:
        # No year start may fall within the span, so the year is ticked once
        ax.set_xticks([sum(ax.get_xlim()) / 2])
    else:
        step = next((step for step in YEAR_STEPS if (end - start) / step <= max_intervals), YEAR_STEPS[-1])
        ax.xaxis.set_major_locator(mdates.YearLocator(base=step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def add_row_label(ax, text: str):
    """Write the label of a row of panels (e.g. the drift type) vertically, left of the panel.

    Args:
        ax (matplotlib.axes.Axes): First panel of the row.
        text (str): Label of the row.
    """
    ax.text(-0.3, 0.5, text, fontsize=15, color="dimgray", rotation=90, transform=ax.transAxes, ha="center", va="center")
