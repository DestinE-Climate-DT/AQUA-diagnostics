"""Utility for the ocean drift plots."""

import matplotlib.dates as mdates
import numpy as np

# Round year steps for the time axis ticks
YEAR_STEPS = (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500)

# Ticks of the depth axis, which set_depth_axis stretches near the surface
DEPTH_TICKS = (100, 500, 1000, 2000, 3000, 4000, 5000, 6000)


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


def set_depth_axis(ax):
    """Stretch the depth axis of a panel near the surface, where most of the variability is.

    Args:
        ax (matplotlib.axes.Axes): Panel with depth on the y-axis.
    """
    limits = ax.get_ylim()
    ax.set_yscale("function", functions=(np.sqrt, np.square))
    ax.set_ylim(limits)
    low, high = sorted(limits)
    ax.set_yticks([tick for tick in DEPTH_TICKS if low <= tick <= high])
