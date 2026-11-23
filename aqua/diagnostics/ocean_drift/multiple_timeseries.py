"""Module to plot multiple timeseries data.

This function is custom for the Ocean Drift diagnostics in AQUA.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from aqua.core.graphics import ConfigStyle, plot_timeseries
from aqua.core.logger import log_configure
from aqua.diagnostics.base.defaults import DEFAULT_OCEAN_VERT_COORD

from .util import add_row_label, set_year_ticks


def plot_multi_timeseries(
    maps: list,
    levels: list = None,
    line_plot_colours: list = None,
    figsize: tuple = None,
    variables: list = None,
    vert_coord: str = DEFAULT_OCEAN_VERT_COORD,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    style=None,
    text: list[float] = None,
    title: str = None,
    titles: list[str] = None,
    ylabels: list[str] = None,
    return_fig=True,
    loglevel="WARNING",
    **kwargs,
):
    """Plot multiple time series (e.g., at different levels or for different variables) in a grid layout.

    Args:
        maps (list): List of xarray datasets containing the data to be plotted.
        levels (list, optional): List of levels to plot for each variable (used for labeling and selection).
        line_plot_colours (list, optional): List of colors for the lines in each subplot.
        figsize (tuple, optional): Size of the figure (width, height). If None, it is set automatically.
        variables (list, optional): List of variable names to plot from each dataset.
        vert_coord (str, optional): Name of the vertical dimension coordinate. Default is DEFAULT_OCEAN_VERT_COORD.
        fig (plt.Figure, optional): Matplotlib Figure to plot on. If None, a new figure is created.
        ax (plt.Axes, optional): Matplotlib Axes to plot on. If None, new axes are created.
        style (str, optional): Plot style to use (default is AQUA style).
        text (list, optional): List of text annotations for each subplot.
        title (str, optional): Title for the entire figure.
        titles (list of str, optional): List of titles for each subplot. If None, no titles will be set.
        ylabels (list of str, optional): List of y-axis labels for each subplot, replacing the default
            ones built from the data attributes. A None entry leaves the y-axis of that subplot unlabelled.
        return_fig (bool, optional): If True, return the matplotlib Figure object.
        loglevel (str, optional): Logging level for the function.
        **kwargs: Additional keyword arguments passed to the plotting function.

    Returns:
        matplotlib.figure.Figure or None: The matplotlib Figure object if return_fig is True, otherwise None.

    """
    logger = log_configure(loglevel, "plot_multi_hovmoller")
    ConfigStyle(style=style, loglevel=loglevel)

    if all(isinstance(data_map, xr.Dataset) for data_map in maps):
        nrows = len(maps)
        ncols = len(variables)
        figsize = figsize if figsize is not None else (ncols * 6, nrows * 5 + 1)
        logger.debug("Creating a %d x %d grid with figsize %s", nrows, ncols, figsize)

    # Yearly means once the record is longer than 5 years, so the lines stay readable.
    span = maps[0].time.values[-1].astype("datetime64[M]") - maps[0].time.values[0].astype("datetime64[M]")
    if span > np.timedelta64(60, "M"):
        logger.info("Series is longer than 5 years, plotting yearly means")
        maps = [data.resample(time="1YE").mean(keep_attrs=True) for data in maps]

    fig = plt.figure(figsize=figsize)
    spec = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.28, hspace=0.55)

    data_labels = [str(x) for x in levels]
    row_axes = []

    for j in range(nrows):
        axes_in_row = []
        for i, var in enumerate(variables):
            k = j * len(variables) + i
            ax = fig.add_subplot(spec[j, i])
            logger.debug("Creating subplot for variable %s at (%d, %d)", var, j, i)

            fig, ax = plot_timeseries(
                [maps[j][var].sel({vert_coord: level}) for level in maps[j][var][vert_coord]],
                data_labels=data_labels if i == 0 else None,
                legend=False,
                title=titles[k] if titles else None,
                ax=ax,
                fig=fig,
                loglevel=loglevel,
                colors=line_plot_colours,
            )
            drift_type = maps[j].attrs.get("AQUA_ocean_drift_type", "full")
            if drift_type != "full":
                limit = np.nanmax(np.abs(maps[j][var].values))
                ax.set_ylim(-limit, limit)
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

            if titles and titles[k]:
                ax.title.set_fontsize(13)

            if text and text[k]:
                logger.debug("Setting text position for %s", text[k])
                ax.text(
                    -0.3,
                    0.5,
                    text[k],
                    fontsize=15,
                    color="dimgray",
                    rotation=90,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
            axes_in_row.append(ax)
        row_axes.append(axes_in_row)

    # Layout first, then place one legend under each row using the final axes positions.
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.92)
    for axes_in_row in row_axes:
        handles, labels = axes_in_row[0].get_legend_handles_labels()
        boxes = [row_ax.get_position() for row_ax in axes_in_row]
        row_width = boxes[0].x1 - boxes[0].x0
        x_center = ((boxes[0].x0 + boxes[-1].x1) / 2 - boxes[0].x0) / row_width
        axes_in_row[0].legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(x_center, -0.28),
            bbox_transform=axes_in_row[0].transAxes,
            ncol=len(labels),
            frameon=False,
            fontsize=9,
            borderaxespad=0,
        )

    if title:
        logger.debug("Setting super title to %s", title)
        fig.suptitle(title, fontsize=ncols * 10)

    if return_fig:
        return fig
