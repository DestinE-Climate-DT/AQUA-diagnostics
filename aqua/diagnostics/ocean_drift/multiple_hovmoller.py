"""Module to plot multiple Hovmoller data.

This function is custom for the Ocean Drift diagnostics in AQUA.
"""

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib import ticker

from aqua.core.graphics import ConfigStyle, plot_hovmoller
from aqua.core.logger import log_configure

from .util import add_row_label, set_year_ticks


def plot_multi_hovmoller(
    maps: list,
    contour: bool = True,
    sym: bool = False,
    figsize: tuple = None,
    variables: list = None,
    invert_space_coord=True,
    style=None,
    vmin: list[float] = None,
    vmax: list[float] = None,
    cmap: list[str] = None,
    text: list[float] = None,
    nlevels: int = 21,
    title: str = None,
    titles: list[str] = None,
    cbar_labels: list[str] = None,
    return_fig=True,
    loglevel="WARNING",
):
    """Plot multiple hovmoller diagrams in a grid layout.

    Args:
        maps (list): List of xarray datasets containing the data to be plotted.
        contour (bool): Whether to plot contours. Default is True.
        sym (bool): Whether to use symmetric color limits. Default is False.
        figsize (tuple): Size of the figure. Default is None, which will be
            set based on the number of variables and datasets.
        variables (list): List of variable names to be plotted.
            If None, all variables in the datasets will be used.
        invert_space_coord (bool): Whether to invert the space coordinate. Default is True.
        style (str): Style to be used for the plot. Default is None. This will use the default AQUA style.
        vmin (list[float]): List of minimum color limits for each subplot.
            If None, or None for a subplot, the limits are evaluated from its data.
        vmax (list[float]): List of maximum color limits for each subplot.
            If None, or None for a subplot, the limits are evaluated from its data.
        cmap (list[str]): List of colormaps for each subplot. If None, the diverging PuOr_r is used.
        text (list[float]): List of text values to be displayed on the plot for each variable.
        nlevels (int): Number of contour levels. Default is 21, i.e. 20 color intervals.
        title (str): Title for the entire figure. Default is None.
        titles (list[str]): List of titles for each subplot. If None, no titles will be set.
        cbar_labels (list[str]): List of colorbar labels for each subplot.
            If None, or None for a subplot, its colorbar has no label.
        return_fig (bool): Whether to return the figure object. Default is True.
        loglevel (str): Logging level. Default is "WARNING".

    Returns:
        plt.Figure: The matplotlib figure object containing the hovmoller plots.

    """
    logger = log_configure(loglevel, "plot_multi_hovmoller")
    ConfigStyle(style=style, loglevel=loglevel)

    if all(isinstance(data_map, xr.Dataset) for data_map in maps):
        nrows = len(maps)
        ncols = len(variables)
        figsize = figsize if figsize is not None else (ncols * 6, nrows * 5 + 1)
        logger.debug("Creating a %d x %d grid with figsize %s", nrows, ncols, figsize)

    fig = plt.figure(figsize=figsize, layout="constrained")
    spec = fig.add_gridspec(nrows=nrows, ncols=ncols)

    for j in range(nrows):
        for i, var in enumerate(variables):
            k = j * len(variables) + i
            ax = fig.add_subplot(spec[j, i])
            logger.debug("Creating subplot for variable %s at (%d, %d)", var, j, i)

            fig, ax = plot_hovmoller(
                data=maps[j][var],
                box_text=False,
                dim=None,
                contour=contour,
                sym=sym,
                vmin=vmin[k] if vmin else None,
                vmax=vmax[k] if vmax else None,
                cmap=cmap[k] if cmap else "PuOr_r",
                nlevels=nlevels,
                invert_space_coord=invert_space_coord,
                cbar=False,
                title=titles[k] if titles else None,
                return_fig=True,
                ax=ax,
                fig=fig,
                loglevel=loglevel,
            )
            set_year_ticks(ax)
            if i > 0:
                # Same depth axis as the first column: its label would only crowd the colorbar on the left
                ax.set_ylabel("")
            if titles and titles[k]:
                ax.title.set_fontsize(13)
            if text and text[k]:
                add_row_label(ax, text[k])
            # Retrieve last plotted object for colorbar (QuadMesh or ContourSet)
            if ax.collections:
                mappable = ax.collections[-1]
            elif ax.images:
                mappable = ax.images[-1]
            else:
                logger.warning("No mappable object found for subplot (%d, %d)", j, i)
                continue

            # Colorbar next to its panel, placed by the constrained layout
            cbar = fig.colorbar(mappable, ax=ax, orientation="vertical", pad=0.02)
            # Round tick values, not tied to the boundaries of the color intervals and
            # kept within the color limits, so that none is drawn next to an extension
            low, high = cbar.norm.vmin, cbar.norm.vmax
            tolerance = 1e-6 * (high - low)
            ticks = ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(low, high)
            cbar.set_ticks([tick for tick in ticks if low - tolerance <= tick <= high + tolerance])
            if cbar_labels and cbar_labels[k]:
                cbar.set_label(cbar_labels[k])

    if title:
        logger.debug("Setting super title to %s", title)
        fig.suptitle(title, fontsize=ncols * 10)

    if return_fig:
        return fig
