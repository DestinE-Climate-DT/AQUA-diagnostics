"""Module for plotting multiple vertical line profiles in a grid layout."""

import matplotlib.pyplot as plt
import xarray as xr

from aqua.core.graphics import ConfigStyle, plot_vertical_lines
from aqua.core.logger import log_configure
from aqua.diagnostics.base.defaults import DEFAULT_OCEAN_VERT_COORD


def plot_multi_vertical_lines(
    data_list: list,
    ref_data_list: list,
    nrows: int,
    ncols: int,
    figsize: tuple = None,
    data_label: str = None,
    obs_label: str = None,
    variables: list = None,
    vert_coord: str = DEFAULT_OCEAN_VERT_COORD,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    style=None,
    text: list[str] = None,
    xtext: list = None,
    ytext: list = None,
    title: str = None,
    titles: list = None,
    return_fig=True,
    loglevel="WARNING",
):
    """Plot multiple vertical line profiles in a grid layout.

    Rows correspond to datasets in ``data_list``; columns correspond to ``variables``.
    Legend is shown on the first column only. ``xtext``/``ytext`` override axis labels
    set by ``plot_vertical_lines`` when provided.

    Args:
        data_list (list): Model datasets to plot (one row each).
        ref_data_list (list): Reference datasets overlaid as dashed lines (same length as ``data_list``).
        nrows (int): Number of rows in the subplot grid.
        ncols (int): Number of columns in the subplot grid.
        figsize (tuple, optional): Figure size (width, height). Auto-set if None.
        data_label (str, optional): Legend label for the model (solid) line.
        obs_label (str, optional): Legend label for the reference (dashed) line.
        variables (list, optional): Variable names to plot (one column each).
        vert_coord (str, optional): Vertical coordinate name. Default is DEFAULT_OCEAN_VERT_COORD.
        fig (plt.Figure, optional): Existing figure to plot on.
        ax (plt.Axes, optional): Existing axes to plot on.
        style (str, optional): AQUA plot style name.
        text (list[str], optional): Side annotations per subplot (length = nrows × len(variables)).
        xtext (list, optional): X-axis labels, one per variable/column.
        ytext (list, optional): Y-axis labels, one per variable/column.
        title (str, optional): Figure suptitle.
        titles (list, optional): Subplot titles, one per variable/column.
        return_fig (bool, optional): If True, return the Figure.
        loglevel (str, optional): Logging level.

    Returns:
        matplotlib.figure.Figure or None: The Figure if ``return_fig`` is True, otherwise None.

    """
    logger = log_configure(loglevel, "plot_multi_vertical_lines")
    ConfigStyle(style=style, loglevel=loglevel)

    if all(isinstance(data_map, xr.Dataset) for data_map in data_list):
        # nrows = 1  # len(data_list)
        # ncols = len(variables)
        figsize = figsize if figsize is not None else (ncols * 2, nrows * 3 + 1)
        logger.debug("Creating a %d x %d grid with figsize %s", nrows, ncols, figsize)

    fig = plt.figure(figsize=figsize)
    spec = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.2, hspace=0.1)

    for j in range(nrows):
        for i, var in enumerate(variables):
            k = j * len(variables) + i
            ax = fig.add_subplot(spec[j, i])
            logger.debug("Creating subplot for variable %s at (%d, %d)", var, j, i)

            fig, ax = plot_vertical_lines(
                data=data_list[j][var],
                ref_data=ref_data_list[j][var] if ref_data_list else None,
                labels=data_label,
                ref_label=obs_label,
                lev_name=vert_coord,
                invert_yaxis=True,
                return_fig=True,
                ax=ax,
                fig=fig,
                loglevel=loglevel,
            )
            if titles:
                ax.set_title(titles[i], fontsize=12)
            if xtext:
                ax.set_xlabel(xtext[i])
            if ytext:
                ax.set_ylabel(ytext[i])
            if i == 0:  # only first column
                ax.legend(fontsize=14)
            else:
                ax.legend().remove()

            if text:
                logger.debug("Adding text in the plot: %s", text)
                ax.text(
                    -0.3,
                    0.33,
                    text[k],
                    fontsize=15,
                    color="dimgray",
                    rotation=90,
                    transform=ax.transAxes,
                    ha="center",
                )

    # Adjust overall layout
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.05, right=0.95)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for title

    if title:
        logger.debug("Setting super title to %s", title)
        fig.suptitle(title, fontsize=ncols * 5, y=1.05)

    if return_fig:
        return fig
