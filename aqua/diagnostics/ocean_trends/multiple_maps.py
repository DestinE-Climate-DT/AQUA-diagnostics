"""Module for plotting multiple 2D trend maps in a grid layout."""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import ticker

from aqua.core.graphics import ConfigStyle
from aqua.core.graphics.single_map import plot_single_map
from aqua.core.logger import log_configure
from aqua.core.util import evaluate_colorbar_limits, generate_colorbar_ticks

xr.set_options(keep_attrs=True)


def plot_maps(
    maps: list[xr.DataArray],
    style=None,
    title: str = None,
    titles: list = None,
    proj: ccrs.Projection = ccrs.PlateCarree(),
    extent: list = None,
    col_vmin: list = None,
    col_vmax: list = None,
    sym: bool = True,
    cmap: str = "RdBu_r",
    cbar_labels: list = None,
    ytext: list = None,
    nrows: int = 6,
    ncols: int = 2,
    transform_first: bool = False,
    cyclic_lon: bool = True,
    loglevel: str = "WARNING",
    return_fig: bool = True,
    nlevels: int = 12,
    **kwargs,
):
    """Plot multiple 2D maps (xarray DataArrays) in a grid layout.

    Parameters
    ----------
    maps : list[xr.DataArray]
        List of xarray DataArrays to plot.
    style : str, optional
        Plot style preset or name. If None, uses the default AQUA style.
    title : str, optional
        Overall figure title.
    titles : list[str], optional
        List of subplot titles corresponding to each DataArray in `maps`.
    proj : cartopy.crs.Projection, optional
        Map projection for plotting (default: PlateCarree).
    extent : list[float], optional
        Geographic extent as [lon_min, lon_max, lat_min, lat_max].
    cmap : str, optional
        Colormap name to use (default: "RdBu_r").
    cbar_labels : list[str], optional
        List of colorbar labels for each subplot.
    sym : bool, optional
        If True, use symmetric colorbar limits around zero (default: False).
    col_vmin : list[float], optional
        Per-column minimum colorbar values. Overrides automatic limits when provided.
    col_vmax : list[float], optional
        Per-column maximum colorbar values. Overrides automatic limits when provided.
    ytext : list[str], optional
        Text annotations to place on the y-axis of each subplot.
    nrows : int, optional
        Number of rows in the subplot grid (default: 6).
    ncols : int, optional
        Number of columns in the subplot grid (default: 2).
    transform_first : bool, optional
        If True, apply coordinate transformation before plotting.
    cyclic_lon : bool, optional
        Whether to make longitude cyclic for continuous global maps.
    loglevel : str, optional
        Logging verbosity level (default: "WARNING").
    return_fig : bool, optional
        If True, return the Matplotlib figure object (default: True).
    nlevels : int, optional
        Number of discrete color levels in the colormap (default: 12).
    **kwargs
        Additional keyword arguments passed to the underlying contour or pcolormesh function.

    Returns
    -------
    matplotlib.figure.Figure or None
        The Matplotlib figure if `return_fig=True`, otherwise None.

    Notes
    -----
    This function provides a convenient way to visualize multiple geospatial fields
    using Cartopy. Handles projection setup, cyclic longitude wrapping, and optional
    labeling automatically.

    """
    logger = log_configure(loglevel, "plot_maps")
    ConfigStyle(style=style, loglevel=loglevel)

    if maps is None or any(not isinstance(data_map, xr.DataArray) for data_map in maps):
        raise ValueError("Maps should be a list of xarray.DataArray")
    if not maps or len(maps) != nrows * ncols:
        raise ValueError("The number of maps must equal nrows * ncols.")
    logger.debug("Loading maps")
    maps = [data_map.compute() for data_map in maps]

    figsize = (ncols * 4.5, nrows * 2.5 + 1.4)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        subplot_kw={"projection": proj},
        squeeze=False,
    )
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.9 / figsize[1], top=1 - 0.8 / figsize[1], hspace=0.15)
    axs = axs.flatten()

    for i in range(len(maps)):
        row = i // ncols
        col = i % ncols

        vmin = col_vmin[col] if col_vmin is not None else None
        vmax = col_vmax[col] if col_vmax is not None else None
        if vmin is None or vmax is None:
            col_maps = [maps[j] for j in range(len(maps)) if j % ncols == col and bool(maps[j].notnull().any())]
            # Keep empty panels aligned with the other variables. An entirely
            # empty column has no meaningful automatic scale.
            auto_min, auto_max = evaluate_colorbar_limits(maps=col_maps, sym=sym) if col_maps else (-1, 1)
            vmin = auto_min if vmin is None else vmin
            vmax = auto_max if vmax is None else vmax
        if sym:
            vmin, vmax = -max(abs(vmin), abs(vmax)), max(abs(vmin), abs(vmax))

        ticks = np.linspace(vmin, vmax, int(nlevels / 2) + 1)
        if len(ticks) < 3:  # ensure at least 3 ticks for colorbar
            ticks = np.linspace(vmin, vmax, 3)
        logger.debug("Colorbar limits for map %d: vmin=%s, vmax=%s", i, vmin, vmax)

        logger.debug("Plotting map %d", i)
        fig, ax = plot_single_map(
            data=maps[i],
            contour=True,
            proj=proj,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            nlevels=nlevels,
            title=titles[i] if titles is not None else None,
            cmap=cmap,
            cbar=False,
            transform_first=transform_first,
            add_land=False,
            return_fig=True,
            cyclic_lon=cyclic_lon,
            fig=fig,
            ax=axs[i],
            gridlines=False,
            loglevel=loglevel,
            ax_pos=(nrows, ncols, i + 1),
            ticks_rounding=0,
            coastlines=False,
            **kwargs,
        )
        ax.set_aspect("auto")  # NEW: stretch plot to fill subplot
        ax.set_facecolor(color="grey")  # adding land
        # Geographic labels are supplied by the gridliner below.
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.3)

        gl.xlabel_style = {"color": "gray", "size": 8}
        gl.ylabel_style = {"color": "gray", "size": 8}

        gl.top_labels = False
        gl.right_labels = False

        if row == nrows - 1:
            gl.bottom_labels = True
        else:
            gl.bottom_labels = False

        gl.left_labels = col == 0

        if ytext and ytext[i]:
            ax.text(
                -0.18,
                0.5,
                ytext[i],
                fontsize=11,
                color="dimgray",
                rotation=90,
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
        if row == nrows - 1:
            if ax.collections:
                mappable = ax.collections[-1]
            elif ax.images:
                mappable = ax.images[-1]
            else:
                logger.warning("No mappable object found for subplot %d", i)
                continue

            # Update mappable normalization and cmap
            mappable.set_norm(plt.Normalize(vmin=vmin, vmax=vmax))
            mappable.set_cmap(cmap)

            pos = ax.get_position()
            cax = fig.add_axes([pos.x0, pos.y0 - 0.5 / figsize[1], pos.width, 0.14 / figsize[1]])
            cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
            if cbar_labels is not None:
                cbar.set_label(cbar_labels[i], fontsize=9)
            cbar.ax.tick_params(labelsize=8)

            cbar_ticks = generate_colorbar_ticks(
                vmin=vmin,
                vmax=vmax,
                sym=sym,
                nlevels=4,
                ticks_rounding=4,
                loglevel=loglevel,
            )
            cbar.set_ticks(cbar_ticks)
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))  # always scientific notation
            cbar.ax.xaxis.set_major_formatter(formatter)
            cbar.ax.xaxis.offsetText.set_fontsize(8)
        if titles and titles[i]:
            ax.set_title(titles[i], fontsize=9)

    if title:
        fig.suptitle(title, fontsize=12, y=1 - 0.15 / figsize[1])

    if return_fig:
        return fig
