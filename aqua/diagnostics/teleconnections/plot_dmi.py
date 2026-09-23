import matplotlib.pyplot as plt
import xarray as xr

from aqua.core.graphics import indexes_plot, plot_maps, plot_maps_diff, plot_single_map, plot_single_map_diff
from aqua.core.logger import log_configure
from aqua.diagnostics.base import TitleBuilder

from .base import PlotBaseMixin, _homogeneize_maps


class PlotDMI(PlotBaseMixin):
    """
    Class for plotting the DMI index.
    This class inherits from the PlotBaseMixin and implements the necessary methods
    to plot the DMI index.
    """

    def __init__(
        self,
        indexes=None,
        ref_indexes=None,
        outputdir: str = "./",
        rebuild: bool = True,
        loglevel: str = "WARNING",
    ):
        """
        Plot the DMI index.

        Args:
            indexes (list): List of DMI index objects to plot.
            ref_indexes (list): List of reference DMI index objects to plot.
            outputdir (str): Directory to save the plot. Default is './'.
            rebuild (bool): Whether to rebuild the plot if it already exists. Default is True.
            loglevel (str): Logging level. Default is 'WARNING'.
        """
        super().__init__(
            indexes=indexes,
            ref_indexes=ref_indexes,
            diagnostic="dmi",
            outputdir=outputdir,
            rebuild=rebuild,
            loglevel=loglevel,
        )
        self.logger = log_configure(log_name="PlotDMI", log_level=loglevel)

    def plot_index(self, thresh: float = 0.5, labels: list = None):
        """
        Plot the DMI index.

        Args:
            thresh (float): Threshold for highlighting significant values. Default is 0.5.
            labels (list): List of labels for the plot. Default is None.
        """
        # Join the indexes in a single list
        indexes = self.indexes + self.ref_indexes

        labels = super().set_labels() if labels is None else labels

        title = TitleBuilder(
            diagnostic="DMI index", model=self.models, exp=self.exps, ref_model=self.ref_models, ref_exp=self.ref_exps
        ).generate()

        fig, axs = indexes_plot(
            indexes=indexes, thresh=thresh, suptitle=title, ylabel="DMI index", labels=labels, loglevel=self.loglevel
        )

        if isinstance(axs, plt.Axes):
            axs = [axs]

        return fig, axs

    def set_index_description(self):
        return super().set_index_description(index_name="DMI index")

    def plot_maps(
        self,
        maps=None,
        ref_maps=None,
        statistic: str = None,
        vmin: float = None,
        vmax: float = None,
        vmin_diff: float = None,
        vmax_diff: float = None,
        **kwargs,
    ):
        """
        Plot maps for DMI regression/correlation products.

        Args:
            maps (xarray.DataArray or list): Maps to plot.
            ref_maps (xarray.DataArray or list): Reference maps to compare against.
            statistic (str): Name of the statistic to plot.
            vmin (float): Minimum contour color value.
            vmax (float): Maximum contour color value.
            vmin_diff (float): Minimum filled-difference color value.
            vmax_diff (float): Maximum filled-difference color value.
            **kwargs: Additional arguments for the plotting function.

        Returns:
            matplotlib.figure.Figure or None: Figure object if successful, else None.
        """
        map_to_check = maps if isinstance(maps, xr.DataArray) else maps[0]
        var = map_to_check.shortName if hasattr(map_to_check, "shortName") else map_to_check.long_name
        if statistic == "correlation" and vmin is None and vmax is None:
            vmin = -1.0
            vmax = 1.0
            vmin_diff = -0.5
            vmax_diff = 0.5
        elif statistic == "regression" and vmin is None and vmax is None and var == "tos":
            vmin = -1.0
            vmax = 1.0
            vmin_diff = -1.0
            vmax_diff = 1.0

        maps, ref_maps = _homogeneize_maps(maps=maps, ref_maps=ref_maps, var=var)

        # Case 1: no reference maps
        if maps is not None and ref_maps is None:
            # Case 1a: single map
            if isinstance(maps, xr.DataArray):
                title = TitleBuilder(
                    diagnostic=f"DMI {statistic} map ({var})",
                    model=maps.AQUA_model,
                    exp=maps.AQUA_exp,
                    timeseason=getattr(maps, "AQUA_season", None),
                ).generate()

                fig, _ = plot_single_map(
                    data=maps, vmin=vmin, vmax=vmax, title=title, return_fig=True, loglevel=self.loglevel, **kwargs
                )
                return fig

            # Case 1b: multiple maps
            elif isinstance(maps, list):
                titles = []
                for map in maps:
                    title = TitleBuilder(
                        diagnostic=f"DMI {statistic} map ({var})",
                        model=map.AQUA_model,
                        exp=map.AQUA_exp,
                        timeseason=getattr(map, "AQUA_season", None),
                    ).generate()
                    titles.append(title)
                fig = plot_maps(
                    maps=maps, vmin=vmin, vmax=vmax, titles=titles, return_fig=True, loglevel=self.loglevel, **kwargs
                )
                return fig

        # Case 2: reference maps are present
        if ref_maps is not None:
            # Case 2a: one map and one reference map
            if isinstance(maps, xr.DataArray) and isinstance(ref_maps, xr.DataArray):
                title = TitleBuilder(
                    diagnostic=f"DMI {statistic} map ({var})",
                    model=maps.AQUA_model,
                    exp=maps.AQUA_exp,
                    ref_model=ref_maps.AQUA_model,
                    ref_exp=ref_maps.AQUA_exp,
                    timeseason=getattr(maps, "AQUA_season", None),
                ).generate()
                fig, _ = plot_single_map_diff(
                    data=maps,
                    data_ref=ref_maps,
                    vmin_contour=vmin if vmin is not None else None,
                    vmax_contour=vmax if vmax is not None else None,
                    vmin_fill=vmin_diff if vmin_diff is not None else None,
                    vmax_fill=vmax_diff if vmax_diff is not None else None,
                    sym=True if vmax_diff is None and vmin_diff is None else False,
                    sym_contour=True if vmax is None and vmin is None else False,
                    title=title,
                    return_fig=True,
                    loglevel=self.loglevel,
                    **kwargs,
                )
                return fig

            # Case 2b: maps are list and ref_maps is one
            if isinstance(maps, list) and isinstance(ref_maps, xr.DataArray):
                titles = []
                for map in maps:
                    title = f"{map.AQUA_model} {map.AQUA_exp}"
                    titles.append(title)
                title = TitleBuilder(
                    diagnostic=f"DMI {statistic} map ({var})",
                    ref_model=ref_maps.AQUA_model,
                    ref_exp=ref_maps.AQUA_exp,
                    timeseason=getattr(ref_maps, "AQUA_season", None),
                ).generate()

                maps_ref = [ref_maps] * len(maps)
                fig = plot_maps_diff(
                    maps=maps,
                    maps_ref=maps_ref,
                    vmin_contour=vmin if vmin is not None else None,
                    vmax_contour=vmax if vmax is not None else None,
                    vmin_fill=vmin_diff if vmin_diff is not None else None,
                    vmax_fill=vmax_diff if vmax_diff is not None else None,
                    sym=True if vmax_diff is None and vmin_diff is None else False,
                    sym_contour=True if vmax is None and vmin is None else False,
                    titles=titles,
                    title=title,
                    return_fig=True,
                    loglevel=self.loglevel,
                    **kwargs,
                )
                return fig

            # Case 2c: maps is one and ref_maps are list
            if isinstance(maps, xr.DataArray) and isinstance(ref_maps, list):
                titles = []
                for map in ref_maps:
                    title = f"Compared to {map.AQUA_model} {map.AQUA_exp}"
                    titles.append(title)
                title = TitleBuilder(
                    diagnostic=f"DMI {statistic} map ({var})",
                    model=maps.AQUA_model,
                    exp=maps.AQUA_exp,
                    timeseason=getattr(maps, "AQUA_season", None),
                ).generate()

                maps = [maps] * len(ref_maps)
                fig = plot_maps_diff(
                    maps=maps,
                    maps_ref=ref_maps,
                    vmin_contour=vmin if vmin is not None else None,
                    vmax_contour=vmax if vmax is not None else None,
                    vmin_fill=vmin_diff if vmin_diff is not None else None,
                    vmax_fill=vmax_diff if vmax_diff is not None else None,
                    sym=True if vmax_diff is None and vmin_diff is None else False,
                    sym_contour=True if vmax is None and vmin is None else False,
                    titles=titles,
                    title=title,
                    return_fig=True,
                    loglevel=self.loglevel,
                    **kwargs,
                )
                return fig

            # Case 2d: both maps and ref_maps are lists
            if isinstance(maps, list) and isinstance(ref_maps, list):
                self.logger.error("Both maps and ref_maps are lists. This case is not implemented yet.")
                return None

    def set_map_description(self, maps=None, ref_maps=None, statistic: str = None):
        """
        Set the description for DMI maps.

        Args:
            maps (xarray.DataArray or list): Maps being plotted.
            ref_maps (xarray.DataArray or list): Reference maps.
            statistic (str): Name of the statistic being plotted.

        Returns:
            str: Description text.
        """
        return super().set_map_description(maps=maps, ref_maps=ref_maps, statistic=statistic, telecname="DMI")
