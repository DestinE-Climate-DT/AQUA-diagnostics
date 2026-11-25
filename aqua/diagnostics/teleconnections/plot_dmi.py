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
        cmap: str = "RdBu_r",
        **kwargs,
    ):
        """
        Plot the maps for the DMI products.

        Args:
            maps (list): List of maps to plot.
            ref_maps (list): List of reference maps to plot.
            statistic (str): Statistic to plot. Default is None.
            vmin (float): Minimum value for the color value. Default is None.
            vmax (float): Maximum value for the color value. Default is None.
            vmin_diff (float): Minimum value for the color value for the difference. Default is None.
            vmax_diff (float): Maximum value for the color value for the difference. Default is None.
            cmap (str): Colormap to use for the plots. Default is 'RdBu_r'.
            **kwargs: Additional arguments for the plotting function.

        Returns:
            fig: Figure object.
        """
        map_to_check = maps if isinstance(maps, xr.DataArray) else maps[0]

        # This var is used by _homogeneize_maps to eventually convert units.
        # The label for title and colorbar is set by _homogeneize_maps, which returns var_label.
        var = map_to_check.shortName if hasattr(map_to_check, "shortName") else map_to_check.long_name
        if statistic == "correlation":
            vmin = -1.0
            vmax = 1.0
            vmin_diff = -0.5
            vmax_diff = 0.5
        elif statistic == "regression" and vmin is None and vmax is None:
            vmin = -1.0
            vmax = 1.0
            vmin_diff = -1.0
            vmax_diff = 1.0

        maps, ref_maps, var_label = _homogeneize_maps(maps=maps, ref_maps=ref_maps, var=var)

        # Case 1: no reference maps
        if maps is not None and ref_maps is None:
            # Case 1a: single map
            if isinstance(maps, xr.DataArray):
                title = self.set_map_title(
                    telecname="DMI",
                    statistic=statistic,
                    var=var_label,
                    model=maps.AQUA_model,
                    exp=maps.AQUA_exp,
                    season=getattr(maps, "AQUA_season", None),
                )

                fig, _ = plot_single_map(
                    data=maps,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    cbar_label=var_label,
                    title=title,
                    return_fig=True,
                    loglevel=self.loglevel,
                    **kwargs,
                )
                return fig

            # Case 1b: multiple maps
            elif isinstance(maps, list):
                titles = []
                for map in maps:
                    title = self.set_map_title(
                        telecname="DMI",
                        statistic=statistic,
                        var=var_label,
                        model=map.AQUA_model,
                        exp=map.AQUA_exp,
                        season=getattr(map, "AQUA_season", None),
                    )
                    titles.append(title)
                fig = plot_maps(
                    maps=maps,
                    vmin=vmin,
                    vmax=vmax,
                    titles=titles,
                    cmap=cmap,
                    cbar_label=var_label,
                    return_fig=True,
                    loglevel=self.loglevel,
                    **kwargs,
                )
                return fig

        # # Case 2: reference maps (maps and ref_maps are not None)
        if ref_maps is not None:
            # Case 2a: both maps and ref_maps are only one (we consider only both lists of one or both xarrays)
            if isinstance(maps, xr.DataArray) and isinstance(ref_maps, xr.DataArray):
                title = self.set_map_title(
                    telecname="DMI",
                    statistic=statistic,
                    var=var_label,
                    model=maps.AQUA_model,
                    exp=maps.AQUA_exp,
                    season=getattr(maps, "AQUA_season", None),
                    ref_model=ref_maps.AQUA_model,
                    ref_exp=ref_maps.AQUA_exp,
                )
                fig, _ = plot_single_map_diff(
                    data=maps,
                    data_ref=ref_maps,
                    vmin_contour=vmin if vmin is not None else None,
                    vmax_contour=vmax if vmax is not None else None,
                    vmin_fill=vmin_diff if vmin_diff is not None else None,
                    vmax_fill=vmax_diff if vmax_diff is not None else None,
                    sym=True if vmax_diff is None and vmin_diff is None else False,
                    sym_contour=True if vmax is None and vmin is None else False,
                    cmap=cmap,
                    cbar_label=var_label,
                    title=title,
                    return_fig=True,
                    loglevel=self.loglevel,
                    **kwargs,
                )
                return fig

            # Case 2b: maps are list and ref_maps is only one
            if isinstance(maps, list) and isinstance(ref_maps, xr.DataArray):
                titles = []
                for map in maps:
                    title = f"{map.AQUA_model} {map.AQUA_exp}"
                    titles.append(title)
                title = self.set_map_title(
                    telecname="DMI",
                    statistic=statistic,
                    var=var_label,
                    ref_model=ref_maps.AQUA_model,
                    ref_exp=ref_maps.AQUA_exp,
                    season=getattr(ref_maps, "AQUA_season", None),
                )

                # plot_maps_diff wants a list of reference maps of the same length as maps
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
                    cmap=cmap,
                    cbar_label=var_label,
                    titles=titles,
                    title=title,
                    return_fig=True,
                    loglevel=self.loglevel,
                    **kwargs,
                )
                return fig

            # Case 2c: maps is only one and ref_maps is list
            if isinstance(maps, xr.DataArray) and isinstance(ref_maps, list):
                titles = []
                for map in ref_maps:
                    title = f"Compared to {map.AQUA_model} {map.AQUA_exp}"
                    titles.append(title)
                title = self.set_map_title(
                    telecname="DMI",
                    statistic=statistic,
                    var=var_label,
                    model=maps.AQUA_model,
                    exp=maps.AQUA_exp,
                    season=getattr(maps, "AQUA_season", None),
                )

                # plot_maps_diff wants a list of reference maps of the same length as maps
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
                    cmap=cmap,
                    cbar_label=var_label,
                    titles=titles,
                    title=title,
                    return_fig=True,
                    loglevel=self.loglevel,
                    **kwargs,
                )
                return fig

            # Case 2d: maps and ref_maps are lists
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
