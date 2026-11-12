"""Plot maps of the sea ice fraction or thickness climatology computed by SeaIce against a reference"""

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import xarray as xr

from aqua.core.graphics import ConfigStyle, plot_single_map, plot_single_map_diff
from aqua.core.logger import log_configure
from aqua.core.util import (
    apply_circular_window,
    evaluate_colorbar_limits,
    generate_colorbar_ticks,
    get_projection,
    get_realizations,
    int_month_name,
    set_map_title,
    time_to_string,
    to_list,
    unit_to_latex,
)
from aqua.diagnostics.base import SAVE_FORMAT, OutputSaver, TitleBuilder

from .util import _check_list_regions_type

xr.set_options(keep_attrs=True)

# colour limits of the difference maps
BIAS_LIMITS = {"fraction": (-1, 1), "thickness": (-5, 5)}
# [west, east, south, north] of the circular window of the azimuthal equidistant view, for each pole
POLAR_EXTENT = {"north": [-180, 180, 50, 90], "south": [-180, 180, -50, -90]}


class Plot2DSeaIce:
    """
    Plot maps of the sea ice fraction or thickness climatology of a model, a reference and their difference.

    Args:
        ref (xr.Dataset | xr.DataArray | list): Reference climatologies, as computed by ``SeaIce``.
        models (xr.Dataset | xr.DataArray | list): Model climatologies, as computed by ``SeaIce``.
        regions_to_plot (list, optional): Region long names to plot, matching the 'AQUA_region' attribute
            of the data. If None, all the regions of the data are plotted.
        diagnostic_name (str, optional): The name used for the output files. Default is 'seaice'.
        outputdir (str, optional): Output directory for saving plots. Default is './'.
        rebuild (bool, optional): Whether to overwrite existing figures. Default is True.
        dpi (int, optional): Resolution of saved figures. Default is 300.
        loglevel (str, optional): Logging level. Default is 'WARNING'.
    """

    def __init__(
        self,
        ref=None,
        models=None,
        regions_to_plot: list = None,
        diagnostic_name: str = "seaice",
        outputdir="./",
        rebuild=True,
        dpi=300,
        loglevel="WARNING",
    ):
        self.loglevel = loglevel
        self.logger = log_configure(log_level=self.loglevel, log_name="Plot2DSeaIce")

        self.realizations = get_realizations(models)
        self.ref = self._handle_data(ref)
        self.models = self._handle_data(models)

        if regions_to_plot is None:
            # regions in order of appearance in the data
            regions_to_plot = list(dict.fromkeys(da.attrs["AQUA_region"] for da in self.models + self.ref))
        self.regions_to_plot = _check_list_regions_type(regions_to_plot, logger=self.logger)

        self.diagnostic_name = diagnostic_name
        self.outputdir = outputdir
        self.rebuild = rebuild
        self.dpi = dpi

    def plot_2d_seaice(
        self,
        method="fraction",
        months=None,
        projection="orthographic",
        plot_ref_contour=False,
        save_format=SAVE_FORMAT,
        show=False,
        **kwargs,
    ):
        """
        Plot, for each region and model, the reference, the model and their difference for the requested months.

        Args:
            method (str): 'fraction' or 'thickness'. Default is 'fraction'.
            months (list, optional): Months to plot, one row each. Default is [3, 9].
            projection (str): Projection name, centred on the pole of the region. With 'azimuthal_equidistant'
                the maps are cut to a circle around the pole. Default is 'orthographic'.
            plot_ref_contour (bool): Whether to draw the reference contour on the model map. Default is False.
            save_format (str or list, optional): Format(s) to save the figure. Default is SAVE_FORMAT.
            show (bool): If True, display the figure (e.g. in Jupyter notebooks). Default is False.
            **kwargs: Additional keyword arguments for the maps, among which:
                bias_vmin_vmax (dict): 'vmin' and 'vmax' of the difference map.
                cbar_ticks_rounding (int): Rounding for colorbar ticks.
                add_land (bool): Whether to add land to the plot.
                gridlines (bool): Whether to add gridlines to the plot.
                line_levels (list): Levels of the reference contour. Default is [0.2].

        Raises:
            ValueError: If a month or the method is not valid, or if models or reference data are missing.
            TypeError: If the removed arguments 'plot_type' or 'projkw' are passed.
        """
        removed = sorted({"plot_type", "projkw"} & set(kwargs))
        if removed:
            raise TypeError(f"{removed} are no longer arguments: only difference maps are drawn, see 'projection'")
        self.months = [3, 9] if months is None else months
        if not all(1 <= m <= 12 for m in self.months):
            raise ValueError("Invalid month value. Months must be between 1 and 12.")
        if method not in BIAS_LIMITS:
            raise ValueError(f"Unsupported method '{method}'. Supported methods are {list(BIAS_LIMITS)}.")
        self.method = method
        if not self.models or not self.ref:
            raise ValueError("Missing models or reference data")
            
        # before the first figure exists: the maps set the style only when they are drawn,
        # after titles and frames of the first row have taken the matplotlib defaults
        ConfigStyle(style=kwargs.get("style"), loglevel=self.loglevel)

        for region in self.regions_to_plot:
            ref = [da for da in self.ref if da.attrs.get("AQUA_region") == region]
            models = [da for da in self.models if da.attrs.get("AQUA_region") == region]
            if not ref or not models:
                self.logger.error("Both model and reference data are needed for the maps. Skipping %s", region)
                continue
            self.logger.info("Plotting region: %s", region)

            for model in models:
                fig, description = self._plot_bias_map(ref[0], model, projection, plot_ref_contour, **kwargs)
                self._save_plots(
                    fig=fig,
                    data=model,
                    data_ref=ref[0],
                    diagnostic_product="bias",
                    description=description,
                    extra_keys={"method": self.method, "region": region},
                    save_format=save_format,
                )
                if show:
                    plt.show()
                plt.close(fig)

    def _plot_bias_map(self, ref, model, projection, plot_ref_contour, **kwargs):
        """Draw one row of reference, model and difference maps for each month; return the figure and its caption."""
        ticks_rounding = kwargs.get("cbar_ticks_rounding", 1)
        add_land = kwargs.pop("add_land", True)
        gridlines = kwargs.pop("gridlines", True)
        bias_limits = kwargs.get("bias_vmin_vmax", {})
        vmin = bias_limits.get("vmin", BIAS_LIMITS[self.method][0])
        vmax = bias_limits.get("vmax", BIAS_LIMITS[self.method][1])
        cbar_kwargs = {"orientation": "vertical", "ticks_rounding": ticks_rounding, "fraction": 0.046, "pad": 0.04}

        # centred on the last latitude of the reference towards its pole: the model, merged with
        # the regions of the other hemisphere, covers both
        central_latitude = max(ref["lat"].values, key=abs)
        if projection == "azimuthal_equidistant":
            # the circular window needs the exact pole: around 89.5 degrees the Arctic extent collapses to a line
            central_latitude = 90.0 if central_latitude > 0 else -90.0
        proj = get_projection(projection, central_longitude=0.0, central_latitude=central_latitude)
        fig = plt.figure(figsize=(3 * 5.1, len(self.months) * 4.8))
        subfigs = fig.subfigures(nrows=len(self.months), ncols=1, squeeze=False).flatten()

        for month, subfig in zip(self.months, subfigs):
            monref = self._mask_ice_at_mid_lats(ref.sel(month=month))
            monmod = self._mask_ice_at_mid_lats(model.sel(month=month))
            axs = subfig.subplots(1, 3, subplot_kw={"projection": proj})

            title = TitleBuilder(
                diagnostic=f"Sea ice {self.method}",
                regions=ref.attrs.get("AQUA_region"),
                timeseason=f"Month: {int_month_name(month)}",
            ).generate()
            subfig.suptitle(title, fontsize=14, y=1.02)

            for ax, data in zip(axs[:2], (monref, monmod)):
                setup = self._get_cmap(data)
                plot_single_map(
                    data,
                    proj=proj,
                    fig=fig,
                    ax=ax,
                    cmap=setup["colormap"],
                    norm=setup["norm"],
                    contour=False,
                    cbar=False,
                    add_land=add_land,
                    gridlines=gridlines,
                    loglevel=self.loglevel,
                    **kwargs,
                )
                self._add_colorbar(fig, data, ax=ax, norm=setup["norm"], boundaries=setup["boundaries"], **cbar_kwargs)

                if plot_ref_contour and ax is axs[1]:
                    # before the title: xarray gives the axis a title of its own
                    self._plot_reference_contour(ax, ref.sel(month=month), **kwargs)
                ax.set_title(f"{set_map_title(data, skip_varname=True)}")

            plot_single_map_diff(
                monmod,
                monref,
                proj=proj,
                fig=fig,
                ax=axs[2],
                add_contour=False,
                add_land=add_land,
                nlevels=26,
                vmin_fill=vmin,
                vmax_fill=vmax,
                sym=False,  # set False to later override with symmetric min-max values
                cbar=False,
                loglevel=self.loglevel,
                gridlines=gridlines,
                **kwargs,
            )
            self._add_colorbar(fig, monref, ax=axs[2], vmin=vmin, vmax=vmax, sym=True, **cbar_kwargs)
            axs[2].set_title(
                f"{set_map_title(monmod, skip_varname=True, put_model_name=True, put_exp_name=False)} - "
                f"{set_map_title(monref, skip_varname=True, put_model_name=True, put_exp_name=False)}"
            )

            if projection == "azimuthal_equidistant":
                for ax in axs:
                    apply_circular_window(ax, extent=POLAR_EXTENT["north" if central_latitude > 0 else "south"])

        description = (
            f"Climatology and differences against the reference dataset for sea ice {model.attrs.get('AQUA_method', '')} "
            f"in the {model.attrs.get('AQUA_region', 'geographic')} region. "
            f"Model data is {model.attrs.get('AQUA_model')}, experiment {model.attrs.get('AQUA_exp')} "
            f"(from {time_to_string(model.attrs.get('AQUA_startdate', ''), format='%Y-%m')} "
            f"to {time_to_string(model.attrs.get('AQUA_enddate', ''), format='%Y-%m')}). "
            f"Reference dataset is {ref.attrs.get('AQUA_model')}, experiment {ref.attrs.get('AQUA_exp')} "
            f"(from {time_to_string(ref.attrs.get('AQUA_startdate', ''), format='%Y-%m')} "
            f"to {time_to_string(ref.attrs.get('AQUA_enddate', ''), format='%Y-%m')}). "
        )
        if self.method == "fraction":
            description += "The red contour line represents where the sea ice fraction is equal to 0.2."
        return fig, description

    def _get_colorbar_ticks(self, data, vmin=None, vmax=None, norm=None, boundaries=None, sym=False, ticks_rounding=1):
        """Ticks of the colorbar: every second boundary of a discrete norm, otherwise 10 levels over the data range."""
        if norm is not None:
            return boundaries[::2] + [boundaries[-1]]
        if vmin is None or vmax is None:
            vmin, vmax = evaluate_colorbar_limits(maps=[data], sym=sym)
        return generate_colorbar_ticks(vmin=vmin, vmax=vmax, sym=sym, nlevels=10, ticks_rounding=ticks_rounding)

    def _add_colorbar(
        self,
        fig,
        data,
        ax,
        vmin=None,
        vmax=None,
        norm=None,
        boundaries=None,
        sym=False,
        orientation="horizontal",
        ticks_rounding=1,
        **cb_kwargs,
    ):
        """
        Add to the figure the colorbar of the map drawn on an axis.

        Args:
            fig (matplotlib.Figure): The figure to which the colorbar will be added.
            data (xarray.DataArray): The data of the map, used for the ticks range and the label.
            ax (matplotlib.axes.Axes): The axis of the map.
            vmin (float, optional): Minimum value for the colorbar. Defaults to None.
            vmax (float, optional): Maximum value for the colorbar. Defaults to None.
            norm (matplotlib.colors.Normalize, optional): Normalization of the map. Defaults to None.
            boundaries (list, optional): Boundaries of a discrete normalization. Defaults to None.
            sym (bool, optional): If True, use symmetric limits for the colorbar. Defaults to False.
            orientation (str, optional): 'horizontal' or 'vertical'. Defaults to 'horizontal'.
            ticks_rounding (int, optional): Rounding for colorbar ticks. Defaults to 1.
            **cb_kwargs: Additional keyword arguments for the colorbar.

        Returns:
            matplotlib.colorbar.Colorbar: The created colorbar object.
        """
        cb = fig.colorbar(ax.collections[0], ax=ax, orientation=orientation, **cb_kwargs)
        cb.set_ticks(
            self._get_colorbar_ticks(
                data, vmin=vmin, vmax=vmax, norm=norm, boundaries=boundaries, sym=sym, ticks_rounding=ticks_rounding
            )
        )
        cb.ax.ticklabel_format(style="sci", axis="x", scilimits=(-3, 3))

        units = data.attrs.get("units", "")
        if units:
            units_latex = unit_to_latex(units)
            if not (units_latex.startswith("[") and units_latex.endswith("]")):
                units_latex = f"[{units_latex}]"
            units = " " + units_latex
        cb.set_label(f"Sea-ice {data.attrs.get('AQUA_method', '')}{units}", fontsize=11)
        return cb

    def _get_cmap(self, datarr):
        """
        Get the personalised colormap for sea ice variable.
        Args:
            datarr (xarray.DataArray): DataArray containing sea ice data with attributes.
        Returns:
            dict: A dictionary containing the colormap, norm, and boundaries.
        """
        self.logger.debug("Using method '%s' for colormap generation", self.method)

        if self.method == "fraction":
            # Define a custom blue-to-white gradient colormap
            source_colors = [
                [0.15, 0.35, 0.55],
                [0.4, 0.7, 0.85],
                [0.5, 0.75, 0.9],
                [0.6, 0.8, 0.95],
                [0.7, 0.85, 1.0],
                [0.8, 0.9, 1.0],
                [0.9, 0.95, 1.0],
                [1.0, 1.0, 1.0],
            ]
            colormap = mcolors.LinearSegmentedColormap.from_list("custom_fraction_colormap", source_colors, N=15)
            norm = None
            boundaries = None
        else:
            # Define custom boundaries for a discrete normalization
            boundaries = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8.0, 10, 15, 20, 30]
            colormap = plt.get_cmap("turbo")
            norm = mcolors.BoundaryNorm(boundaries, ncolors=colormap.N, clip=True)

        return {"colormap": colormap, "norm": norm, "boundaries": boundaries}

    def _plot_reference_contour(self, ax, ref_data, **kwargs):
        """
        Add the contour of the reference data to an axis.

        Args:
            ax (matplotlib.axes._subplots.AxesSubplot): Axis to add the contour to.
            ref_data (xarray.DataArray): Reference data of the month plotted on the axis.
            **kwargs: Additional keyword arguments for customization, such as:
                line_levels (list): List of contour levels to draw. Default is [0.2].
        """
        ref_data.plot.contour(
            ax=ax,
            transform=ccrs.PlateCarree(),
            levels=kwargs.get("line_levels", [0.2]),
            colors="red",
            linewidths=1,
            linestyles="-",
            add_colorbar=False,
        )

    def _mask_ice_at_mid_lats(self, datarr):
        """
        Further clean the data array.
        For 'thickness' method, remove values below 0.01 and mask lats inside the [-50, 50] range.
        For 'fraction' method, for lats values inside the range [-45, 40] overwrite NaNs with 0.

        Args:
            datarr (xarray.DataArray): DataArray containing sea ice data with attributes.

        Returns:
            xarray.DataArray: Cleaned DataArray with masked values.
        """
        lat = datarr["lat"].broadcast_like(datarr)
        if self.method == "thickness":
            mask = ((lat > -50) & (lat < 50)) | (datarr < 0.01)
            return datarr.where(~mask)
        elif self.method == "fraction":
            mask = (lat >= -45) & (lat <= 40)
            return datarr.where(~mask, 0)  # overwrite NaNs

    def _handle_data(self, datain) -> list:
        """
        Flatten the input into a list of xarray.DataArray objects.

        Args:
            datain (xarray.DataArray, xarray.Dataset, list, tuple): A Dataset contributes all its data variables;
                the None entries of a list (a missing dataset) are skipped.

        Returns:
            list: A flat list of xarray.DataArray objects, empty if there is no data.

        Raises:
            TypeError: If an entry is neither a Dataset nor a DataArray.
        """
        data_arrays = []
        for data in to_list(datain):
            if isinstance(data, xr.Dataset):
                data_arrays.extend(data.data_vars.values())
            elif isinstance(data, xr.DataArray):
                data_arrays.append(data)
            elif data is not None:
                raise TypeError(f"Unsupported type in 'datain' list: {type(data)}")
        return data_arrays

    def _save_plots(self, fig, data, diagnostic_product, description, data_ref=None, extra_keys=None, save_format=None):
        """
        Handles the saving of a figure using OutputSaver.

        Args:
            fig (matplotlib.Figure): The figure to save.
            data (xarray.DataArray): Model data, with its provenance in the attributes.
            diagnostic_product (str): Name of the diagnostic product.
            description (str): Description of the figure.
            data_ref (xarray.DataArray, optional): Reference data.
            extra_keys (dict, optional): Extra keys for filename.
            save_format (str or list, optional): Format(s) to save the figure. If empty, nothing is saved.
        """
        if not save_format:
            return

        outputsaver = OutputSaver(
            diagnostic=self.diagnostic_name,
            catalog=data.attrs.get("AQUA_catalog", ""),
            model=data.attrs.get("AQUA_model", ""),
            exp=data.attrs.get("AQUA_exp", ""),
            model_ref=data_ref.attrs.get("AQUA_model", "") if data_ref is not None else None,
            exp_ref=data_ref.attrs.get("AQUA_exp", "") if data_ref is not None else None,
            outputdir=self.outputdir,
            loglevel=self.loglevel,
            realization=self.realizations,
        )
        outputsaver.save_figure(
            fig,
            diagnostic_product,
            extra_keys=extra_keys,
            metadata={"Description": description},
            extension=save_format,
            rebuild=self.rebuild,
            dpi=self.dpi,
        )
