from aqua.core.graphics import plot_maps, plot_single_map, plot_vertical_profile
from aqua.core.logger import log_configure
from aqua.core.util import get_projection, get_realizations, time_to_string
from aqua.diagnostics.base import SAVE_FORMAT, OutputSaver, TitleBuilder

from .util import handle_pressure_level


class PlotClimatology:
    def __init__(
        self,
        diagnostic="climatology",
        save_format=SAVE_FORMAT,
        dpi=300,
        outputdir="./",
        cmap="RdBu_r",
        return_fig: bool = False,
        loglevel="WARNING",
    ):
        """
        Initialize the PlotClimatology class.

        Args:
            diagnostic (str): Name of the diagnostic.
            save_format (str or list): Format(s) to save the figures. Default is SAVE_FORMAT.
            dpi (int): Resolution of saved figures.
            outputdir (str): Output directory for saved plots.
            cmap (str): Colormap to use for the plots.
            return_fig (bool): Whether plotting methods should return the figure and axes.
            loglevel (str): Logging level.
        """
        self.diagnostic = diagnostic
        self.format_to_save = save_format
        self.dpi = dpi
        self.outputdir = outputdir
        self.cmap = cmap
        self.return_fig = return_fig
        self.loglevel = loglevel

        self.logger = log_configure(log_level=loglevel, log_name="Climatology")

    def _save_figure(self, fig, diagnostic_product, data, description, var, data_ref=None, plev=None, **kwargs):
        """
        Handles the saving of a figure using OutputSaver.

        Args:
            fig (matplotlib.Figure): The figure to save.
            data (xarray.Dataset): Dataset.
            data_ref (xarray.Dataset, optional): Reference dataset.
            diagnostic_product (str): Name of the diagnostic product.
            description (str): Description of the figure.
            var (str): Variable name.
            plev (float, optional): Pressure level.
        Keyword Args:
            **kwargs: Additional keyword arguments to be passed to the OutputSaver.
        """
        outputsaver = OutputSaver(
            diagnostic=self.diagnostic,
            catalog=data.AQUA_catalog,
            model=data.AQUA_model,
            exp=data.AQUA_exp,
            model_ref=data_ref.AQUA_model if data_ref else None,
            exp_ref=data_ref.AQUA_exp if data_ref else None,
            outputdir=self.outputdir,
            loglevel=self.loglevel,
            **kwargs,
        )

        metadata = {"Description": description}
        extra_keys = {}

        if var is not None:
            extra_keys.update({"var": var})
        if plev is not None:
            extra_keys.update({"plev": plev})

        outputsaver.save_figure(
            fig, diagnostic_product, extra_keys=extra_keys, metadata=metadata, extension=self.format_to_save, dpi=self.dpi
        )

    def plot_climatology(
        self,
        data,
        var,
        plev=None,
        proj="robinson",
        proj_params={},
        vmin=None,
        vmax=None,
        cbar_label=None,
    ):
        """
        Plots the climatology map for a given variable and time range.

        Args:
            data (xarray.Dataset): Climatology dataset to plot.
            var (str): Variable name.
            plev (float, optional): Pressure level to plot (if applicable).
            proj (string, optional): Desired projection for the map.
            proj_params (dict, optional): Additional arguments for the projection (e.g., {'central_longitude': 0}).
            vmin (float, optional): Minimum color scale value.
            vmax (float, optional): Maximum color scale value.
            cbar_label (str, optional): Label for the colorbar.

        Returns:
            tuple: Matplotlib figure and axis objects.
        """
        self.logger.info("Plotting climatology.")

        data = handle_pressure_level(data, var, plev, loglevel=self.loglevel)
        if data is None:
            return None

        realization = get_realizations(data)
        proj = get_projection(proj, **proj_params)

        extra_info = f"at {int(plev / 100)} hPa" if plev else None
        title = TitleBuilder(
            diagnostic="Climatology",
            variable=data[var].attrs.get("long_name", var),
            model=data.AQUA_model,
            exp=data.AQUA_exp,
            extra_info=extra_info,
        ).generate()

        fig, ax = plot_single_map(
            data[var],
            return_fig=True,
            title=title,
            title_size=16,
            vmin=vmin,
            vmax=vmax,
            proj=proj,
            loglevel=self.loglevel,
            cbar_label=cbar_label,
            cmap=self.cmap,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        description = (
            f"Spatial map of the climatology for {data[var].attrs.get('long_name', var).lower()}"
            f"{' at ' + str(int(plev / 100)) + ' hPa' if plev else ''} "
            f"from {time_to_string(data.AQUA_startdate, format='%Y-%m')} "
            f"to {time_to_string(data.AQUA_enddate, format='%Y-%m')} "
            f"for the {data.AQUA_model} model, experiment {data.AQUA_exp}."
        )

        if self.format_to_save:
            self._save_figure(
                fig=fig,
                diagnostic_product="annual_climatology",
                data=data,
                description=description,
                var=var,
                plev=plev,
                realization=realization,
            )

        if self.return_fig:
            return fig, ax
        return None

    def plot_seasonal_climatology(
        self, data, var, plev=None, proj="robinson", proj_params={}, vmin=None, vmax=None, cbar_label=None
    ):
        """
        Plots seasonal climatology for each season (DJF, MAM, JJA, SON).

        Args:
            data (xarray.Dataset): Climatology dataset to plot.
            var (str): Variable name.
            plev (float, optional): Pressure level.
            proj (str, optional): Desired projection for the map.
            proj_params (dict, optional): Additional arguments for the projection.
            vmin (float, optional): Minimum colorbar value.
            vmax (float, optional): Maximum colorbar value.
            cbar_label (str, optional): Label for the colorbar.

        Returns:
            matplotlib.figure.Figure: The resulting figure.
        """
        self.logger.info("Plotting seasonal climatology.")

        data = handle_pressure_level(data, var, plev, loglevel=self.loglevel)
        if data is None:
            return None

        realization = get_realizations(data)

        season_list = ["DJF", "MAM", "JJA", "SON"]

        extra_info = f"at {int(plev / 100)} hPa" if plev else None
        title = TitleBuilder(
            diagnostic="Seasonal Climatology",
            variable=data[var].attrs.get("long_name", var),
            model=data.AQUA_model,
            exp=data.AQUA_exp,
            extra_info=extra_info,
        ).generate()

        plot_kwargs = {
            "maps": [data[var].sel(season=season) for season in season_list],
            "proj": get_projection(proj, **proj_params),
            "return_fig": True,
            "title": title,
            "title_size": 16,
            "titles": season_list,
            "titles_size": 14,
            "figsize": (10, 8),
            "cbar_label": cbar_label,
            "cmap": self.cmap,
            "loglevel": self.loglevel,
        }

        if vmin is not None:
            plot_kwargs["vmin"] = vmin
        if vmax is not None:
            plot_kwargs["vmax"] = vmax

        fig = plot_maps(**plot_kwargs)

        description = (
            f"Seasonal climatology of {data[var].attrs.get('long_name', var).lower()}"
            f"{' at ' + str(int(plev / 100)) + ' hPa' if plev else ''} "
            f"(from {time_to_string(data.AQUA_startdate, format='%Y-%m')} "
            f"to {time_to_string(data.AQUA_enddate, format='%Y-%m')}) "
            f"for the {data.AQUA_model} model, experiment {data.AQUA_exp}."
        )

        if self.format_to_save:
            self._save_figure(
                fig=fig,
                diagnostic_product="seasonal_climatology",
                data=data,
                description=description,
                var=var,
                plev=plev,
                realization=realization,
            )

        if self.return_fig:
            return fig
        return None

    def plot_vertical_climatology(
        self,
        data,
        var,
        plev_min=None,
        plev_max=None,
        vmin=None,
        vmax=None,
        vmin_contour=None,
        vmax_contour=None,
        nlevels=18,
    ):
        """
        Calculates and plots the vertical profile of climatology.

        Args:
            data (xarray.Dataset): Dataset to analyze.
            var (str): Variable name to analyze.
            plev_min (float, optional): Minimum pressure level.
            plev_max (float, optional): Maximum pressure level.
            vmin (float, optional): Minimum colorbar value.
            vmax (float, optional): Maximum colorbar value.
            vmin_contour (float, optional): Minimum contour value.
            vmax_contour (float, optional): Maximum contour value.
            nlevels (int, optional): Number of contour levels for the plot.
        """
        self.logger.info("Plotting vertical climatology for variable: %s", var)

        realization = get_realizations(data)

        title = TitleBuilder(
            diagnostic="Vertical Climatology",
            variable=data[var].attrs.get("long_name", var),
            model=data.AQUA_model,
            exp=data.AQUA_exp,
        ).generate()

        description = (
            f"Vertical cross-section of {data[var].attrs.get('long_name', var).lower()} for "
            f"{data.AQUA_model} {data.AQUA_exp} (from {time_to_string(data.AQUA_startdate, format='%Y-%m')} "
            f"to {time_to_string(data.AQUA_enddate, format='%Y-%m')})."
        )

        fig, ax = plot_vertical_profile(
            data=data[var].mean(dim="lon"),
            var=var,
            lev_min=plev_min,
            lev_max=plev_max,
            vmin=vmin,
            vmax=vmax,
            logscale=True,
            cmap=self.cmap,
            nlevels=nlevels,
            title=title,
            title_size=16,
            return_fig=True,
            loglevel=self.loglevel,
        )

        if self.format_to_save:
            self._save_figure(
                fig=fig,
                diagnostic_product="vertical_climatology",
                data=data,
                description=description,
                var=var,
                realization=realization,
            )

        if self.return_fig:
            return fig, ax

        self.logger.info("Vertical climatology plot completed successfully.")
        return None
