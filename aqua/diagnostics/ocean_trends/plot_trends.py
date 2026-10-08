"""Module for plotting ocean trend diagnostics."""

from typing import Union

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr

from aqua.core.logger import log_configure
from aqua.core.util import get_realizations, time_to_string, unit_to_latex
from aqua.diagnostics.base import SAVE_FORMAT, OutputSaver, TitleBuilder
from aqua.diagnostics.base.defaults import DEFAULT_OCEAN_VERT_COORD

from .multiple_maps import plot_maps
from .multivar_vertical_profiles import plot_multivars_vertical_profile

xr.set_options(keep_attrs=True)


class PlotTrends:
    """Class for plotting ocean trend diagnostics from xarray Datasets."""

    def __init__(
        self,
        data: xr.Dataset,
        diagnostic_name: str = "trends",
        vert_coord: str = DEFAULT_OCEAN_VERT_COORD,
        outputdir: str = ".",
        rebuild: bool = True,
        loglevel: str = "WARNING",
    ):
        """Class to plot trends from xarray Dataset.

        Args:
            data (xr.Dataset): Input xarray Dataset containing trend data.
            diagnostic_name (str, optional): Name of the diagnostic for filenames. Defaults to "trends".
            vert_coord (str, optional): Name of the vertical dimension coordinate. Defaults to DEFAULT_OCEAN_VERT_COORD.
            outputdir (str, optional): Directory to save output plots. Defaults to ".".
            rebuild (bool, optional): Whether to rebuild output files. Defaults to True.
            loglevel (str, optional): Logging level. Default is "WARNING".

        """
        self.loglevel = loglevel
        self.logger = log_configure(self.loglevel, "PlotTrends")

        if not isinstance(data, xr.Dataset) or not data.data_vars:
            raise ValueError("PlotTrends requires a nonempty xarray.Dataset.")
        self.data = data
        self.diagnostic_name = diagnostic_name
        self.vert_coord = vert_coord
        self.outputdir = outputdir
        self.rebuild = rebuild

        self.vars = list(self.data.data_vars)
        self.logger.debug("Variables in data: %s", self.vars)

        # Initialize metadata attributes
        self._get_info()

        self.outputsaver = OutputSaver(
            diagnostic=self.diagnostic_name,
            catalog=self.catalog,
            model=self.model,
            exp=self.exp,
            outputdir=outputdir,
            realization=self.realizations,
            loglevel=self.loglevel,
        )

    def plot_multilevel(
        self,
        levels: list = None,
        rebuild: bool = True,
        cbar_limits: dict = None,
        sym: bool = False,
        save_format: Union[str, list] = SAVE_FORMAT,
        dpi: int = 300,
    ):
        """Plot multi-level maps of trends.

        Args:
            levels (list, optional): Depths in metres. Zero selects the native level nearest the surface.
                Defaults to [10, 100, 500, 1000, 3000, 5000]. Empty levels are skipped only when all variables are NaN.
            rebuild (bool, optional): If True, rebuild existing output files. Defaults to True.
            cbar_limits (dict, optional): Per-variable colorbar limits as {var: {'vmin': v, 'vmax': v}}. Defaults to None.
            sym (bool, optional): If True, use symmetric colorbar limits. Defaults to False.
            save_format (str or list, optional): Format(s) to save the figure in. Defaults to SAVE_FORMAT.
            dpi (int, optional): Resolution of the saved figure. Defaults to 300.

        """
        self.diagnostic_product = "multilevel_trend"
        if levels is not None:
            if not len(levels):
                raise ValueError("At least one depth level is required.")
            self.levels = list(levels)
        else:
            self.levels = [10, 100, 500, 1000, 3000, 5000]
        self.logger.debug("Levels set to: %s", self.levels)
        self.cbar_limits = cbar_limits
        self.set_vmin_vmax()
        self.sym = sym
        self.set_central_longitude()
        self.set_data_list()
        self.set_suptitle(plot_type="Multi-level Trends")
        self.set_title()
        self.set_description(content="Multi-level Trends")
        self.set_ytext()
        self.set_cbar_labels()
        self.set_nrowcol()
        self.set_extent()
        fig = plot_maps(
            maps=self.data_list,
            nrows=self.nrows,
            ncols=self.ncols,
            proj=ccrs.PlateCarree(central_longitude=self.central_longitude),
            title=self.suptitle,
            col_vmin=self.vmin,
            col_vmax=self.vmax,
            sym=self.sym,
            titles=self.title_list,
            extent=self.extent,
            cbar_labels=self.cbar_labels,
            ytext=self.ytext,
            return_fig=True,
            loglevel=self.loglevel,
        )

        self.save_plot(
            fig,
            diagnostic_product=self.diagnostic_product,
            metadata={"description": self.description},
            rebuild=rebuild,
            format=save_format,
            dpi=dpi,
            extra_keys={"region": self.region},
        )
        plt.close(fig)

    def plot_zonal(self, rebuild: bool = True, save_format: Union[str, list] = SAVE_FORMAT, dpi: int = 300):
        """Plot zonal mean vertical profiles of trends.

        Args:
            rebuild (bool, optional): If True, rebuild existing output files. Defaults to True.
            save_format (str or list, optional): Format(s) to save the figure in. Defaults to SAVE_FORMAT.
            dpi (int, optional): Resolution of the saved figure. Defaults to 300.

        """
        if any(set(self.data[var].dims) != {self.vert_coord, "lat"} for var in self.vars):
            raise ValueError("plot_zonal() requires latitude-depth fields; prepare the longitude mean explicitly.")
        self.diagnostic_product = "zonal_mean"
        self.levels = None
        self.set_data_list()
        self.set_suptitle(plot_type="Trends of zonal mean")
        self.set_title()
        self.set_description(content="Zonal trends of " + ", ".join(self.data[v].attrs.get("long_name", v) for v in self.vars))
        self.set_ytext()
        self.set_cbar_labels()
        self.set_nrowcol()
        fig = plot_multivars_vertical_profile(
            maps=self.data_list,
            nrows=self.nrows,
            ncols=self.ncols,
            vert_coord=self.vert_coord,
            title=self.suptitle,
            titles=self.title_list,
            cbar_labels=self.cbar_labels,
            ytext=self.ytext,
            return_fig=True,
            sym=True,
            loglevel=self.loglevel,
        )

        self.save_plot(
            fig,
            diagnostic_product=self.diagnostic_product,
            metadata={"description": self.description},
            rebuild=rebuild,
            extra_keys={"region": self.region},
            format=save_format,
            dpi=dpi,
        )
        plt.close(fig)

    def set_vmin_vmax(self):
        """Set per-variable colorbar min/max from cbar_limits if provided."""
        limits = self.cbar_limits or {}
        self.vmin = [limits.get(var, {}).get("vmin") for var in self.vars]
        self.vmax = [limits.get(var, {}).get("vmax") for var in self.vars]

    def set_nrowcol(self):
        """Set the number of rows and columns for the subplot grid."""
        if hasattr(self, "levels") and self.levels:
            self.nrows = len(self.levels)
        else:
            self.nrows = 1
        self.ncols = len(self.vars)

    def set_extent(self):
        """Set the extent for the plot."""
        self.extent = [
            self.data_list[0].lon.min().values,
            self.data_list[0].lon.max().values,
            self.data_list[0].lat.min().values,
            self.data_list[0].lat.max().values,
        ]
        self.logger.debug("Extent set to: %s", self.extent)

    def set_ytext(self):
        """Set the y-axis text for the multi-level plots."""
        self.ytext = []
        if hasattr(self, "levels") and self.levels:
            for level in self.levels:
                for i in range(len(self.vars)):
                    if i == 0:
                        self.ytext.append(f"{level}m")
                    else:
                        self.ytext.append(None)

    def set_central_longitude(self):
        """Set the central longitude for the map projection from the data."""
        self.central_longitude = self.data.lon.mean().values
        self.logger.debug("Central longitude set to: %s", self.central_longitude)

    def set_data_list(self):
        """Prepare the list of data arrays to plot."""
        self.data_list = []
        if hasattr(self, "levels") and self.levels:
            if any(set(self.data[var].dims) != {self.vert_coord, "lat", "lon"} for var in self.vars):
                raise ValueError("Multilevel maps require depth-latitude-longitude fields; regrid through Reader first.")
            # Plotting is an output boundary. Compute a separate view once, leaving the
            # original Dataset and the caller's requested levels unchanged.
            interpolated = self.data.interp({self.vert_coord: self.levels}).compute()
            retained_levels = []
            for level in self.levels:
                if level == 0:
                    index = abs(self.data[self.vert_coord]).argmin().item()
                    level_data = self.data.isel({self.vert_coord: index}).compute()
                else:
                    level_data = interpolated.sel({self.vert_coord: level})
                if all(bool(level_data[var].isnull().all()) for var in self.vars):
                    self.logger.warning("All variables are NaN at %sm; skipping this level", level)
                    continue
                retained_levels.append(level)
                for var in self.vars:
                    data_level_var = level_data[var].copy(deep=False)
                    data_level_var.attrs["long_name"] = f"{data_level_var.attrs.get('long_name', var)} at {level}m"
                    self.data_list.append(data_level_var)
            self.levels = retained_levels
            if not self.data_list:
                raise ValueError("No valid trend data at the requested depth levels.")
        else:
            for var in self.vars:
                data_var = self.data[var]
                self.data_list.append(data_var)

    def set_suptitle(self, plot_type=None):
        """Set the title for the plot."""
        self.suptitle = TitleBuilder(diagnostic=plot_type, regions=self.region, model=self.model, exp=self.exp).generate()
        self.logger.debug("Suptitle set to: %s", self.suptitle)

    def set_title(self):
        """Set the title for each subplot panel."""
        self.title_list = []
        for i in range(len(self.data_list)):
            var = self.vars[i % len(self.vars)]
            self.title_list.append(self.data[var].attrs.get("long_name", var) if i < len(self.vars) else " ")
        self.logger.debug("Title list set to: %s", self.title_list)

    def set_cbar_labels(self):
        """Set the colorbar labels for each subplot from variable units."""
        self.cbar_labels = []
        for data_var in self.data_list:
            units = data_var.attrs.get("units", "")
            units_latex = unit_to_latex(units) if units else ""
            self.cbar_labels.append(f"{data_var.attrs.get('short_name', data_var.name)} ({units_latex})")
        self.logger.debug("Colorbar labels set to: %s", self.cbar_labels)

    def set_description(self, content=None):
        """Set the description metadata for the plot."""
        model_startdate = self.data.attrs.get("AQUA_startdate", None)
        model_enddate = self.data.attrs.get("AQUA_enddate", None)
        self.description = f"{content} in the {self.region} region of {self.model} {self.exp}."
        if model_startdate and model_enddate:
            self.description += (
                f" ({time_to_string(model_startdate, format='%Y-%m')} to {time_to_string(model_enddate, format='%Y-%m')})"
            )

    def save_plot(
        self,
        fig,
        diagnostic_product: str,
        extra_keys: dict = None,
        rebuild: bool = True,
        dpi: int = 300,
        format: Union[str, list] = SAVE_FORMAT,
        metadata: dict = None,
    ):
        """Save the plot to a file.

        Args:
            fig (matplotlib.figure.Figure): The figure to be saved.
            diagnostic_product (str): The name of the diagnostic product. Default is None.
            extra_keys (dict): Extra keys to be used for the filename (e.g. season). Default is None.
            rebuild (bool): If True, the output files will be rebuilt. Default is True.
            dpi (int): The dpi of the figure. Default is 300.
            format (str or list): Format(s) to save the figure. Default is SAVE_FORMAT.
            metadata (dict): The metadata to be used for the figure. Default is None.
                             They will be complemented with the metadata from the outputsaver.
                             We usually want to add here the description of the figure.

        """
        extra_keys = {**(extra_keys or {}), "region": self.region}
        if "AQUA_dim_mean" in self.data.attrs:
            extra_keys["dim_mean"] = self.data.attrs["AQUA_dim_mean"]

        self.outputsaver.save_figure(
            fig,
            diagnostic_product=diagnostic_product,
            rebuild=rebuild,
            extra_keys=extra_keys,
            metadata=metadata,
            extension=format,
            dpi=dpi,
        )

    def _get_info(self):
        """Extract model, catalog, exp, region from data attributes."""
        first_var = self.data[self.vars[0]]
        attrs = {**self.data.attrs, **first_var.attrs}
        self.catalog = attrs.get("AQUA_catalog")
        self.model = attrs.get("AQUA_model")
        self.exp = attrs.get("AQUA_exp")
        self.realizations = get_realizations(first_var if "AQUA_realization" in first_var.attrs else self.data)
        self.region = self.data.attrs.get("AQUA_region", "global")
