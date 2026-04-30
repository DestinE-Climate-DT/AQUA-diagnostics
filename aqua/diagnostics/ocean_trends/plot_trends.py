from typing import Union

import cartopy.crs as ccrs
import xarray as xr

from aqua.core.logger import log_configure
from aqua.core.util import get_realizations, unit_to_latex, time_to_string
from aqua.diagnostics.base import SAVE_FORMAT, OutputSaver, TitleBuilder
from aqua.diagnostics.base.defaults import DEFAULT_OCEAN_VERT_COORD

from .multiple_maps import plot_maps
from .multivar_vertical_profiles import plot_multivars_vertical_profile

xr.set_options(keep_attrs=True)


class PlotTrends:
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

    def plot_multilevel(self, 
                        levels: list = None, 
                        rebuild: bool = True, 
                        cbar_limits: dict = None,
                        sym: bool = False,
                        save_format: Union[str, list] = SAVE_FORMAT, 
                        dpi: int = 300):
        """Plot multi-level maps of trends.

        Args:
            levels (list, optional): List of depth levels to plot. Defaults to None.
            save_format (str or list, optional): Format(s) to save the figure in (e.g. 'png', 'pdf', 'svg').
        """
        self.diagnostic_product = "multilevel_trend"
        if levels:
            self.levels = levels
        else:
            self.levels = [10, 100, 500, 1000, 3000, 5000]
        self.logger.debug(f"Levels set to: {self.levels}")
        self.cbar_limits = cbar_limits
        self.vmin = None
        self.vmax = None
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

    def plot_zonal(self, rebuild: bool = True, save_format: Union[str, list] = SAVE_FORMAT, dpi: int = 300):
        """
        Plot zonal mean vertical profiles of trends.

        Args:
            save_format (str or list, optional): Format(s) to save the figure in (e.g. 'png', 'pdf', 'svg').
            dpi (int, optional): Dots per inch for the output figure. Defaults to 300.
        """
        self.diagnostic_product = "zonal_mean"
        self.set_data_list()
        self.set_suptitle(plot_type='Trends of zonal mean')
        self.set_title()
        self.set_description(content="Trends of zonal mean of temperature (left) and salinity (right)")
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
    def set_vmin_vmax(self):
        self.vmin = []
        self.vmax = []
        if self.cbar_limits:
            self.vmin = [self.cbar_limits[var]['vmin'] for var in self.vars]
            self.vmax = [self.cbar_limits[var]['vmax'] for var in self.vars]

    def set_nrowcol(self):
        if hasattr(self, "levels") and self.levels:
            self.nrows = len(self.levels)
        else:
            self.nrows = 1
        self.ncols = len(self.vars)

    def set_extent(self):
        """Set the extent for the plot."""
        self.extent = [self.data_list[0].lon.min().values, self.data_list[0].lon.max().values,
                       self.data_list[0].lat.min().values, self.data_list[0].lat.max().values]
        self.logger.debug(f"Extent set to: {self.extent}")

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
        self.central_longitude = self.data.lon.mean().values
        self.logger.debug(f"Central longitude set to: {self.central_longitude}")

    def set_data_list(self):
        """Prepare the list of data arrays to plot."""
        self.data_list = []
        if hasattr(self, "levels") and self.levels:
            self.data = self.data.interp({self.vert_coord: self.levels})
            for level in self.levels:
                for var in self.vars:
                    if level == 0:
                        data_level_var = self.data[var].isel({self.vert_coord: -1})
                    else:
                        data_level_var = self.data[var].sel({self.vert_coord: level})

                    if data_level_var.isnull().all():
                        self.logger.warning(f"All values are NaN for {var} at {level}m")
                        self.levels.pop(self.levels.index(level))
                        break

                    data_level_var.attrs["long_name"] = f"{data_level_var.attrs.get('long_name', var)} at {level}m"
                    self.data_list.append(data_level_var)
        else:
            for var in self.vars:
                data_var = self.data[var]
                self.data_list.append(data_var)

    def set_suptitle(self, plot_type=None):
        """Set the title for the plot."""
        self.suptitle = TitleBuilder(
            diagnostic=plot_type, regions=self.region, model=self.model, exp=self.exp
        ).generate()
        self.logger.debug(f"Suptitle set to: {self.suptitle}")

    def set_title(self):
        """
        Set the title for the plot.
        This method can be extended to set specific titles based on the data.
        """
        self.title_list = []
        for j in range(len(self.data_list)):
            for var in self.vars:
                if j == 0:
                    units = self.data[var].attrs.get("units", "")
                    units_latex = unit_to_latex(units) if units else ""
                    title = f"{self.data[var].attrs.get('long_name', var)} ({units_latex})"
                    self.title_list.append(title)
                else:
                    self.title_list.append(" ")
        self.logger.debug("Title list set to: %s", self.title_list)

    def set_cbar_labels(self):
        """
        Set the colorbar labels for the plot.
        This method can be extended to set specific colorbar labels based on the data.
        """
        self.cbar_labels = []
        for _ in range(len(self.data_list)):
            for var in self.vars:
                units = self.data[var].attrs.get("units", "")
                units_latex = unit_to_latex(units) if units else ""
                cbar_label = f"{self.data[var].attrs.get('short_name', var)} ({units_latex})"
                self.cbar_labels.append(cbar_label)
        self.logger.debug("Colorbar labels set to: %s", self.cbar_labels)

    def set_description(self, content=None):
        """
        Set the description metadata for the plot.
        """

        self.description = f"{content} in the {self.region} region of {self.model} {self.exp}."

    def save_plot(
        self,
        fig,
        diagnostic_product: str,
        extra_keys: dict = {},
        rebuild: bool = True,
        dpi: int = 300,
        format: Union[str, list] = SAVE_FORMAT,
        metadata: dict = None,
    ):
        """
        Save the plot to a file.

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
        extra_keys.update({"region": self.region})

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
        self.catalog = self.data[self.vars[0]].AQUA_catalog
        self.model = self.data[self.vars[0]].AQUA_model
        self.exp = self.data[self.vars[0]].AQUA_exp
        self.realizations = get_realizations(self.data[self.vars[0]])
        self.region = self.data.attrs.get("AQUA_region", "global")
