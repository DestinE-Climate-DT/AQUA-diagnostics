"""Module for plotting Mixed Layer Depth (MLD) maps."""

import math
from typing import Union

import cartopy.crs as ccrs
import xarray as xr

from aqua.core.logger import log_configure
from aqua.core.util import cbar_get_label, get_realizations, time_to_string
from aqua.diagnostics.base import SAVE_FORMAT, OutputSaver, TitleBuilder

from .mld_profiles import plot_maps

xr.set_options(keep_attrs=True)


class PlotMLD:
    """Class for plotting Mixed Layer Depth (MLD) maps."""

    def __init__(
        self,
        data: xr.Dataset,
        obs: xr.Dataset = None,
        diagnostic_name: str = "ocean_stratification",
        outputdir: str = ".",
        loglevel: str = "WARNING",
    ):
        """Class to plot Mixed Layer Depth (MLD) maps.

        Args:
            data (xr.Dataset): Dataset containing the MLD data to be plotted.
            obs (xr.Dataset, optional): Dataset containing observational MLD data for comparison. Default is None.
            clim_time (str, optional): Climatological time period for the data. Default is "January".
            diagnostic_name (str, optional): Name of the diagnostic. Default is "ocean_stratification".
            outputdir (str, optional): Directory to save the output plots. Default is the current directory.
            loglevel (str, optional): Logging level. Default is "WARNING".

        """
        self.data = data
        self.obs = obs

        self.loglevel = loglevel
        self.logger = log_configure(self.loglevel, "PlotStratification")

        self.diagnostic = diagnostic_name
        self.vars = list(self.data.data_vars)
        self.logger.debug("Variables in data: %s", self.vars)

        self.catalog = self.data[self.vars[0]].AQUA_catalog
        self.model = self.data[self.vars[0]].AQUA_model
        self.exp = self.data[self.vars[0]].AQUA_exp
        self.realizations = get_realizations(self.data[self.vars[0]])
        self.region = self.data.attrs.get("AQUA_region", "global")

        if self.obs:
            self.obs_catalog = self.obs[self.vars[0]].AQUA_catalog
            self.obs_model = self.obs[self.vars[0]].AQUA_model
            self.obs_exp = self.obs[self.vars[0]].AQUA_exp

        self.outputsaver = OutputSaver(
            diagnostic=self.diagnostic,
            catalog=self.catalog,
            model=self.model,
            exp=self.exp,
            outputdir=outputdir,
            realization=self.realizations,
            loglevel=self.loglevel,
        )

    def plot_mld(
        self,
        rebuild: bool = True,
        region: str = None,
        proj_name: str = "PlateCarree",
        extent: list = None,
        save_format: Union[str, list] = SAVE_FORMAT,
        dpi: int = 300,
    ):
        """Generate and save the MLD map plot.

        Args:
            rebuild (bool, optional): If True, rebuild existing output files. Default is True.
            region (str, optional): Region name to override the dataset's default. Default is None.
            proj_name (str, optional): Cartopy projection name. Default is "PlateCarree".
            extent (list, optional): Map extent as [lonmin, lonmax, latmin, latmax]. Default is None.
            save_format (str or list, optional): Format(s) to save the figure. Default is SAVE_FORMAT.
            dpi (int, optional): Resolution of the saved figure. Default is 300.

        """
        self.diagnostic_product = "mld"
        self.clim_time = self.data.attrs.get("AQUA_stratification_climatology", "Total")
        self.region = region if region else self.region
        self.extent = extent
        self.data_list = [self.data, self.obs] if self.obs else [self.data]
        self.set_central_lat_lon()
        self.set_proj(proj_name=proj_name)
        self.set_data_map_list()
        self.set_extent()
        self.set_data_map_list()
        self.set_suptitle()
        self.set_title()
        self.set_description()
        self.set_ytext()
        self.set_nrowcol()
        self.set_figsize()
        self.set_cbar_labels(var="mld")
        self.set_cbar_limits()
        fig = plot_maps(
            maps=self.data_map_list,
            nrows=self.nrows,
            ncols=self.ncols,
            proj=self.proj,
            extent=self.extent,
            title=self.suptitle,
            titles=self.title_list,
            cbar_number="single",
            cbar_label=self.cbar_label,
            figsize=self.figsize,
            cmap="jet",
            ytext=self.ytext,
            return_fig=True,
            vmax=self.vmax,
            vmin=self.vmin,
            nlevels=self.nlevels,
            sym=False,
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

    def set_figsize(self):
        """Set the figure size based on the number of rows and columns."""
        self.figsize = (3.5 * self.ncols, 3.5 * self.nrows)

    def set_nrowcol(self):
        """Set the number of rows and columns for the subplot grid."""
        if hasattr(self, "levels") and self.levels:
            self.nrows = len(self.levels)
        else:
            self.nrows = 1
        self.ncols = len(self.vars)
        if self.obs:
            self.ncols = self.ncols * 2

    def set_ytext(self):
        """Set the y-axis text labels for each subplot."""
        self.ytext = []
        if hasattr(self, "levels") and self.levels:
            for level in self.levels:
                for i in range(len(self.vars)):
                    if i == 0:
                        self.ytext.append(f"{level}m")
                    else:
                        self.ytext.append(None)

    def set_extent(self):
        """Set the map extent based on the data's coordinate range and region."""
        lonmin = self.data.lon.min().values
        lonmax = self.data.lon.max().values

        if self.region == "arctic":
            latmin = 40
            latmax = 90
        elif self.region == "antarctic":
            latmin = -90
            latmax = -40
        else:
            latmin = self.data.lat.min().values
            latmax = self.data.lat.max().values
        self.extent = [lonmin, lonmax, latmin, latmax]
        self.logger.debug(f"Map extent set to: {self.extent}")

    def set_central_lat_lon(self):
        """Set the central latitude and longitude for the map projection."""
        if self.region == "arctic":
            self.central_longitude = 0
            self.central_latitude = 90
        elif self.region == "antarctic":
            self.central_longitude = 0
            self.central_latitude = -90
        else:
            self.central_longitude = self.data.lon.mean().values
            self.central_latitude = self.data.lat.mean().values

    def set_proj(self, proj_name: str = "PlateCarree"):
        """Set the Cartopy map projection.

        Args:
            proj_name (str, optional): Projection name ('PlateCarree' or 'Orthographic'). Default is 'PlateCarree'.

        """
        if proj_name == "PlateCarree":
            self.proj = ccrs.PlateCarree(central_longitude=self.central_longitude)
        elif proj_name == "Orthographic":
            self.proj = ccrs.Orthographic(central_longitude=self.central_longitude, central_latitude=self.central_latitude)
        else:
            raise ValueError(f"Unknown projection name: {proj_name}")
        self.logger.debug(f"Projection set to: {proj_name}")

    def set_data_map_list(self):
        """Build the list of DataArrays to be plotted as maps."""
        self.data_map_list = []
        for data in self.data_list:
            if hasattr(self, "levels") and self.levels:
                data = data.interp(level=self.levels)
                for level in self.levels:
                    for var in self.vars:
                        if level == 0:
                            data_level_var = data[var].isel(level=-1)
                        else:
                            data_level_var = data[var].sel(level=level)

                        data_level_var.attrs["long_name"] = f"{data_level_var.attrs.get('long_name', var)} at {level}m"
                        self.data_map_list.append(data_level_var)
            else:
                for var in self.vars:
                    data_var = data[var]
                    self.data_map_list.append(data_var)

    def set_cbar_labels(self, var: str = None):
        """Set the colorbar label for the given variable.

        Args:
            var (str, optional): Variable name to derive the colorbar label from.

        """
        self.cbar_label = cbar_get_label(data=self.data[var], cbar_label=None, loglevel=self.loglevel)

    def set_convert_lon(self, data=None):
        """Convert longitude from 0-360 to -180 to 180 and sort accordingly."""
        data = data.assign_coords(lon=((data.lon + 180) % 360) - 180)
        data = data.sortby("lon")

        # lat_limits = data.attrs['AQUA_lat_limits']
        lon_limits = data.attrs["AQUA_lon_limits"]

        if lon_limits is not None:
            lon_min, lon_max = lon_limits
            lon_min = ((lon_min + 180) % 360) - 180
            lon_max = ((lon_max + 180) % 360) - 180
            ds_reg = self.data
            if lon_min < lon_max:
                ds_reg = ds_reg.sel(lon=slice(lon_min, lon_max))
            else:
                ds_reg = xr.concat(
                    [
                        ds_reg.sel(lon=slice(lon_min, 180)),
                        ds_reg.sel(lon=slice(-180, lon_max)),
                    ],
                    dim="lon",
                )
            data = ds_reg
        return data

    def _round_up(self, value):
        if value % 100 == 0:
            return value  # Already a multiple of 100
        elif value % 100 <= 50:
            return math.ceil(value / 50) * 50  # Round up to next 50
        else:
            return math.ceil(value / 100) * 100  # Round up to next 100

    def set_cbar_limits(self):
        """Set the colorbar limits and number of levels for MLD plots."""
        self.vmin = 0.0
        if self.obs:
            self.vmax = max(self.obs["mld"].max(), self.obs["mld"].max())
        else:
            self.vmax = self.data["mld"].max()
        self.vmax = self._round_up(self.vmax)
        if self.vmax < 200:
            nlevels = 10
        elif self.vmax > 1500:
            nlevels = 100
        else:
            nlevels = 50
        self.nlevels = nlevels
        self.logger.debug(f"Colorbar limits set to vmin: {self.vmin}, vmax: {self.vmax}, nlevels: {self.nlevels}")

    def set_suptitle(self, plot_type=None):
        """Set the title for the MLD plot."""
        self.suptitle = TitleBuilder(
            diagnostic="MLD",
            regions=self.region,
            model=self.model,
            exp=self.exp,
            timeseason=f"{self.clim_time} climatology",
        ).generate()
        self.logger.debug(f"Suptitle set to: {self.suptitle}")

    def set_title(self):
        """Set the title for each subplot panel."""
        self.title_list = []
        for j in range(len(self.data_map_list)):
            attrs = self.data_map_list[j].attrs
            for i, var in enumerate(self.vars):
                # if j == 0:
                # title = f"{var} ({self.data[var].attrs.get('units')})"
                title = f"{attrs.get('AQUA_model')} {attrs.get('AQUA_exp')}"
                self.title_list.append(title)
                # else:
                #     self.title_list.append(" ")
        self.logger.debug("Title list set to: %s", self.title_list)

    def set_description(self):
        """Build the figure description string including model and observation date ranges."""
        model_startdate = self.data.attrs.get("startdate", None)
        model_enddate = self.data.attrs.get("enddate", None)
        self.description = (
            f"{self.clim_time} climatology of mixed layer depth in the {self.region} region for {self.model} {self.exp}"
        )
        if model_startdate and model_enddate:
            self.description += (
                f" (from {time_to_string(model_startdate, format='%Y-%m')} to {time_to_string(model_enddate, format='%Y-%m')})"
            )
        if self.obs:
            obs_startdate = self.obs.attrs.get("startdate", None)
            obs_enddate = self.obs.attrs.get("enddate", None)
            self.description += f" with reference {self.obs_model} {self.obs_exp}"
            if obs_startdate and obs_enddate:
                self.description += (
                    f" (from {time_to_string(obs_startdate, format='%Y-%m')} to {time_to_string(obs_enddate, format='%Y-%m')})"
                )

        self.description += "."

    def save_plot(
        self,
        fig,
        diagnostic_product: str = None,
        extra_keys: dict = None,
        rebuild: bool = True,
        dpi: int = 300,
        format: str = SAVE_FORMAT,
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
        self.outputsaver.save_figure(
            fig,
            diagnostic_product=diagnostic_product,
            rebuild=rebuild,
            extra_keys=extra_keys,
            metadata=metadata,
            dpi=dpi,
            extension=format,
        )
