"""Module for plotting ocean stratification vertical profiles."""

import math
from typing import Union

import xarray as xr

from aqua.core.logger import log_configure
from aqua.core.util import cbar_get_label, get_realizations, time_to_string, unit_to_latex
from aqua.diagnostics.base import SAVE_FORMAT, OutputSaver, TitleBuilder
from aqua.diagnostics.base.defaults import DEFAULT_OCEAN_VERT_COORD

from .multiple_vertical_line import plot_multi_vertical_lines

xr.set_options(keep_attrs=True)


class PlotStratification:
    """Class for plotting ocean stratification vertical profiles."""

    def __init__(
        self,
        data: xr.Dataset,
        obs: xr.Dataset = None,
        diagnostic_name: str = "ocean_stratification",
        vert_coord: str = DEFAULT_OCEAN_VERT_COORD,
        outputdir: str = ".",
        loglevel: str = "WARNING",
    ):
        """Initialize PlotStratification with model and observational datasets.

        Args:
            data (xr.Dataset): Dataset containing stratification variables to plot.
            obs (xr.Dataset, optional): Observational dataset for comparison. Default is None.
            diagnostic_name (str, optional): Name of the diagnostic. Default is "ocean_stratification".
            vert_coord (str, optional): Vertical coordinate name. Default is DEFAULT_OCEAN_VERT_COORD.
            outputdir (str, optional): Directory to save output plots. Default is ".".
            loglevel (str, optional): Logging level. Default is "WARNING".

        """
        self.data = data
        self.obs = obs

        self.loglevel = loglevel
        self.logger = log_configure(self.loglevel, "PlotStratification")

        self.diagnostic = diagnostic_name
        self.vert_coord = vert_coord
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

    def plot_stratification(
        self,
        rebuild: bool = True,
        save_format: Union[str, list] = SAVE_FORMAT,
        dpi: int = 300,
    ):
        """Generate and save the stratification vertical profile plot.

        Args:
            rebuild (bool, optional): If True, rebuild existing output files. Default is True.
            save_format (str or list, optional): Format(s) to save the figure. Default is SAVE_FORMAT.
            dpi (int, optional): Resolution of the saved figure. Default is 300.

        """
        self.diagnostic_product = "stratification"
        self.clim_time = self.data.attrs.get("AQUA_stratification_climatology", "Total")
        # self.data_list = [self.data, self.obs] if self.obs else [self.data]
        self.set_data_list()
        self.set_suptitle()
        self.set_title()
        self.set_description()
        self.set_xtext()
        self.set_ytext()
        self.set_nrowcol()
        # self.set_cbar_labels(var= 'rho')
        self.set_label_line_plot()
        fig = plot_multi_vertical_lines(
            data_list=self.data_list,
            ref_data_list=self.ref_data_list if self.obs else None,
            nrows=self.nrows,
            ncols=self.ncols,
            variables=self.vars,
            vert_coord=self.vert_coord,
            data_label=self.data_label,
            obs_label=self.obs_label if self.obs else None,
            title=self.suptitle,
            titles=self.title_list,
            figsize=(4 * self.ncols, 5 * self.nrows),
            xtext=self.xtext,
            ytext=self.ytext,
            return_fig=True,
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

    def set_nrowcol(self):
        """Set subplot grid layout: ``nrows=1``, ``ncols=len(self.vars)``."""
        self.nrows = 1
        self.ncols = len(self.vars)

    def set_xtext(self):
        """Build x-axis labels from variable name and units (one per column)."""
        self.xtext = []
        for var in self.vars:
            units = self.data[var].attrs.get("units", "")
            units_latex = unit_to_latex(units) if units else ""
            self.xtext.append(f"{var} ({units_latex})")
        self.logger.debug("X-axis text labels set to: %s", self.xtext)

    def set_ytext(self):
        """Build y-axis labels from depth levels. Empty unless ``self.levels`` is set."""
        self.ytext = []
        if hasattr(self, "levels") and self.levels:
            for level in self.levels:
                for i in range(len(self.vars)):
                    if i == 0:
                        self.ytext.append(f"{level}m")
                    else:
                        self.ytext.append(None)

    def set_label_line_plot(self):
        """Set the legend labels for the model and observation lines."""
        self.data_label = self.model
        if self.obs:
            self.obs_label = self.obs.attrs.get("model", "Observation")

    def set_data_list(self):
        """Populate the data and reference data lists for plotting."""
        self.data_list = [self.data]
        if self.obs:
            self.ref_data_list = [self.obs]
        # for data in self.data:
        #     for var in self.vars:
        #         data_var = data[[var]]
        #         self.data_list.append(data_var)

    def set_cbar_labels(self, var: str = None):
        """Set the colorbar label for the given variable.

        Args:
            var (str, optional): Variable name to derive the colorbar label from.

        """
        self.cbar_label = cbar_get_label(data=self.data[var], cbar_label=None, loglevel=self.loglevel)

    def _round_up(self, value):
        """Round a value up to the nearest 50 or 100 for colorbar limits."""
        if value % 100 == 0:
            return value  # Already a multiple of 100
        elif value % 100 <= 50:
            return math.ceil(value / 50) * 50  # Round up to next 50
        else:
            return math.ceil(value / 100) * 100  # Round up to next 100

    def set_cbar_limits(self):
        """Set colorbar limits and level count (used by MLD plotting, not stratification profiles)."""
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

    def set_suptitle(self):
        """Set the figure suptitle for the stratification plot."""
        self.suptitle = TitleBuilder(
            diagnostic="Stratification",
            regions=self.region,
            model=self.model,
            exp=self.exp,
            timeseason=f"{self.clim_time} climatology",
        ).generate()
        self.logger.debug(f"Suptitle set to: {self.suptitle}")

    def set_title(self):
        """Set subplot titles from each variable's ``long_name`` (one per column)."""
        self.title_list = []
        for j in range(len(self.data_list)):
            for var in self.vars:
                if j == 0:
                    title = f"{self.data[var].attrs.get('long_name', var)}"
                    self.title_list.append(title)
                else:
                    self.title_list.append(" ")
        self.logger.debug("Title list set to: %s", self.title_list)

    def set_description(self):
        """Build the figure description string including model and observation date ranges."""
        model_startdate = self.data.attrs.get("startdate", None)
        model_enddate = self.data.attrs.get("enddate", None)
        self.description = (
            f"Vertical profiles of temperature, salinity and density for the spatially averaged"
            f" {self.region} region, {self.clim_time} climatology for {self.model} {self.exp} (solid)"
        )
        if model_startdate and model_enddate:
            self.description += (
                f" (from {time_to_string(model_startdate, format='%Y-%m')} to {time_to_string(model_enddate, format='%Y-%m')})"
            )
        if self.obs:
            obs_startdate = self.obs.attrs.get("startdate", None)
            obs_enddate = self.obs.attrs.get("enddate", None)
            self.description += f" with reference {self.obs.attrs['model']} {self.obs.attrs['exp']} (dashed)"
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
        metadata: dict = None,
        dpi: int = 300,
        format: Union[str, list] = SAVE_FORMAT,
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
            extension=format,
            dpi=dpi,
        )
