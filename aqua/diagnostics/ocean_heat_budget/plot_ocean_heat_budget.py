"""Plot class of the OceanHeatBudget tool."""

import matplotlib.pyplot as plt

from aqua.core.graphics import plot_timeseries
from aqua.core.logger import log_configure
from aqua.core.util import DEFAULT_REALIZATION, time_to_string
from aqua.diagnostics.base import SAVE_FORMAT, OutputSaver, TitleBuilder


class PlotOceanHeatBudget:
    """
    Plot the result of an OceanHeatBudget instance: the budget terms and their difference.

    One dataset only: extend to lists of results if a multi-model comparison is needed (shortcut).
    The class never touches a Reader: everything it needs (dataset, period, names) is read from the
    attributes of the result, so a result can be plotted again from its NetCDF file.
    """

    def __init__(
        self,
        data,
        diagnostic_name: str = "ocean_heat_budget",
        outputdir: str = ".",
        loglevel: str = "WARNING",
    ):
        """
        Initialize the PlotOceanHeatBudget class.

        Args:
            data (xr.Dataset): The result of OceanHeatBudget.run(), with 'net_surface_flux' and 'hc_tendency'.
            diagnostic_name (str, optional): The name used for the output files. Default is 'ocean_heat_budget'.
            outputdir (str, optional): The output directory. Default is '.'.
            loglevel (str, optional): The log level. Default is 'WARNING'.
        """
        self.loglevel = loglevel
        self.logger = log_configure(log_level=loglevel, log_name="PlotOceanHeatBudget")
        self.data = data
        self.diagnostic_name = diagnostic_name
        self.outputdir = outputdir
        self.get_data_info()

    def get_data_info(self):
        """Read catalog, model, exp, realization, dates and names from the AQUA_* attributes."""
        attrs = self.data.attrs
        self.catalog = attrs["AQUA_catalog"]
        self.model = attrs["AQUA_model"]
        self.exp = attrs["AQUA_exp"]
        self.realization = attrs.get("AQUA_realization", DEFAULT_REALIZATION)
        self.startdate = attrs.get("AQUA_startdate")
        self.enddate = attrs.get("AQUA_enddate")
        self.flux = self.data["net_surface_flux"]
        self.tendency = self.data["hc_tendency"]
        self.difference = (self.flux - self.tendency).assign_attrs(
            long_name="Net surface heat flux minus heat content tendency", units=self.flux.attrs.get("units")
        )

    def set_data_labels(self) -> list:
        """Return the labels of the two budget terms."""
        return [self.flux.attrs["long_name"], self.tendency.attrs["long_name"]]

    def set_title(self, diagnostic: str = "Ocean heat budget") -> str:
        """Return the title of a figure, built with TitleBuilder."""
        return TitleBuilder(
            diagnostic=diagnostic,
            model=self.model,
            exp=self.exp,
            realizations=self.realization,
            startyear=time_to_string(self.startdate, "%Y") if self.startdate else None,
            endyear=time_to_string(self.enddate, "%Y") if self.enddate else None,
        ).generate(max_chars=80)

    def set_description(self, difference: bool = False) -> str:
        """Return the caption of a figure: what is shown, for which dataset and period."""
        flux, tendency = self.set_data_labels()
        if difference:
            description = f"Global ocean mean of the difference between the {flux.lower()} and the {tendency.lower()}"
        else:
            description = f"Global ocean mean of the {flux.lower()} and the {tendency.lower()}"
        description += f" for {self.model} {self.exp}"
        if self.startdate and self.enddate:
            description += f" from {time_to_string(self.startdate, '%Y')} to {time_to_string(self.enddate, '%Y')}"
        description += ". Fluxes are positive downward, i.e. into the ocean."
        if difference:
            description += " The dashed line is the mean of the difference over the period."
        return description

    def create_budget_figure(self):
        """Draw the two budget terms with the aqua-core timeseries graphics."""
        fig, ax = plot_timeseries(
            monthly_data=[self.flux, self.tendency],
            data_labels=self.set_data_labels(),
            title=self.set_title(),
            loglevel=self.loglevel,
        )
        ax.set_ylabel(r"Heat flux [W m$^{-2}$]")  # the automatic label would be the one of the last curve only
        return fig, ax

    def create_difference_figure(self):
        """Draw the difference between the two budget terms and its mean."""
        fig, ax = plot_timeseries(
            monthly_data=self.difference,
            data_labels=[self.difference.attrs["long_name"]],
            title=self.set_title("Ocean heat budget difference"),
            legend=False,
            loglevel=self.loglevel,
        )
        ax.set_ylabel(r"Heat flux difference [W m$^{-2}$]")
        ax.axhline(float(self.difference.mean()), color="r", linestyle="--", label="mean difference")
        ax.legend()
        return fig, ax

    def save_plot(
        self, fig, diagnostic_product: str, description: str, save_format=SAVE_FORMAT, dpi: int = 300, rebuild: bool = True
    ):
        """
        Save a figure through OutputSaver.

        Args:
            fig (matplotlib.figure.Figure): The figure to save.
            diagnostic_product (str): The name of the product in the filename.
            description (str): The caption embedded in the figure metadata.
            save_format (str or list, optional): The format(s) of the figure. Default is SAVE_FORMAT.
            dpi (int, optional): The resolution of raster formats. Default is 300.
            rebuild (bool, optional): If True, overwrite existing files. Default is True.
        """
        saver = OutputSaver(
            diagnostic=self.diagnostic_name,
            outputdir=self.outputdir,
            catalog=self.catalog,
            model=self.model,
            exp=self.exp,
            realization=self.realization,
            loglevel=self.loglevel,
        )
        saver.save_figure(
            fig,
            diagnostic_product=diagnostic_product,
            metadata={"description": description},
            extension=save_format,
            dpi=dpi,
            rebuild=rebuild,
        )

    def plot(self, save_format=SAVE_FORMAT, dpi: int = 300, rebuild: bool = True, show: bool = False):
        """
        Draw and save the budget and difference figures.

        Args:
            save_format (str or list, optional): The format(s) of the figures. Default is SAVE_FORMAT.
            dpi (int, optional): The resolution of raster formats. Default is 300.
            rebuild (bool, optional): If True, overwrite existing files. Default is True.
            show (bool, optional): If True, keep the figures open for the caller. Default is False.

        Returns:
            list: The (figure, axes) pairs, budget first.
        """
        figures = []
        for product, create, difference in (
            ("budget_timeseries", self.create_budget_figure, False),
            ("budget_difference", self.create_difference_figure, True),
        ):
            fig, ax = create()
            self.save_plot(fig, product, self.set_description(difference), save_format=save_format, dpi=dpi, rebuild=rebuild)
            if not show:
                plt.close(fig)
            figures.append((fig, ax))
        return figures
