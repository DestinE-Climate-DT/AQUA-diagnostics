"""Plot class of the Budget tool."""

import matplotlib.pyplot as plt

from aqua.core.graphics import plot_timeseries
from aqua.core.logger import log_configure
from aqua.core.util import DEFAULT_REALIZATION, time_to_string, unit_to_latex
from aqua.diagnostics.base import SAVE_FORMAT, OutputSaver, TitleBuilder


class PlotBudget:
    """
    Plot the result of a Budget instance: the two quantities together, then their difference.

    One dataset only: extend to lists of results if a multi-model comparison is needed (shortcut).
    The class never touches a Reader: dataset, period and names are read from the attributes of the result,
    so a result can be plotted again from its NetCDF file.
    """

    def __init__(
        self,
        data,
        name: str,
        expected: float = 0.0,
        diagnostic_name: str = "budget",
        outputdir: str = ".",
        loglevel: str = "WARNING",
    ):
        """
        Initialize the PlotBudget class.

        Args:
            data (xr.Dataset): The result of Budget.run(), with the 'first' and 'second' quantities.
            name (str): The name of the budget, used in the filenames (as in Budget.run()).
            expected (float, optional): The expected value of first minus second: 0 for a closed budget,
                another value e.g. for the radiation imbalance of a warming climate. Default is 0.
            diagnostic_name (str, optional): The name used for the output files. Default is 'budget'.
            outputdir (str, optional): The output directory. Default is '.'.
            loglevel (str, optional): The log level. Default is 'WARNING'.
        """
        self.loglevel = loglevel
        self.logger = log_configure(log_level=loglevel, log_name="PlotBudget")
        self.data = data
        self.name = name
        self.expected = expected
        self.diagnostic_name = diagnostic_name
        self.outputdir = outputdir
        self.get_data_info()

    def get_data_info(self):
        """Read catalog, model, exp, realization, dates and the two quantities from the result."""
        attrs = self.data.attrs
        self.catalog = attrs["AQUA_catalog"]
        self.model = attrs["AQUA_model"]
        self.exp = attrs["AQUA_exp"]
        self.realization = attrs.get("AQUA_realization", DEFAULT_REALIZATION)
        self.startdate = attrs.get("AQUA_startdate")
        self.enddate = attrs.get("AQUA_enddate")
        self.first = self.data["first"]
        self.second = self.data["second"]
        self.units = self.first.attrs.get("units", "")
        self.difference = (self.first - self.second).assign_attrs(long_name="Difference", units=self.units)
        self.mean_difference = float(self.difference.mean())

    def set_data_labels(self) -> list:
        """Return the labels of the two quantities."""
        return [self.first.attrs["long_name"], self.second.attrs["long_name"]]

    def set_title(self, diagnostic: str = "Budget") -> str:
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
        first, second = self.set_data_labels()
        if difference:
            description = f"Global mean of the {first} minus the {second}"
        else:
            description = f"Global mean of the {first} and of the {second}"
        description += f" for {self.model} {self.exp}"
        if self.startdate and self.enddate:
            description += f" from {time_to_string(self.startdate, '%Y')} to {time_to_string(self.enddate, '%Y')}"
        description += "."
        if difference:
            description += (
                f" The dashed line is the mean difference over the period ({self.mean_difference:.2f} {self.units}),"
                f" the dotted line the expected value ({self.expected:g} {self.units})."
            )
        return description

    def create_budget_figure(self):
        """Draw the two quantities with the aqua-core timeseries graphics."""
        fig, ax = plot_timeseries(
            monthly_data=[self.first, self.second],
            data_labels=self.set_data_labels(),
            title=self.set_title(),
            loglevel=self.loglevel,
        )
        ax.set_ylabel(f"[{unit_to_latex(self.units)}]")  # the automatic label would be the one of the last curve only
        return fig, ax

    def create_difference_figure(self):
        """Draw the difference between the two quantities, its mean and its expected value."""
        fig, ax = plot_timeseries(
            monthly_data=self.difference,
            data_labels=["Difference"],
            title=self.set_title("Budget difference"),
            legend=False,
            loglevel=self.loglevel,
        )
        ax.set_ylabel(f"Difference [{unit_to_latex(self.units)}]")
        ax.axhline(self.mean_difference, color="r", linestyle="--", label=f"mean difference ({self.mean_difference:.2f})")
        ax.axhline(self.expected, color="k", linestyle=":", label=f"expected ({self.expected:g})")
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
            extra_keys={"budget": self.name},
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
