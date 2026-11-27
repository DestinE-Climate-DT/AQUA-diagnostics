"""Plot class of the Budget tool."""

from aqua.core.graphics import plot_timeseries
from aqua.core.util import time_to_string, unit_to_latex
from aqua.diagnostics.base import SAVE_FORMAT, PlotBase


class PlotBudget(PlotBase):
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
        rebuild: bool = True,
        save_format=SAVE_FORMAT,
        dpi: int = 300,
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
            rebuild (bool, optional): If True, overwrite existing files. Default is True.
            save_format (str or list, optional): The format(s) of the figures. Default is SAVE_FORMAT.
            dpi (int, optional): The resolution of raster formats. Default is 300.
            loglevel (str, optional): The log level. Default is 'WARNING'.
        """
        super().__init__(
            diagnostic_name=diagnostic_name,
            outputdir=outputdir,
            rebuild=rebuild,
            save_format=save_format,
            dpi=dpi,
            loglevel=loglevel,
        )
        self.data = data
        self.name = name
        self.expected = expected
        self.get_data_info(data)

    def get_data_info(self, data, ref_data=None):
        """Read dataset and period from the result (see PlotBase), then the two quantities and their difference."""
        super().get_data_info(data, ref_data)
        self.first = data["first"]
        self.second = data["second"]
        self.units = self.first.attrs.get("units", "")
        self.difference = (self.first - self.second).assign_attrs(long_name="Difference", units=self.units)
        self.mean_difference = float(self.difference.mean())

    def _years(self) -> tuple:
        """Return the first and last year of the period, None when unknown."""
        startdate, enddate = self.startdates[0], self.enddates[0]
        return (
            time_to_string(startdate, "%Y") if startdate else None,
            time_to_string(enddate, "%Y") if enddate else None,
        )

    def set_data_labels(self) -> list:
        """Return the labels of the two quantities."""
        return [self.first.attrs["long_name"], self.second.attrs["long_name"]]

    def set_title(self, diagnostic: str = "Budget") -> str:
        """Return the title of a figure, built with TitleBuilder."""
        startyear, endyear = self._years()
        return super().set_title(
            diagnostic=diagnostic, realizations=self.realizations, startyear=startyear, endyear=endyear, max_chars=80
        )

    def set_description(self, difference: bool = False) -> str:
        """Return the caption of a figure: what is shown, for which dataset and period."""
        first, second = self.set_data_labels()
        if difference:
            description = f"Global mean of the {first} minus the {second}"
        else:
            description = f"Global mean of the {first} and of the {second}"
        description += f" for {self.models[0]} {self.exps[0]}"
        startyear, endyear = self._years()
        if startyear and endyear:
            description += f" from {startyear} to {endyear}"
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

    def plot(self, show: bool = False):
        """
        Draw and save the budget and difference figures.

        Args:
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
            self.save_plot(
                fig,
                diagnostic_product=product,
                extra_keys={"budget": self.name},
                description=self.set_description(difference),
                close=not show,
            )
            figures.append((fig, ax))
        return figures
