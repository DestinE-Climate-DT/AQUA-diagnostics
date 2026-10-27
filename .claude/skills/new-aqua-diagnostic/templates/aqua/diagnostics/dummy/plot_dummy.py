"""Plot class of the Dummy tool."""

import matplotlib.pyplot as plt

from aqua.core.graphics import plot_timeseries
from aqua.core.logger import log_configure
from aqua.core.util import DEFAULT_REALIZATION, strlist_to_phrase, time_to_string, to_list
from aqua.diagnostics.base import SAVE_FORMAT, OutputSaver, TitleBuilder


class PlotDummy:
    """
    Plot the results of one or more Dummy instances.

    The class never touches a Reader: everything it needs (datasets, region, period) is read from the
    AQUA_* attributes of the results, so a result can be plotted again from its NetCDF file.
    """

    def __init__(
        self,
        data,
        ref_data=None,
        diagnostic_name: str = "dummy",
        outputdir: str = ".",
        loglevel: str = "WARNING",
    ):
        """
        Initialize the PlotDummy class.

        Args:
            data (xr.DataArray or list): The results of Dummy.run(), one per dataset.
            ref_data (xr.DataArray, optional): The result for a reference dataset.
            diagnostic_name (str, optional): The name used for the output files. Default is 'dummy'.
            outputdir (str, optional): The output directory. Default is '.'.
            loglevel (str, optional): The log level. Default is 'WARNING'.
        """
        self.loglevel = loglevel
        self.logger = log_configure(log_level=loglevel, log_name="PlotDummy")
        self.data = to_list(data)
        self.ref_data = ref_data
        self.diagnostic_name = diagnostic_name
        self.outputdir = outputdir
        self.get_data_info()

    def get_data_info(self):
        """Read catalog, model, exp, realization, region, dates and names from the AQUA_* attributes."""
        first = self.data[0]
        self.catalogs = [d.attrs["AQUA_catalog"] for d in self.data]
        self.models = [d.attrs["AQUA_model"] for d in self.data]
        self.exps = [d.attrs["AQUA_exp"] for d in self.data]
        self.realizations = [d.attrs.get("AQUA_realization", DEFAULT_REALIZATION) for d in self.data]
        self.region = first.attrs.get("AQUA_region")
        self.startdate = first.attrs.get("AQUA_startdate")
        self.enddate = first.attrs.get("AQUA_enddate")
        self.short_name = first.attrs.get("short_name", first.name)
        self.long_name = first.attrs.get("long_name", self.short_name)
        self.units = first.attrs.get("units")

    def set_data_labels(self) -> list:
        """Return one label per dataset."""
        return [f"{model} {exp}" for model, exp in zip(self.models, self.exps)]

    def set_ref_label(self) -> str:
        """Return the label of the reference, or None."""
        if self.ref_data is None:
            return None
        return f"{self.ref_data.attrs['AQUA_model']} {self.ref_data.attrs['AQUA_exp']}"

    def set_title(self) -> str:
        """Return the title of the figure, built with TitleBuilder."""
        return TitleBuilder(
            diagnostic="Annual mean",
            variable=self.long_name,
            regions=self.region,
            model=self.models,
            exp=self.exps,
            realizations=self.realizations,
            startyear=time_to_string(self.startdate, "%Y") if self.startdate else None,
            endyear=time_to_string(self.enddate, "%Y") if self.enddate else None,
        ).generate(max_chars=80)

    def set_description(self) -> str:
        """Return the caption of the figure: what is shown, for which datasets, region and period."""
        description = f"Annual mean of {self.long_name}"
        if self.region is not None:
            description += f" over {self.region}"
        description += f" for {strlist_to_phrase(self.set_data_labels())}"
        if self.startdate and self.enddate:
            description += f" from {time_to_string(self.startdate, '%Y')} to {time_to_string(self.enddate, '%Y')}"
        if self.ref_data is not None:
            description += f", compared with {self.set_ref_label()}"
        return description + "."

    def create_figure(self):
        """Draw the figure with the aqua-core graphics."""
        return plot_timeseries(
            annual_data=self.data,
            ref_annual_data=self.ref_data,
            data_labels=self.set_data_labels(),
            ref_label=self.set_ref_label(),
            title=self.set_title(),
            loglevel=self.loglevel,
        )

    def save_plot(self, fig, save_format=SAVE_FORMAT, dpi: int = 300, rebuild: bool = True):
        """
        Save the figure through OutputSaver.

        Args:
            fig (matplotlib.figure.Figure): The figure to save.
            save_format (str or list, optional): The format(s) of the figure. Default is SAVE_FORMAT.
            dpi (int, optional): The resolution of raster formats. Default is 300.
            rebuild (bool, optional): If True, overwrite existing files. Default is True.
        """
        # All the plotted models and references go to the OutputSaver
        saver_kwargs = {
            "catalog": self.catalogs,
            "model": self.models,
            "exp": self.exps,
            "realization": OutputSaver.unpack_list(self.realizations),  # one value gives r1, a list gives [r1]
        }
        if self.ref_data is not None:
            saver_kwargs.update(
                catalog_ref=self.ref_data.attrs["AQUA_catalog"],
                model_ref=self.ref_data.attrs["AQUA_model"],
                exp_ref=self.ref_data.attrs["AQUA_exp"],
            )
        saver = OutputSaver(diagnostic=self.diagnostic_name, outputdir=self.outputdir, loglevel=self.loglevel, **saver_kwargs)

        extra_keys = {"var": self.short_name}
        if self.region is not None:
            extra_keys["region"] = self.region
        saver.save_figure(
            fig,
            diagnostic_product="annual_mean",
            extra_keys=extra_keys,
            metadata={"description": self.set_description()},
            extension=save_format,
            dpi=dpi,
            rebuild=rebuild,
        )

    def plot(self, save_format=SAVE_FORMAT, dpi: int = 300, rebuild: bool = True, show: bool = False):
        """
        Draw and save the figure.

        Args:
            save_format (str or list, optional): The format(s) of the figure. Default is SAVE_FORMAT.
            dpi (int, optional): The resolution of raster formats. Default is 300.
            rebuild (bool, optional): If True, overwrite existing files. Default is True.
            show (bool, optional): If True, keep the figure open for the caller. Default is False.

        Returns:
            tuple: The matplotlib figure and axes.
        """
        fig, ax = self.create_figure()
        self.save_plot(fig, save_format=save_format, dpi=dpi, rebuild=rebuild)
        if not show:
            plt.close(fig)
        return fig, ax
