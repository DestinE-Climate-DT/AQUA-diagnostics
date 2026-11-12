"""Plot the regional sea ice extent or volume computed by SeaIce, as time series or seasonal cycles"""

from typing import Union

import matplotlib.pyplot as plt
import xarray as xr

from aqua.core.graphics import ConfigStyle, plot_seasonalcycle, plot_timeseries
from aqua.core.logger import log_configure
from aqua.core.util import get_realizations, strlist_to_phrase, to_list
from aqua.diagnostics.base import SAVE_FORMAT, OutputSaver, TitleBuilder

from .util import _check_list_regions_type, extract_dates, extract_std_period_dates

xr.set_options(keep_attrs=True)


class PlotSeaIce:
    """
    Plot the integrated sea ice extent or volume of one or more models against reference data.

    Every input is a result of ``SeaIce`` (a Dataset with one variable per region) or a list of them, all for the
    same method. The figure has one panel per region; models, method, regions and periods are read from the
    ``AQUA_*`` attributes of the data.

    Args:
        monthly_models (xr.Dataset | list[xr.Dataset], optional): Monthly series, or seasonal cycles, of the models.
        annual_models (xr.Dataset | list[xr.Dataset], optional): Annual means of the models.
        monthly_ref (xr.Dataset | list[xr.Dataset], optional): Monthly series, or seasonal cycles, of the references.
        annual_ref (xr.Dataset | list[xr.Dataset], optional): Annual means of the references.
        monthly_std_ref (xr.Dataset | list[xr.Dataset], optional): Standard deviation of the references by month.
        annual_std_ref (xr.Dataset | list[xr.Dataset], optional): Standard deviation of the annual reference means.
        regions_to_plot (list, optional): Region long names to plot (e.g. ['Arctic', 'Antarctic']).
            If None, all the regions of the data are plotted.
        diagnostic_name (str, optional): The name used for the output files. Default is 'seaice'.
        outputdir (str, optional): Directory to save output plots. Default is './'.
        rebuild (bool, optional): Whether to overwrite existing figures. Default is True.
        dpi (int, optional): Resolution of saved figures. Default is 300.
        loglevel (str, optional): Logging level. Default is 'WARNING'.
    """

    def __init__(
        self,
        monthly_models=None,
        annual_models=None,
        monthly_ref=None,
        annual_ref=None,
        monthly_std_ref=None,
        annual_std_ref=None,
        regions_to_plot: list = None,
        diagnostic_name: str = "seaice",
        outputdir="./",
        rebuild=True,
        dpi=300,
        loglevel="WARNING",
    ):
        self.loglevel = loglevel
        self.logger = log_configure(log_level=self.loglevel, log_name="PlotSeaIce")

        # input_name of data (is a list of Datasets); the None entries of a list (a missing dataset) are skipped
        inputs = {
            "monthly_models": monthly_models,
            "annual_models": annual_models,
            "monthly_ref": monthly_ref,
            "annual_ref": annual_ref,
            "monthly_std_ref": monthly_std_ref,
            "annual_std_ref": annual_std_ref,
        }
        self.data = {input_name: [ds for ds in to_list(value) if ds is not None] for input_name, value in inputs.items()}

        if regions_to_plot is not None:
            regions_to_plot = _check_list_regions_type(regions_to_plot, logger=self.logger)
        self.regions_to_plot = regions_to_plot

        self.diagnostic_name = diagnostic_name
        self.outputdir = outputdir
        self.rebuild = rebuild
        self.dpi = dpi

    def _region_data(self, input_name: str, region: str):
        """Variables of a region among the datasets of one input_name, None if there are none."""
        data = [da for ds in self.data[input_name] for da in ds.data_vars.values() if da.attrs["AQUA_region"] == region]
        return data or None

    @staticmethod
    def _label(data: xr.DataArray) -> str:
        """Legend label of a dataset: model and experiment."""
        return " ".join(str(data.attrs[key]) for key in ("AQUA_model", "AQUA_exp") if key in data.attrs)

    def set_description(self, plot_type: str, method: str, regions: list) -> str:
        """
        Build the caption of the figure from the attributes of the data.

        Args:
            plot_type (str): 'timeseries' or 'seasonalcycle'.
            method (str): The sea ice method plotted.
            regions (list): The regions plotted.

        Returns:
            str: The caption.
        """
        input_name = "Seasonal cycle of " if plot_type == "seasonalcycle" else "Time series of "
        regions_str = f"{strlist_to_phrase(regions)} region{'s' if len(regions) != 1 else ''} "

        # models and std cover the same period in every region: read it from the last one
        models = self._region_data("monthly_models", regions[-1]) or []
        models_str = strlist_to_phrase([f"{self._label(da)} (from {' to '.join(extract_dates(da))})" for da in models])

        # one reference per region: the same dataset is mentioned once
        refs = []
        for region in regions:
            for da in self._region_data("monthly_ref", region) or []:
                ref = f"{self._label(da)} (from {' to '.join(extract_dates(da))})".strip()
                if ref not in refs:
                    refs.append(ref)
        if not refs:
            refs_str = ""
        elif len(refs) == 1:
            refs_str = f" {refs[0]} is used as a reference."
        elif len(refs) == 2:
            refs_str = f" {refs[0]} and {refs[1]} are used as reference data for the respective regions."
        else:
            refs_str = f" {strlist_to_phrase(refs, oxford_comma=True)} are used as references."

        std = self._region_data("monthly_std_ref", regions[-1])
        std_str = ""
        if std:
            std_str = f" Shaded areas represent ±2σ uncertainty bands (from {' to '.join(extract_std_period_dates(std[0]))})."

        return f"{input_name}sea ice {method} integrated over {regions_str}for {models_str} {refs_str}{std_str}"

    def plot_seaice(
        self,
        plot_type: str = "timeseries",
        save_format: Union[str, list] = SAVE_FORMAT,
        style: str = None,
        show: bool = False,
        **kwargs,
    ):
        """
        Plot the sea ice data of each region, either as time series or as seasonal cycle, and save the figure.

        Args:
            plot_type (str, optional): 'timeseries' or 'seasonalcycle'. Default is 'timeseries'.
            save_format (str or list, optional): Format(s) to save the figure. Default is SAVE_FORMAT.
            style (str, optional): Plotting style. Default is the one of the configuration file.
            show (bool, optional): If True, display the figure instead of closing it. Default is False.
            **kwargs: Additional keyword arguments passed to the plotting function.

        Raises:
            ValueError: If the plot type is not valid.
        """
        if plot_type not in ("timeseries", "seasonalcycle"):
            raise ValueError("Invalid plot_type. Allowed plots are: ['timeseries', 'seasonalcycle']")

        models = self.data["monthly_models"]
        # regions in order of appearance in the data
        arrays = [da for datasets in self.data.values() for ds in datasets for da in ds.data_vars.values()]
        regions = list(dict.fromkeys(da.attrs["AQUA_region"] for da in arrays))
        if self.regions_to_plot:
            regions = [region for region in regions if region in self.regions_to_plot]
        if not models or not regions:
            self.logger.warning("No model data for the requested regions: nothing to plot")
            return
        method = models[0].attrs.get("AQUA_method", "Unknown")
        self.logger.info("Plotting sea ice %s %s for %s", method, plot_type, regions)

        ConfigStyle(style=style, loglevel=self.loglevel)
        width = 6 if plot_type == "seasonalcycle" else 10
        fig, axes = plt.subplots(nrows=len(regions), ncols=1, figsize=(width, 4 * len(regions)), squeeze=False)

        for ax, region in zip(axes.flatten(), regions):
            data = self._region_data("monthly_models", region)
            ref = self._region_data("monthly_ref", region)
            std = self._region_data("monthly_std_ref", region)
            data_labels = [self._label(da) for da in data] if data else None

            if plot_type == "timeseries":
                plot_timeseries(
                    monthly_data=data,
                    annual_data=self._region_data("annual_models", region),
                    ref_monthly_data=ref,
                    ref_annual_data=self._region_data("annual_ref", region),
                    std_monthly_data=std,
                    std_annual_data=self._region_data("annual_std_ref", region),
                    data_labels=data_labels,
                    ref_label=[self._label(da) for da in ref] if ref else None,
                    style=style,
                    fig=fig,
                    ax=ax,
                    **kwargs,
                )
            else:
                # plot_seasonalcycle draws one reference
                plot_seasonalcycle(
                    data=data,
                    ref_data=ref[0] if ref else None,
                    std_data=std[0] if std else None,
                    data_labels=data_labels,
                    ref_label=self._label(ref[0]) if ref else None,
                    style=style,
                    fig=fig,
                    ax=ax,
                    **kwargs,
                )
            ax.set_title(TitleBuilder(diagnostic=f"Sea ice {method}", regions=region).generate())

        plt.tight_layout()

        if save_format:
            outputsaver = OutputSaver(
                diagnostic=self.diagnostic_name,
                catalog=[ds.attrs.get("AQUA_catalog") for ds in models],
                model=[ds.attrs.get("AQUA_model") for ds in models],
                exp=[ds.attrs.get("AQUA_exp") for ds in models],
                realization=get_realizations(models),
                outputdir=self.outputdir,
                loglevel=self.loglevel,
            )
            outputsaver.save_figure(
                fig=fig,
                diagnostic_product=plot_type,
                extra_keys={"method": method, "region": "_".join(regions)},
                metadata={"Description": self.set_description(plot_type, method, regions)},
                rebuild=self.rebuild,
                extension=save_format,
                dpi=self.dpi,
            )
        if show:
            plt.show()
        plt.close(fig)
