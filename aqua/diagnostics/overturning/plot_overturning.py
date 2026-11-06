"""Plot class of the Overturning tool."""

import matplotlib.pyplot as plt
import numpy as np

from aqua.core.graphics import plot_timeseries, plot_vertical_profile
from aqua.core.logger import log_configure
from aqua.core.util import DEFAULT_REALIZATION, strlist_to_phrase, time_to_string, to_list
from aqua.diagnostics.base import SAVE_FORMAT, OutputSaver, TitleBuilder


class PlotOverturning:
    """
    Plot the results of one or more Overturning instances: the time series of the overturning index at each
    latitude, and the time-mean streamfunction as a latitude-depth section.

    The class never touches a Reader: everything it needs (datasets, basin, period) is read from the
    AQUA_* attributes of the results, so a result can be plotted again from its NetCDF files.
    """

    def __init__(
        self,
        streamfunction=None,
        index_monthly=None,
        index_annual=None,
        ref_index_monthly=None,  # REFERENCE
        ref_index_annual=None,  # REFERENCE
        diagnostic_name: str = "overturning",
        outputdir: str = ".",
        loglevel: str = "WARNING",
    ):
        """
        Initialize the PlotOverturning class.

        Args:
            streamfunction (xr.DataArray or list, optional): Time-mean streamfunctions (depth, lat), one per dataset.
            index_monthly (xr.DataArray or list, optional): Monthly indices (time, lat), one per dataset.
            index_annual (xr.DataArray or list, optional): Annual indices (time, lat), one per dataset.
            ref_index_monthly (xr.DataArray, optional): The monthly index of a reference (e.g. RAPID).
            ref_index_annual (xr.DataArray, optional): The annual index of a reference.
            diagnostic_name (str, optional): The name used for the output files. Default is 'overturning'.
            outputdir (str, optional): The output directory. Default is '.'.
            loglevel (str, optional): The log level. Default is 'WARNING'.
        """
        self.loglevel = loglevel
        self.logger = log_configure(log_level=loglevel, log_name="PlotOverturning")
        self.streamfunction = [d for d in to_list(streamfunction) if d is not None]
        self.index_monthly = [d for d in to_list(index_monthly) if d is not None]
        self.index_annual = [d for d in to_list(index_annual) if d is not None]
        self.ref_index_monthly = ref_index_monthly  # REFERENCE
        self.ref_index_annual = ref_index_annual  # REFERENCE
        self.diagnostic_name = diagnostic_name
        self.outputdir = outputdir
        self.get_data_info()

    def get_data_info(self):
        """Read catalog, model, exp, realization, basin and dates from the AQUA_* attributes."""
        data = self.streamfunction or self.index_annual or self.index_monthly
        if not data:
            raise ValueError("Nothing to plot: no streamfunction and no index given")
        self.catalogs = [d.attrs["AQUA_catalog"] for d in data]
        self.models = [d.attrs["AQUA_model"] for d in data]
        self.exps = [d.attrs["AQUA_exp"] for d in data]
        self.realizations = [d.attrs.get("AQUA_realization", DEFAULT_REALIZATION) for d in data]
        self.region = data[0].attrs.get("AQUA_region")
        self.startdate = data[0].attrs.get("AQUA_startdate")
        self.enddate = data[0].attrs.get("AQUA_enddate")

    @property
    def ref_data(self):  # REFERENCE
        """The reference index, annual or monthly, or None."""
        return self.ref_index_annual if self.ref_index_annual is not None else self.ref_index_monthly

    def set_data_labels(self) -> list:
        """Return one label per dataset."""
        return [f"{model} {exp}" for model, exp in zip(self.models, self.exps)]

    def set_ref_label(self) -> str:  # REFERENCE (the whole method)
        """Return the label of the reference, or None."""
        if self.ref_data is None:
            return None
        return f"{self.ref_data.attrs['AQUA_model']} {self.ref_data.attrs['AQUA_exp']}"

    @staticmethod
    def _period(start, end) -> str:
        """Return 'YYYY-YYYY', or an empty string without dates."""
        if not (start and end):
            return ""
        return f"{time_to_string(start, '%Y')}-{time_to_string(end, '%Y')}"

    def set_title(self, product: str, lat: float = None, index: int = None) -> str:
        """
        Return the title of a figure, built with TitleBuilder.

        Args:
            product (str): 'index' or 'streamfunction'.
            lat (float, optional): The latitude of the index.
            index (int, optional): The dataset of a streamfunction section (all datasets if None).
        """
        select = slice(None) if index is None else slice(index, index + 1)
        diagnostic = f"Overturning at {lat:g}N" if product == "index" else "Overturning streamfunction"
        return TitleBuilder(
            diagnostic=diagnostic,
            regions=self.region,
            model=self.models[select],
            exp=self.exps[select],
            realizations=self.realizations[select],
            startyear=time_to_string(self.startdate, "%Y") if self.startdate else None,
            endyear=time_to_string(self.enddate, "%Y") if self.enddate else None,
        ).generate(max_chars=80)

    def set_description(self, product: str, data, lat: float = None, labels: list = None) -> str:
        """
        Return the caption of a figure: what is shown, for which datasets, basin and period.

        Args:
            product (str): 'index' or 'streamfunction'.
            data (xr.DataArray): One of the plotted results, for its attributes.
            lat (float, optional): The latitude of the index.
            labels (list, optional): The labels of the plotted datasets. Default is all of them.
        """
        basin = f" of the {self.region}" if self.region else ""
        if product == "index":
            description = f"Overturning index{basin} at {lat:g}N, the {data.attrs.get('long_name')}"
        else:
            description = f"Time-mean meridional overturning streamfunction{basin} as a function of latitude and depth"
        description += f", in Sv ({data.attrs.get('comment', '')})"
        description += f", for {strlist_to_phrase(labels or self.set_data_labels())}"
        period = self._period(self.startdate, self.enddate)
        if period:
            description += f" over {period}"
        if product == "index" and self.ref_data is not None:  # REFERENCE
            description += f", compared with {self.set_ref_label()}"  # REFERENCE
            ref_period = self._period(self.ref_data.attrs.get("AQUA_startdate"), self.ref_data.attrs.get("AQUA_enddate"))
            if ref_period:  # REFERENCE
                description += f" ({ref_period})"  # REFERENCE
        return description + "."

    def plot_index(self, save_format=SAVE_FORMAT, dpi: int = 300, rebuild: bool = True, show: bool = False) -> list:
        """
        Draw and save one time series of the overturning index per latitude.

        Args:
            save_format (str or list, optional): The format(s) of the figure. Default is SAVE_FORMAT.
            dpi (int, optional): The resolution of raster formats. Default is 300.
            rebuild (bool, optional): If True, overwrite existing files. Default is True.
            show (bool, optional): If True, keep the figures open for the caller. Default is False.

        Returns:
            list: The (fig, ax) tuples.
        """
        data = self.index_annual or self.index_monthly
        if not data:
            self.logger.warning("No overturning index to plot")
            return []

        def _at(series, lat):
            if series is None or lat not in series["lat"].values:
                return None
            return series.sel(lat=lat)

        figures = []
        for lat in data[0]["lat"].values:
            fig, ax = plot_timeseries(
                monthly_data=[_at(d, lat) for d in self.index_monthly] or None,
                annual_data=[_at(d, lat) for d in self.index_annual] or None,
                ref_monthly_data=_at(self.ref_index_monthly, lat),  # REFERENCE
                ref_annual_data=_at(self.ref_index_annual, lat),  # REFERENCE
                data_labels=self.set_data_labels(),
                ref_label=self.set_ref_label(),  # REFERENCE
                title=self.set_title("index", lat=lat),
                loglevel=self.loglevel,
            )
            ax.set_ylabel("Overturning [Sv]")
            self.save_plot(
                fig,
                diagnostic_product="index",
                extra_keys={"lat": f"{lat:g}N"},
                description=self.set_description("index", data[0], lat=lat),
                save_format=save_format,
                dpi=dpi,
                rebuild=rebuild,
            )
            if not show:
                plt.close(fig)
            figures.append((fig, ax))
        return figures

    def plot_streamfunction(
        self,
        vmin: float = -20.0,
        vmax: float = 20.0,
        nlevels: int = 21,
        save_format=SAVE_FORMAT,
        dpi: int = 300,
        rebuild: bool = True,
        show: bool = False,
    ) -> list:
        """
        Draw and save the latitude-depth section of the time-mean streamfunction, one figure per dataset.

        Args:
            vmin (float, optional): The lower limit of the colour scale (Sv). Default is -20.
            vmax (float, optional): The upper limit of the colour scale (Sv). Default is 20.
            nlevels (int, optional): The number of contour levels. Default is 21 (2 Sv steps).
            save_format (str or list, optional): The format(s) of the figure. Default is SAVE_FORMAT.
            dpi (int, optional): The resolution of raster formats. Default is 300.
            rebuild (bool, optional): If True, overwrite existing files. Default is True.
            show (bool, optional): If True, keep the figures open for the caller. Default is False.

        Returns:
            list: The (fig, ax) tuples.
        """
        figures = []
        levels = np.linspace(vmin, vmax, nlevels)
        for i, psi in enumerate(self.streamfunction):
            vert_coord = next(d for d in psi.dims if d != "lat")
            fig, ax = plot_vertical_profile(
                psi,
                var="Overturning",
                lev_name=vert_coord,
                x_coord="lat",
                vmin=vmin,
                vmax=vmax,
                nlevels=nlevels,
                cmap="RdBu_r",
                add_land=True,
                title=self.set_title("streamfunction", index=i),
                return_fig=True,
                loglevel=self.loglevel,
            )
            lines = ax.contour(psi["lat"], psi[vert_coord], psi, levels=levels, colors="k", linewidths=0.4)
            ax.clabel(lines, fontsize=7, fmt="%g")
            ax.set_ylabel("Depth [m]")
            self.save_plot(
                fig,
                diagnostic_product="streamfunction",
                description=self.set_description("streamfunction", psi, labels=[self.set_data_labels()[i]]),
                index=i,
                save_format=save_format,
                dpi=dpi,
                rebuild=rebuild,
            )
            if not show:
                plt.close(fig)
            figures.append((fig, ax))
        return figures

    def save_plot(
        self,
        fig,
        diagnostic_product: str,
        description: str,
        extra_keys: dict = None,
        index: int = None,
        save_format=SAVE_FORMAT,
        dpi: int = 300,
        rebuild: bool = True,
    ):
        """
        Save a figure through OutputSaver.

        Args:
            fig (matplotlib.figure.Figure): The figure to save.
            diagnostic_product (str): 'index' or 'streamfunction'.
            description (str): The caption, stored in the figure metadata.
            extra_keys (dict, optional): Additional filename keys (the basin is added here).
            index (int, optional): The only dataset of the figure (all datasets and the reference if None).
            save_format (str or list, optional): The format(s) of the figure. Default is SAVE_FORMAT.
            dpi (int, optional): The resolution of raster formats. Default is 300.
            rebuild (bool, optional): If True, overwrite existing files. Default is True.
        """
        select = slice(None) if index is None else slice(index, index + 1)
        saver_kwargs = {
            "catalog": self.catalogs[select],
            "model": self.models[select],
            "exp": self.exps[select],
            "realization": OutputSaver.unpack_list(self.realizations[select]),  # one value gives r1, a list [r1]
        }
        if index is None and self.ref_data is not None:  # REFERENCE (the whole block)
            saver_kwargs.update(
                catalog_ref=self.ref_data.attrs["AQUA_catalog"],
                model_ref=self.ref_data.attrs["AQUA_model"],
                exp_ref=self.ref_data.attrs["AQUA_exp"],
            )
        saver = OutputSaver(diagnostic=self.diagnostic_name, outputdir=self.outputdir, loglevel=self.loglevel, **saver_kwargs)
        keys = dict(extra_keys or {})
        if self.region is not None:
            keys["region"] = self.region
        saver.save_figure(
            fig,
            diagnostic_product=diagnostic_product,
            extra_keys=keys,
            metadata={"description": description},
            extension=save_format,
            dpi=dpi,
            rebuild=rebuild,
        )

    def plot(self, plot_params: dict = None, save_format=SAVE_FORMAT, dpi: int = 300, rebuild: bool = True):
        """
        Draw and save all the figures: the index time series and the streamfunction sections.

        Args:
            plot_params (dict, optional): Options of the section: vmin, vmax, nlevels.
            save_format (str or list, optional): The format(s) of the figure. Default is SAVE_FORMAT.
            dpi (int, optional): The resolution of raster formats. Default is 300.
            rebuild (bool, optional): If True, overwrite existing files. Default is True.
        """
        self.plot_index(save_format=save_format, dpi=dpi, rebuild=rebuild)
        self.plot_streamfunction(**(plot_params or {}), save_format=save_format, dpi=dpi, rebuild=rebuild)
