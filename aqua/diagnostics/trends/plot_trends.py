"""Module for plotting 2D trend maps."""

from typing import Union

import matplotlib.pyplot as plt
import xarray as xr

from aqua.core.graphics import plot_single_map
from aqua.core.logger import log_configure
from aqua.core.util import get_realizations, to_list, unit_to_latex
from aqua.diagnostics.base import SAVE_FORMAT, OutputSaver, TitleBuilder

xr.set_options(keep_attrs=True)


class PlotTrends:
    """
    Class to plot 2D trend maps from a Dataset produced by :class:`Trends`.

    One figure per variable is produced through :func:`aqua.core.graphics.plot_single_map`
    and saved via :class:`aqua.diagnostics.base.OutputSaver`. Multi-variable handling
    follows the ocean-style pattern (``self.vars = list(self.data.data_vars)``) used
    by :class:`PlotStratification` and :class:`PlotHovmoller`.
    """

    def __init__(
        self,
        data: xr.Dataset,
        diagnostic_name: str = "trends",
        outputdir: str = ".",
        rebuild: bool = True,
        loglevel: str = "WARNING",
    ):
        """
        Initialize the PlotTrends class.

        Args:
            data (xr.Dataset): Trend coefficients (one variable per ``data_var``).
            diagnostic_name (str, optional): Diagnostic name used in output filenames.
                Defaults to ``'trends'``.
            outputdir (str, optional): Output directory. Defaults to ``'.'``.
            rebuild (bool, optional): Overwrite existing files. Defaults to True.
            loglevel (str, optional): Logging level. Defaults to ``'WARNING'``.
        """
        self.loglevel = loglevel
        self.logger = log_configure(log_level=loglevel, log_name="PlotTrends")

        if not isinstance(data, xr.Dataset):
            raise TypeError("PlotTrends expects an xarray.Dataset of trend coefficients.")

        self.data = data
        self.diagnostic_name = diagnostic_name
        self.outputdir = outputdir
        self.rebuild = rebuild

        self.vars = list(self.data.data_vars)
        self.logger.debug("Variables in data: %s", self.vars)

        self.get_data_info()

        self.outputsaver = OutputSaver(
            diagnostic=self.diagnostic_name,
            catalog=self.catalog,
            model=self.model,
            exp=self.exp,
            outputdir=outputdir,
            realization=self.realization,
            loglevel=self.loglevel,
        )

    def get_data_info(self):
        """Extract catalog/model/exp/region/realization and analysis period from data attributes."""
        first_var = self.data[self.vars[0]]
        self.catalog = first_var.attrs.get("AQUA_catalog")
        self.model = first_var.attrs.get("AQUA_model")
        self.exp = first_var.attrs.get("AQUA_exp")
        self.realization = get_realizations(first_var)
        self.region = self.data.attrs.get("AQUA_region")
        self.startdate = self.data.attrs.get("AQUA_startdate")
        self.enddate = self.data.attrs.get("AQUA_enddate")
        self.start_year = self.startdate[:4] if self.startdate else None
        self.end_year = self.enddate[:4] if self.enddate else None

    def set_title(self, var: str) -> str:
        """Build the figure title for a given variable."""
        long_name = self.data[var].attrs.get("long_name", var)
        return TitleBuilder(
            diagnostic="Trend",
            variable=long_name,
            regions=[self.region] if self.region is not None else None,
            catalog=self.catalog,
            model=self.model,
            exp=self.exp,
            startyear=self.start_year,
            endyear=self.end_year,
        ).generate()

    def set_description(self, var: str) -> str:
        """Build the figure description metadata for a given variable."""
        long_name = self.data[var].attrs.get("long_name", var)
        period = f" between {self.start_year} and {self.end_year}" if self.start_year and self.end_year else ""
        description = (
            f"Trend of {long_name} in the {self.region or 'global'} region "
            f"from {self.catalog} {self.model} {self.exp}{period}."
        )
        self.logger.info("Description: %s", description)
        return description

    def save_plot(
        self,
        fig,
        var: str,
        description: str,
        rebuild: bool,
        save_format: Union[str, list],
        dpi: int,
    ):
        """Save the trend figure of one variable through the OutputSaver."""
        short_name = self.data[var].attrs.get("short_name", var)
        extra_keys = {"var": short_name}
        if self.region is not None:
            extra_keys["region"] = self.region
        self.outputsaver.save_figure(
            fig,
            diagnostic_product="map_trend",
            rebuild=rebuild,
            extra_keys=extra_keys,
            metadata={"description": description},
            extension=save_format,
            dpi=dpi,
        )

    def plot_trend(
        self,
        var=None,
        vmin: float = None,
        vmax: float = None,
        cmap: str = "RdBu_r",
        sym: bool = None,
        rebuild: bool = None,
        save_format: Union[str, list] = SAVE_FORMAT,
        dpi: int = 300,
        show: bool = False,
    ):
        """
        Plot one trend map per variable.

        Args:
            var (str or list, optional): Variable(s) to plot. Defaults to all variables.
            vmin (float, optional): Colorbar minimum. If None, derived from data.
            vmax (float, optional): Colorbar maximum. If None, derived from data.
            cmap (str, optional): Colormap. Defaults to ``'RdBu_r'``.
            sym (bool, optional): Symmetric limits around zero. If None, True when no
                explicit ``vmin``/``vmax`` are given.
            rebuild (bool, optional): Overwrite existing files. Defaults to ``self.rebuild``.
            save_format (str or list, optional): Output format(s). Defaults to ``SAVE_FORMAT``.
            dpi (int, optional): Output DPI. Defaults to 300.
            show (bool, optional): If True, display each figure interactively
                (useful in notebooks). Defaults to False.
        """
        vars_to_plot = to_list(var) if var is not None else self.vars
        rebuild = self.rebuild if rebuild is None else rebuild

        for v in vars_to_plot:
            if v not in self.data.data_vars:
                self.logger.warning("Variable %s not in data, skipping", v)
                continue

            da = self.data[v]
            short_name = da.attrs.get("short_name", v)
            units = da.attrs.get("units", "")
            units_latex = unit_to_latex(units) if units else ""
            cbar_label = f"{short_name} ({units_latex})" if units_latex else short_name
            sym_value = (vmin is None and vmax is None) if sym is None else sym

            title = self.set_title(v)
            description = self.set_description(v)

            fig, _ = plot_single_map(
                data=da,
                title=title,
                cbar_label=cbar_label,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                sym=sym_value,
                return_fig=True,
                loglevel=self.loglevel,
            )

            self.save_plot(fig, v, description, rebuild, save_format, dpi)

            if show:
                plt.show()
            plt.close(fig)
