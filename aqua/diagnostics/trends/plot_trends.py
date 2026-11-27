"""Plot 2D trend maps"""

from typing import Union

import matplotlib.pyplot as plt
import xarray as xr

from aqua.core.graphics import plot_single_map
from aqua.core.util import time_to_string, to_list, unit_to_latex
from aqua.diagnostics.base import SAVE_FORMAT, PlotBase


class PlotTrends(PlotBase):
    """
    Class to plot 2D trend maps from a Dataset produced by :class:`Trends`.

    One figure per variable is produced through :func:`aqua.core.graphics.plot_single_map`
    and saved via :class:`aqua.diagnostics.base.PlotBase`. Multi-variable handling
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
        super().__init__(diagnostic_name=diagnostic_name, outputdir=outputdir, rebuild=rebuild, loglevel=loglevel)

        if not isinstance(data, xr.Dataset):
            raise TypeError("PlotTrends expects an xarray.Dataset of trend coefficients.")

        self.data = data
        self.vars = list(self.data.data_vars)
        self.logger.debug("Variables in data: %s", self.vars)

        self.get_data_info(self.data)

    def set_title(self, var: str) -> str:
        """Build the figure title for a given variable."""
        long_name = self.data[var].attrs.get("long_name", var)
        startdate, enddate = self.startdates[0], self.enddates[0]
        return super().set_title(
            diagnostic="Trend",
            variable=long_name,
            catalog=self.catalogs,
            startyear=startdate[:4] if startdate else None,
            endyear=enddate[:4] if enddate else None,
        )

    def set_description(self, var: str) -> str:
        """Build the figure description metadata for a given variable."""
        long_name = self.data[var].attrs.get("long_name", var)
        startdate, enddate = self.startdates[0], self.enddates[0]
        period = (
            f" between {time_to_string(startdate, format='%Y-%m')} and {time_to_string(enddate, format='%Y-%m')}"
            if startdate is not None and enddate is not None
            else ""
        )
        description = (
            f"Trend of {long_name} in the {self.region or 'global'} region "
            f"from {self.catalogs[0]} {self.models[0]} {self.exps[0]}{period}."
        )
        self.logger.info("Description: %s", description)
        return description

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

        Raises:
            ValueError: When a variable is not a horizontal field with latitude and longitude coordinates.
        """
        vars_to_plot = to_list(var) if var is not None else self.vars

        for v in vars_to_plot:
            if v not in self.data.data_vars:
                self.logger.warning("Variable %s not in data, skipping", v)
                continue

            da = self.data[v].copy(deep=False)
            if "lat" not in da.coords or "lon" not in da.coords:
                raise ValueError(f"Trend map for '{v}' requires latitude and longitude coordinates.")

            horizontal_dims = set(da.lat.dims) | set(da.lon.dims)
            if not horizontal_dims or set(da.dims) != horizontal_dims:
                raise ValueError(f"Trend map for '{v}' requires a horizontal field.")

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

            extra_keys = {"var": short_name}
            if self.region is not None:
                extra_keys["region"] = self.region
            self.save_plot(
                fig,
                diagnostic_product="map_trend",
                extra_keys=extra_keys,
                description=description,
                rebuild=rebuild,
                save_format=save_format,
                dpi=dpi,
                close=not show,
            )
            if show:
                plt.show()
                plt.close(fig)
