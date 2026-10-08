"""Module for plotting trends maps for atmospheric, surface, ocean depth products."""

from typing import Union

import matplotlib.pyplot as plt
import xarray as xr

from aqua.core.graphics import plot_single_map
from aqua.core.logger import log_configure
from aqua.core.util import get_realizations, time_to_string, to_list, unit_to_latex
from aqua.diagnostics.base import SAVE_FORMAT, OutputSaver, TitleBuilder
from aqua.diagnostics.base.defaults import DEFAULT_OCEAN_VERT_COORD
from aqua.diagnostics.ocean_trends import PlotTrends as PlotOceanTrends

xr.set_options(keep_attrs=True)


class PlotTrends:
    """
    Plot 2D trend maps and ocean depth products from a Dataset produced by :class:`Trends`.

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
        vert_coord: str = DEFAULT_OCEAN_VERT_COORD,
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
            vert_coord (str, optional): Ocean depth coordinate for multilevel/zonal plots.
                Defaults to ``'depth'``. Depth values are in metres.
        """
        self.loglevel = loglevel
        self.logger = log_configure(log_level=loglevel, log_name="PlotTrends")

        if not isinstance(data, xr.Dataset):
            raise TypeError("PlotTrends expects an xarray.Dataset of trend coefficients.")
        if not data.data_vars:
            raise ValueError("PlotTrends requires at least one trend variable.")

        self.data = data
        self.diagnostic_name = diagnostic_name
        self.outputdir = outputdir
        self.rebuild = rebuild
        self.vert_coord = vert_coord

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
        attrs = {**self.data.attrs, **first_var.attrs}
        self.catalog = attrs.get("AQUA_catalog")
        self.model = attrs.get("AQUA_model")
        self.exp = attrs.get("AQUA_exp")
        self.realization = get_realizations(first_var if "AQUA_realization" in first_var.attrs else self.data)
        self.region = self.data.attrs.get("AQUA_region", "global")
        self.startdate = self.data.attrs.get("AQUA_startdate")
        self.enddate = self.data.attrs.get("AQUA_enddate")
        self.start_year = time_to_string(self.startdate, format="%Y") if self.startdate is not None else None
        self.end_year = time_to_string(self.enddate, format="%Y") if self.enddate is not None else None

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
        period = (
            f" between {time_to_string(self.startdate, format='%Y-%m')} and {time_to_string(self.enddate, format='%Y-%m')}"
            if self.startdate is not None and self.enddate is not None
            else ""
        )
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
        if "AQUA_dim_mean" in self.data.attrs:
            extra_keys["dim_mean"] = self.data.attrs["AQUA_dim_mean"]
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
        extent: list = None,
        proj=None,
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
            extent (list, optional): Map bounds as [west, east, south, north].
            proj (cartopy.crs.Projection, optional): Map projection. Uses the core default when omitted.
        """
        vars_to_plot = to_list(var) if var is not None else self.vars
        rebuild = self.rebuild if rebuild is None else rebuild

        for v in vars_to_plot:
            if v not in self.data.data_vars:
                self.logger.warning("Variable %s not in data, skipping", v)
                continue

            da = self.data[v]
            if self.vert_coord in da.dims or da.ndim > 2:
                raise ValueError("Select a depth level before plot_trend(), or use plot_multilevel().")
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
                extent=extent,
                **({"proj": proj} if proj is not None else {}),
                return_fig=True,
                loglevel=self.loglevel,
            )

            self.save_plot(fig, v, description, rebuild, save_format, dpi)

            if show:
                plt.show()
            plt.close(fig)

    def _ocean_plotter(self):
        """Reuse the ocean presentation layer without duplicating its plotting helpers."""
        if self.vert_coord not in self.data.dims:
            raise ValueError(f"Ocean trend plots require the vertical dimension '{self.vert_coord}'.")
        return PlotOceanTrends(
            data=self.data,
            diagnostic_name=self.diagnostic_name,
            vert_coord=self.vert_coord,
            outputdir=self.outputdir,
            rebuild=self.rebuild,
            loglevel=self.loglevel,
        )

    def plot_multilevel(self, levels=None, rebuild=None, cbar_limits=None, sym=False, save_format=SAVE_FORMAT, dpi=300):
        """Plot ocean trends at selected depths using the existing ocean layout.

        Args:
            levels (list, optional): Depths in metres. Defaults to the ocean plotter's levels.
                Zero selects the native level nearest the surface.
            rebuild (bool, optional): Overwrite existing files. Defaults to the instance setting.
            cbar_limits (dict, optional): Per-variable limits: ``{var: {'vmin': low, 'vmax': high}}``.
            sym (bool, optional): Use symmetric colour limits. Defaults to False.
            save_format (str or list, optional): Output format(s). Defaults to SAVE_FORMAT.
            dpi (int, optional): Figure resolution. Defaults to 300.
        """
        self._ocean_plotter().plot_multilevel(
            levels=levels,
            rebuild=self.rebuild if rebuild is None else rebuild,
            cbar_limits=cbar_limits,
            sym=sym,
            save_format=save_format,
            dpi=dpi,
        )

    def plot_zonal(self, rebuild=None, save_format=SAVE_FORMAT, dpi=300):
        """Plot a prepared latitude-depth trend section; no spatial averaging is performed.

        Pass either the result of ``compute_trend(dim_mean='lon')`` (fit after a weighted
        mean), or ``coefficients.mean('lon')`` (mean of pointwise fits, the legacy ocean
        product). These operations can differ when valid samples vary over time.

        Args:
            rebuild (bool, optional): Overwrite existing files. Defaults to the instance setting.
            save_format (str or list, optional): Output format(s). Defaults to SAVE_FORMAT.
            dpi (int, optional): Figure resolution. Defaults to 300.
        """
        self._ocean_plotter().plot_zonal(
            rebuild=self.rebuild if rebuild is None else rebuild,
            save_format=save_format,
            dpi=dpi,
        )
