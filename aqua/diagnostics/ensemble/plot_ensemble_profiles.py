import calendar

import matplotlib.pyplot as plt
import xarray as xr

from aqua.core.graphics import ConfigStyle
from aqua.diagnostics.base import SAVE_FORMAT, TitleBuilder

from .base import BaseMixin

xr.set_options(keep_attrs=True)


class PlotEnsembleProfiles(BaseMixin):
    """Class to plot 1D ensemble data such as latitude profiles and seasonal cycles."""

    def __init__(
        self,
        diagnostic_product: str = "EnsembleProfiles",
        catalog_list: list[str] = None,
        model_list: list[str] = None,
        exp_list: list[str] = None,
        source_list: list[str] = None,
        ref_catalog: str = None,
        ref_model: str = None,
        ref_exp: str = None,
        region: str = None,
        outputdir="./",
        loglevel: str = "WARNING",
    ):
        """
        Initialize the PlotEnsembleProfiles class.

        Args:
            diagnostic_product (str, optional): Name of the diagnostic product. Defaults to "EnsembleProfiles".
            catalog_list (list[str], optional): List of catalog names. Defaults to None.
            model_list (list[str], optional): List of model names. Defaults to None.
            exp_list (list[str], optional): List of experiment names. Defaults to None.
            source_list (list[str], optional): List of source names. Defaults to None.
            ref_catalog (str, optional): Reference catalog name. Defaults to None.
            ref_model (str, optional): Reference model name. Defaults to None.
            ref_exp (str, optional): Reference experiment name. Defaults to None.
            region (str, optional): Name of the region. Defaults to None.
            outputdir (str, optional): Directory to save output plots. Defaults to "./".
            loglevel (str, optional): Logging level. Defaults to "WARNING".
        """
        super().__init__(
            loglevel=loglevel,
            diagnostic_product=diagnostic_product,
            catalog_list=catalog_list,
            model_list=model_list,
            exp_list=exp_list,
            source_list=source_list,
            ref_catalog=ref_catalog,
            ref_model=ref_model,
            ref_exp=ref_exp,
            region=region,
            outputdir=outputdir,
        )

    def _draw(self, ax, coord, mean, std, members=None, ref=None):
        """
        Draw the members, the ensemble mean with its +/- 2 STD band and the reference on one axis.

        Args:
            ax (matplotlib.axes.Axes): Axis to draw on.
            coord (str): Name of the coordinate of the x-axis.
            mean (xr.DataArray): Ensemble mean.
            std (xr.DataArray): Ensemble standard deviation.
            members (xr.DataArray, optional): Ensemble members along `ensemble`. Defaults to None.
            ref (xr.DataArray, optional): Reference data. Defaults to None.
        """
        if members is not None:
            for member in members.transpose("ensemble", ...):
                ax.plot(member[coord], member, color="grey", linewidth=0.6, alpha=0.7)

        ax.fill_between(mean[coord], mean - 2 * std, mean + 2 * std, alpha=0.3, label="Ensemble mean ± 2 STD")
        ax.plot(mean[coord], mean, label=self.model)

        if ref is not None:
            ax.plot(ref[coord], ref, color="black", label=self.ref_model or "Reference")

        if coord == "month":
            ax.set_xticks(range(1, 13), [calendar.month_abbr[month] for month in range(1, 13)])
            ax.set_xlabel("Month")
        else:
            # Regional profiles are NaN outside of the region
            valid = mean.dropna(coord)[coord]
            ax.set_xlim(float(valid.min()), float(valid.max()))
            ax.set_xlabel("Latitude" if coord == "lat" else coord)

    def plot(
        self,
        var: str = None,
        mean=None,
        std=None,
        members=None,
        ref=None,
        coord: str = "lat",
        panels: dict = None,
        title=None,
        ylabel=None,
        data_name=None,
        description=None,
        save_format=SAVE_FORMAT,
        dpi=300,
        figure_size=(10, 5),
    ):
        """
        Plot one 1D ensemble, or one panel per selection of the ensemble.

        Args:
            var (str): Variable name.
            mean (xr.DataArray): Ensemble mean.
            std (xr.DataArray): Ensemble standard deviation.
            members (xr.DataArray, optional): Ensemble members along `ensemble`. Defaults to None.
            ref (xr.DataArray or dict, optional): Reference data. With `panels`, a dictionary with the
                same keys as `panels`. Defaults to None.
            coord (str, optional): Name of the coordinate of the x-axis. Defaults to "lat".
            panels (dict, optional): Panel title as key and the `.sel()` arguments of that panel as value.
                If None, a single plot is produced. Defaults to None.
            title (str, optional): Title of the figure. Auto-generated if None.
            ylabel (str, optional): Label of the y-axis. Defaults to the variable name and its units.
            data_name (str, optional): File naming label to distinguish saved plots.
            description (str, optional): Description string for saving the plot. Defaults to None.
            save_format (str or list, optional): Format(s) to save figures in. Defaults to SAVE_FORMAT.
            dpi (int, optional): Resolution for saved figures. Defaults to 300.
            figure_size (tuple, optional): Figure size. Defaults to (10, 5).

        Returns:
            matplotlib.figure.Figure or None: The figure, or None if there is no data to plot.
        """
        self.logger.info("Plotting the ensemble profiles")
        if mean is None or std is None:
            self.logger.warning("No data given to plot the ensemble profiles of %s", var)
            return None

        mean, std = mean.load(), std.load()
        if bool(mean.isnull().all()):
            self.logger.warning("The profile of %s is empty. Skipping %s", var, data_name)
            return None

        if title is None:
            title = TitleBuilder(diagnostic="Ensemble profile", variable=var, regions=self.region, model=self.model).generate()
        if ylabel is None:
            units = mean.attrs.get("units")
            ylabel = f"{var} [{units}]" if units else var

        ConfigStyle(style=None, loglevel=self.loglevel)

        if panels is None:
            fig, ax = plt.subplots(figsize=figure_size)
            self._draw(ax, coord, mean, std, members, ref)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
        else:
            fig, axes = plt.subplots(2, 2, figsize=figure_size, sharex=True, sharey=True, constrained_layout=True)
            for ax, (name, selection) in zip(axes.flat, panels.items()):
                self._draw(
                    ax,
                    coord,
                    mean.sel(selection),
                    std.sel(selection),
                    members.sel(selection) if members is not None else None,
                    (ref or {}).get(name),
                )
                ax.set_title(name)
            for ax in axes[:, 0]:
                ax.set_ylabel(ylabel)
            axes.flat[0].legend()
            fig.suptitle(title)

        self.logger.debug("Saving ensemble profiles")
        self.save_figure(var=var, fig=fig, data_name=data_name, description=description, format=save_format, dpi=dpi)
        return fig
