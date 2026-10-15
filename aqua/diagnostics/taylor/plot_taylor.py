"""Module for plotting Taylor diagrams from the statistics computed by the Taylor class."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D

from aqua.core.logger import log_configure
from aqua.core.util import time_to_string, to_list
from aqua.diagnostics.base import SAVE_FORMAT, OutputSaver, TitleBuilder

from .taylor import ANNUAL, SEASONS
from .taylor_diagram import plot_taylor_diagram

# Markers used to distinguish the entries of the marker_by key, in order of appearance
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "p", "*"]
# Keys that can be used to group the points of the diagram
GROUP_KEYS = ["var", "dataset", "period"]


class PlotTaylor:
    """
    Plot Taylor diagrams from the statistics produced by the Taylor class.

    Each statistics dataset holds one variable of one dataset compared with one reference, for
    one or more periods. Every (dataset, variable, period) combination becomes a point of the diagram,
    whose color and marker are chosen by two of these keys (``color_by`` and ``marker_by``).
    The diagram is normalised by the reference standard deviation by default, so that variables with
    different units can be shown together.

    Args:
        diagnostic_name (str): Name of the diagnostic, used in the filenames. Default is 'taylor'.
        save_format (str or list): Format(s) of the saved figures. Default is SAVE_FORMAT.
            If None or empty, the figures are not saved.
        dpi (int): Resolution of the PNG figures. Default is 300.
        outputdir (str): Output directory. Default is './'.
        rebuild (bool): If True, overwrite existing figures. Default is True.
        loglevel (str): Logging level. Default is 'WARNING'.
    """

    def __init__(
        self,
        diagnostic_name: str = "taylor",
        save_format: str | list = SAVE_FORMAT,
        dpi: int = 300,
        outputdir: str = "./",
        rebuild: bool = True,
        loglevel: str = "WARNING",
    ):
        self.diagnostic_name = diagnostic_name
        self.save_format = save_format
        self.dpi = dpi
        self.outputdir = outputdir
        self.rebuild = rebuild
        self.loglevel = loglevel
        self.logger = log_configure(log_level=loglevel, log_name="PlotTaylor")

    def plot_taylor(
        self,
        statistics: xr.Dataset | list[xr.Dataset],
        period: str | list[str] = ANNUAL,
        color_by: str = "var",
        marker_by: str = "dataset",
        normalize: bool = True,
        title: str = None,
        std_max: float = None,
        extend: bool = None,
        annotate: bool = False,
        save: bool = True,
    ):
        """
        Plot a Taylor diagram of one or more statistics datasets.

        Args:
            statistics (xr.Dataset or list of xr.Dataset): Statistics from Taylor.compute_statistics or Taylor.load.
            period (str or list of str): Period(s) to plot, among 'annual', 'DJF', 'MAM', 'JJA', 'SON'.
                'seasonal' is a shortcut for the four seasons. Default is 'annual'.
            color_by (str): Key distinguishing the colors: 'var', 'dataset' or 'period'. Default is 'var'.
            marker_by (str): Key distinguishing the markers: 'var', 'dataset' or 'period'. Default is 'dataset'.
            normalize (bool): If True, divide by the reference standard deviation. A non normalised diagram
                requires a single reference standard deviation, i.e. one variable and one period. Default is True.
            title (str, optional): Title of the figure. Default is built from the statistics metadata.
            std_max (float, optional): Maximum of the standard deviation axis. Default is automatic.
            extend (bool, optional): Show negative correlations. Default is automatic.
            annotate (bool): If True, write the label of the color key next to each point. Default is False.
            save (bool): If True, save the figure in the formats of save_format. Default is True.

        Returns:
            tuple: Matplotlib figure and polar axes.

        Raises:
            ValueError: If the keys are invalid, or if no point is available for the requested periods.
        """
        if color_by not in GROUP_KEYS or marker_by not in GROUP_KEYS:
            raise ValueError(f"color_by and marker_by must be in {GROUP_KEYS}, got {color_by} and {marker_by}")
        if color_by == marker_by:
            raise ValueError("color_by and marker_by must be different")

        statistics = to_list(statistics)
        periods = SEASONS if period == "seasonal" else to_list(period)

        points = self._collect_points(statistics=statistics, periods=periods)
        if not points:
            raise ValueError(f"No statistics available for the periods {periods}")

        if normalize:
            std = [p["std_norm"] for p in points]
            ref_std = 1.0
            std_label = "Normalised standard deviation"
        else:
            std_refs = np.array([p["std_ref"] for p in points])
            if not np.allclose(std_refs, std_refs[0]):
                raise ValueError("A non normalised diagram needs a single reference, i.e. one variable and one period")
            std = [p["std"] for p in points]
            ref_std = float(std_refs[0])
            units = points[0]["units"]
            std_label = f"Standard deviation [{units}]" if units else "Standard deviation"

        color_values = list(dict.fromkeys(p[color_by] for p in points))
        marker_values = list(dict.fromkeys(p[marker_by] for p in points))
        palette = plt.get_cmap("tab10" if len(color_values) <= 10 else "tab20").colors
        color_map = {value: palette[i % len(palette)] for i, value in enumerate(color_values)}
        marker_map = {value: MARKERS[i % len(MARKERS)] for i, value in enumerate(marker_values)}

        ref_label = self._reference_label(statistics)
        fig, ax = plot_taylor_diagram(
            std=std,
            corr=[p["corr"] for p in points],
            ref_std=ref_std,
            labels=[p[color_by] for p in points] if annotate else None,
            colors=[color_map[p[color_by]] for p in points],
            markers=[marker_map[p[marker_by]] for p in points],
            ref_label=ref_label,
            std_label=std_label,
            std_max=std_max,
            extend=extend,
            annotate=annotate,
            legend=False,
            title=title or self._title(statistics=statistics, periods=periods, points=points),
            loglevel=self.loglevel,
        )

        # Legend with a section for the colors and one for the markers
        handles = [Line2D([], [], marker="*", color="k", linestyle="", markersize=15, label=ref_label)]
        handles += [
            Line2D([], [], marker="o", color=color_map[v], markeredgecolor="k", linestyle="", markersize=9, label=v)
            for v in color_values
        ]
        if len(marker_values) > 1 or marker_by == "dataset":
            handles += [
                Line2D([], [], marker=marker_map[v], color="0.6", markeredgecolor="k", linestyle="", markersize=9, label=v)
                for v in marker_values
            ]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.05, 1.0), numpoints=1, frameon=False)

        if save and self.save_format:
            self._save_figure(fig=fig, statistics=statistics, periods=periods, points=points, normalize=normalize)

        return fig, ax

    def _collect_points(self, statistics: list[xr.Dataset], periods: list[str]) -> list[dict]:
        """
        Build one point per statistics dataset and period.

        Args:
            statistics (list of xr.Dataset): The statistics datasets.
            periods (list of str): The periods to plot.

        Returns:
            list of dict: Each with the values and the grouping keys of a point.
        """
        points = []
        for stats in statistics:
            available = [str(p) for p in stats["period"].values]
            for period in periods:
                if period not in available:
                    self.logger.warning("Period %s not available for %s, skipping it", period, self._var_label(stats))
                    continue
                values = stats.sel(period=period)
                points.append(
                    {
                        "var": self._var_label(stats),
                        "dataset": self._dataset_label(stats),
                        "period": period,
                        "std": float(values["std"]),
                        "std_ref": float(values["std_ref"]),
                        "std_norm": float(values["std_norm"]),
                        "corr": float(values["corr"]),
                        "units": values["std"].attrs.get("units"),
                    }
                )
        return points

    @staticmethod
    def _var_label(stats: xr.Dataset) -> str:
        """Label of the variable, with the pressure level if any."""
        label = stats.attrs.get("short_name", "")
        plev = stats.attrs.get("AQUA_plev")
        if plev is not None:
            label += f" {float(plev) / 100:g} hPa"
        return label

    @staticmethod
    def _dataset_label(stats: xr.Dataset) -> str:
        """Label of the dataset."""
        return f"{stats.attrs.get('AQUA_model')} {stats.attrs.get('AQUA_exp')}"

    @staticmethod
    def _reference_label(statistics: list[xr.Dataset]) -> str:
        """Label of the reference(s), with the period of the first one."""
        refs = list(dict.fromkeys(f"{s.attrs.get('AQUA_model_ref')} {s.attrs.get('AQUA_exp_ref')}" for s in statistics))
        startdate = statistics[0].attrs.get("AQUA_startdate_ref")
        enddate = statistics[0].attrs.get("AQUA_enddate_ref")
        if len(refs) == 1 and startdate and enddate:
            return f"{refs[0]} ({time_to_string(startdate, format='%Y')}-{time_to_string(enddate, format='%Y')})"
        return ", ".join(refs)

    def _title(self, statistics: list[xr.Dataset], periods: list[str], points: list[dict]) -> str:
        """
        Build the default title.

        Args:
            statistics (list of xr.Dataset): The statistics datasets.
            periods (list of str): The plotted periods.
            points (list of dict): The plotted points.

        Returns:
            str: The title.
        """
        variables = list(dict.fromkeys(p["var"] for p in points))
        variable = statistics[0].attrs.get("long_name", variables[0]) if len(variables) == 1 else None
        if variable and statistics[0].attrs.get("AQUA_plev") is not None:
            variable = f"{variable} at {self._var_label(statistics[0]).split(' ', 1)[1]}"

        datasets = self._unique_datasets(statistics)
        startyears = list(dict.fromkeys(time_to_string(s.attrs.get("AQUA_startdate"), format="%Y") for s in statistics))
        endyears = list(dict.fromkeys(time_to_string(s.attrs.get("AQUA_enddate"), format="%Y") for s in statistics))

        return TitleBuilder(
            diagnostic="Seasonal Taylor diagram" if periods == SEASONS else "Taylor diagram",
            variable=variable,
            regions=statistics[0].attrs.get("AQUA_region"),
            model=[d["model"] for d in datasets],
            exp=[d["exp"] for d in datasets],
            startyear=startyears[0] if len(startyears) == 1 else None,
            endyear=endyears[0] if len(endyears) == 1 else None,
            ref_model=list(dict.fromkeys(s.attrs.get("AQUA_model_ref") for s in statistics)),
            ref_exp=list(dict.fromkeys(s.attrs.get("AQUA_exp_ref") for s in statistics)),
            timeseason=None if periods in ([ANNUAL], SEASONS) else ", ".join(periods),
            loglevel=self.loglevel,
        ).generate(max_chars=60)

    @staticmethod
    def _unique_datasets(statistics: list[xr.Dataset]) -> list[dict]:
        """Unique datasets of the statistics, in order of appearance."""
        datasets = {}
        for s in statistics:
            key = (
                s.attrs.get("AQUA_catalog"),
                s.attrs.get("AQUA_model"),
                s.attrs.get("AQUA_exp"),
                s.attrs.get("AQUA_realization"),
            )
            datasets.setdefault(key, {"catalog": key[0], "model": key[1], "exp": key[2], "realization": key[3]})
        return list(datasets.values())

    def _save_figure(
        self, fig: plt.Figure, statistics: list[xr.Dataset], periods: list[str], points: list[dict], normalize: bool = True
    ):
        """
        Save the figure with the OutputSaver.

        Args:
            fig (plt.Figure): The figure to save.
            statistics (list of xr.Dataset): The plotted statistics datasets.
            periods (list of str): The plotted periods.
            points (list of dict): The plotted points.
            normalize (bool): If the diagram is normalised by the reference standard deviation.
        """
        datasets = self._unique_datasets(statistics)
        references = list(
            dict.fromkeys(
                (s.attrs.get("AQUA_catalog_ref"), s.attrs.get("AQUA_model_ref"), s.attrs.get("AQUA_exp_ref"))
                for s in statistics
            )
        )
        outputsaver = OutputSaver(
            diagnostic=self.diagnostic_name,
            catalog=[d["catalog"] for d in datasets],
            model=[d["model"] for d in datasets],
            exp=[d["exp"] for d in datasets],
            # A single realization must be a string, a list would end up as such in the filename
            realization=OutputSaver.unpack_list([d["realization"] for d in datasets]),
            catalog_ref=[r[0] for r in references],
            model_ref=[r[1] for r in references],
            exp_ref=[r[2] for r in references],
            outputdir=self.outputdir,
            loglevel=self.loglevel,
        )

        variables = list(dict.fromkeys(stats.attrs.get("short_name") for stats in statistics))
        plevs = list(dict.fromkeys(stats.attrs.get("AQUA_plev") for stats in statistics))
        extra_keys = {"var": "_".join(variables)}
        if len(plevs) == 1 and plevs[0] is not None:
            extra_keys["plev"] = plevs[0]
        extra_keys["period"] = "seasonal" if periods == SEASONS else "_".join(periods)
        region = statistics[0].attrs.get("AQUA_region")
        if region is not None:
            extra_keys["region"] = region

        description = (
            f"Taylor diagram of {', '.join(dict.fromkeys(p['var'] for p in points))} "
            f"for {', '.join(dict.fromkeys(p['dataset'] for p in points))} "
            f"relative to {self._reference_label(statistics)}, for the {extra_keys['period']} climatology. "
            f"The radial distance is the {'normalised ' if normalize else ''}standard deviation of the spatial pattern, "
            "the angle is its correlation with the reference and the distance from the reference point is the "
            f"{'normalised ' if normalize else ''}centred root mean square difference (dashed green contours)."
        )
        outputsaver.save_figure(
            fig,
            diagnostic_product="taylor_diagram",
            extra_keys=extra_keys,
            metadata={"Description": description},
            extension=self.save_format,
            rebuild=self.rebuild,
            dpi=self.dpi,
        )
