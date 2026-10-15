"""
Taylor diagram graphics function.

This module has no dependency on the rest of the Taylor diagnostic and it follows the
conventions of ``aqua.core.graphics``, so that it can be moved there as it is, only
importing ConfigStyle from ``.styles`` instead.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import ScaledTranslation

from aqua.core.graphics import ConfigStyle
from aqua.core.logger import log_configure
from aqua.core.util import to_list

# Correlation values marked on the angular axis, the negative ones are used for the extended diagram
CORRELATION_TICKS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]


def plot_taylor_diagram(
    std: list[float] | np.ndarray,
    corr: list[float] | np.ndarray,
    ref_std: float = 1.0,
    labels: list[str] | None = None,
    colors: list | None = None,
    markers: list[str] | None = None,
    ref_label: str = "Reference",
    std_label: str = "Standard deviation",
    corr_label: str = "Correlation",
    std_max: float | None = None,
    extend: bool | None = None,
    show_crmse: bool = True,
    crmse_levels: list[float] | None = None,
    annotate: bool = False,
    marker_size: float = 9,
    legend: bool = True,
    title: str | None = None,
    style: str | None = None,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple = (8, 8),
    fontsize: int = 12,
    loglevel: str = "WARNING",
):
    """
    Plot a Taylor diagram (Taylor, 2001, JGR, doi:10.1029/2000JD900719).

    Each point is placed at a radial distance equal to its standard deviation and at an angle
    equal to the arccosine of its correlation with the reference. The distance of a point from
    the reference point is then the centred root mean square difference (CRMSD), shown as contours.
    The function is agnostic to normalisation: pass the standard deviations divided by the reference
    one together with ``ref_std=1`` to obtain a normalised diagram, which is needed to compare
    different variables on the same plot.

    Args:
        std (list or np.ndarray): Standard deviation of each point.
        corr (list or np.ndarray): Correlation with the reference of each point. Values must be in [-1, 1].
        ref_std (float): Standard deviation of the reference. Default is 1, the normalised case.
        labels (list of str, optional): Label of each point, used in the legend and for the annotations.
        colors (list, optional): Matplotlib color of each point. Default is the matplotlib color cycle.
        markers (list of str, optional): Matplotlib marker of each point. Default is 'o'.
        ref_label (str): Label of the reference point. Default is 'Reference'.
        std_label (str): Label of the standard deviation axis. Default is 'Standard deviation'.
        corr_label (str): Label of the correlation axis. Default is 'Correlation'.
        std_max (float, optional): Maximum of the standard deviation axis. Default is
            1.25 times the largest standard deviation, reference included.
        extend (bool, optional): If True, the diagram covers negative correlations as well.
            Default is None, which extends it only if a negative correlation is present.
        show_crmse (bool): If True, draw the contours of the centred RMS difference. Default is True.
        crmse_levels (list of float, optional): Levels of the CRMSD contours. Default is automatic.
        annotate (bool): If True, write the label next to each point. Default is False.
        marker_size (float): Size of the markers. Default is 9.
        legend (bool): If True and labels are provided, add a legend. Default is True.
        title (str, optional): Title of the figure.
        style (str, optional): AQUA style to apply. Default is the one in the configuration file.
        fig (plt.Figure, optional): Figure to draw on. A new one is created if not provided.
        ax (plt.Axes, optional): Polar axes to draw on. A new one is created if not provided.
        figsize (tuple): Size of the figure if a new one is created. Default is (8, 8).
        fontsize (int): Base font size. Default is 12.
        loglevel (str): Logging level. Default is 'WARNING'.

    Returns:
        tuple: Matplotlib figure and polar axes.

    Raises:
        ValueError: If the inputs have inconsistent lengths, or a correlation is outside [-1, 1].
    """
    logger = log_configure(loglevel, "plot_taylor_diagram")
    ConfigStyle(style=style, loglevel=loglevel)

    std = np.asarray(to_list(std), dtype=float)
    corr = np.asarray(to_list(corr), dtype=float)
    npoints = len(std)

    if len(corr) != npoints:
        raise ValueError(f"std and corr must have the same length, got {npoints} and {len(corr)}")
    for name, values in {"labels": labels, "colors": colors, "markers": markers}.items():
        if values is not None and len(values) != npoints:
            raise ValueError(f"{name} must have one entry per point, got {len(values)} for {npoints} points")
    if np.any(np.abs(corr[np.isfinite(corr)]) > 1 + 1e-6):
        raise ValueError(f"Correlations must be in [-1, 1], got {corr}")
    if ref_std <= 0:
        raise ValueError(f"ref_std must be positive, got {ref_std}")

    # Rounding errors can push a perfect correlation slightly above 1
    corr = np.clip(corr, -1, 1)

    valid = np.isfinite(std) & np.isfinite(corr)
    if not np.all(valid):
        logger.warning("Skipping %d points with non finite standard deviation or correlation", np.sum(~valid))

    if extend is None:
        extend = bool(np.any(corr[valid] < 0))
    thetamax = np.pi if extend else np.pi / 2

    if std_max is None:
        std_max = 1.25 * max(np.max(std[valid], initial=0), ref_std)

    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot(1, 1, 1, projection="polar")
    elif ax.name != "polar":
        raise ValueError("The Taylor diagram must be drawn on polar axes, create them with projection='polar'")

    ax.set_thetamin(0)
    ax.set_thetamax(np.degrees(thetamax))
    ax.set_ylim(0, std_max)

    # Angular axis: the correlation
    corr_ticks = np.array(CORRELATION_TICKS)
    if extend:
        corr_ticks = np.concatenate([-corr_ticks[::-1], corr_ticks[1:]])
    # Adding 0.0 turns the negative zero into a plain zero
    tick_labels = [f"{c + 0.0:g}" for c in corr_ticks]
    ax.set_thetagrids(np.degrees(np.arccos(corr_ticks)), labels=tick_labels, fontsize=fontsize - 2)

    # Radial axis: the standard deviation
    ax.tick_params(axis="y", labelsize=fontsize - 2)
    ax.grid(True, linestyle=":", linewidth=0.8, color="0.6")

    # Axis labels: the correlation along the arc, the standard deviation below the horizontal edge
    label_angle = thetamax / 2
    ax.text(
        label_angle,
        1.18 * std_max,
        corr_label,
        rotation=np.degrees(label_angle) - 90,
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    ax.text(
        0,
        0.5 * std_max,
        std_label,
        ha="center",
        va="top",
        fontsize=fontsize,
        transform=ax.transData + ScaledTranslation(0, -0.35, fig.dpi_scale_trans),
    )

    # Contours of the centred RMS difference with respect to the reference point
    if show_crmse:
        rs, ts = np.meshgrid(np.linspace(0, std_max, 200), np.linspace(0, thetamax, 200))
        crmse = np.sqrt(ref_std**2 + rs**2 - 2 * ref_std * rs * np.cos(ts))
        if crmse_levels is None:
            crmse_levels = _crmse_levels(ref_std=ref_std, std_max=std_max)
        contours = ax.contour(
            ts, rs, crmse, levels=crmse_levels, colors="tab:green", linestyles="--", linewidths=0.9, alpha=0.7
        )
        ax.clabel(contours, fmt="%.2g", fontsize=fontsize - 3, inline=True)

    # Reference: a point on the horizontal axis and the arc of its standard deviation
    theta_arc = np.linspace(0, thetamax, 200)
    ax.plot(theta_arc, np.full_like(theta_arc, ref_std), color="k", linestyle="-", linewidth=0.9)
    ax.plot(
        0, ref_std, marker="*", color="k", markersize=marker_size + 6, linestyle="", label=ref_label, clip_on=False, zorder=10
    )

    # Points
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    for i in np.flatnonzero(valid):
        color = colors[i] if colors is not None else default_colors[i % len(default_colors)]
        marker = markers[i] if markers is not None else "o"
        label = labels[i] if labels is not None else None
        theta = np.arccos(corr[i])
        ax.plot(
            theta,
            std[i],
            marker=marker,
            color=color,
            markersize=marker_size,
            markeredgecolor="k",
            markeredgewidth=0.6,
            linestyle="",
            label=label,
            clip_on=False,
            zorder=11,
        )
        if annotate and label is not None:
            ax.annotate(label, xy=(theta, std[i]), xytext=(5, 5), textcoords="offset points", fontsize=fontsize - 3)

        if std[i] > std_max:
            logger.warning("Point %s has standard deviation %.3g beyond std_max %.3g", label or i, std[i], std_max)

    if legend and labels is not None:
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), numpoints=1, fontsize=fontsize - 2, frameon=False)

    if title:
        ax.set_title(title, fontsize=fontsize + 2, pad=30)

    return fig, ax


def _crmse_levels(ref_std: float, std_max: float, nlevels: int = 5) -> np.ndarray:
    """
    Choose round levels for the CRMSD contours.

    Args:
        ref_std (float): Standard deviation of the reference.
        std_max (float): Maximum of the standard deviation axis.
        nlevels (int): Approximate number of levels. Default is 5.

    Returns:
        np.ndarray: The contour levels.
    """
    # The largest CRMSD visible in the first quadrant is at the corner opposite to the reference
    crmse_max = np.hypot(ref_std, std_max)
    step = crmse_max / nlevels
    magnitude = 10 ** np.floor(np.log10(step))
    step = min((s * magnitude for s in (1, 2, 2.5, 5, 10) if s * magnitude >= step), default=magnitude)
    return np.arange(step, crmse_max, step)
