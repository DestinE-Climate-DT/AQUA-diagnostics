"""Tests for the Taylor diagram graphics function."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from aqua.diagnostics.taylor import plot_taylor_diagram
from tests.shared_constants import LOGLEVEL

pytestmark = [pytest.mark.diagnostics]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


class TestPlotTaylorDiagram:
    """Tests of plot_taylor_diagram."""

    def test_basic(self):
        """A quadrant diagram with one line per point plus the reference."""
        fig, ax = plot_taylor_diagram(
            std=[0.8, 1.2], corr=[0.9, 0.6], labels=["a", "b"], markers=["o", "s"], title="Test", loglevel=LOGLEVEL
        )
        assert ax.name == "polar"
        assert ax.get_thetamax() == pytest.approx(90)
        assert ax.get_title() == "Test"
        labels = [line.get_label() for line in ax.get_lines()]
        assert {"a", "b", "Reference"} <= set(labels)
        point = next(line for line in ax.get_lines() if line.get_label() == "a")
        assert point.get_xdata()[0] == pytest.approx(np.arccos(0.9))
        assert point.get_ydata()[0] == pytest.approx(0.8)
        assert ax.get_legend() is not None

    def test_extend_auto(self):
        """A negative correlation extends the diagram to a half circle."""
        _, ax = plot_taylor_diagram(std=[1.0], corr=[-0.3])
        assert ax.get_thetamax() == pytest.approx(180)

    def test_existing_axes_and_options(self):
        """Drawing on given axes, without legend nor contours, with a non normalised reference."""
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="polar")
        fig_out, ax_out = plot_taylor_diagram(
            std=[3.0], corr=[0.7], ref_std=2.5, std_max=5, show_crmse=False, legend=False, fig=fig, ax=ax, annotate=True
        )
        assert fig_out is fig and ax_out is ax
        assert ax.get_ylim()[1] == pytest.approx(5)
        assert ax.get_legend() is None

    def test_non_finite_points_are_skipped(self):
        """Non finite points are not drawn."""
        _, ax = plot_taylor_diagram(std=[1.0, np.nan], corr=[0.5, 0.5], labels=["ok", "nan"])
        labels = [line.get_label() for line in ax.get_lines()]
        assert "ok" in labels and "nan" not in labels

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"std": [1, 2], "corr": [0.5]}, "same length"),
            ({"std": [1], "corr": [0.5], "labels": ["a", "b"]}, "one entry per point"),
            ({"std": [1], "corr": [1.5]}, r"\[-1, 1\]"),
            ({"std": [1], "corr": [0.5], "ref_std": 0}, "positive"),
        ],
    )
    def test_invalid_inputs(self, kwargs, match):
        """Inconsistent inputs raise a ValueError."""
        with pytest.raises(ValueError, match=match):
            plot_taylor_diagram(**kwargs)

    def test_non_polar_axes(self):
        """Cartesian axes are refused."""
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="polar"):
            plot_taylor_diagram(std=[1], corr=[0.5], fig=fig, ax=ax)
