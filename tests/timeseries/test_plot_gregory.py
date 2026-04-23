"""Unit tests for PlotGregory behavior and branching."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from aqua.diagnostics.timeseries.plot_gregory import PlotGregory

pytestmark = [pytest.mark.diagnostics]


class _FakeTime:
    def __init__(self, idx):
        self.values = idx.values


class _FakeData:
    def __init__(self, values, start="2000-01-01", freq="MS", catalog="ci", model="IFS", exp="hist"):
        self._values = list(values)
        idx = pd.date_range(start, periods=len(values), freq=freq)
        self.time = _FakeTime(idx)
        self.AQUA_catalog = catalog
        self.AQUA_model = model
        self.AQUA_exp = exp

    def __len__(self):
        return len(self._values)

    def mean(self, *args, **kwargs):
        return np.mean(self._values)


def test_plot_raises_when_not_enough_data():
    # Avoid style loader touching local AQUA catalog setup.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("aqua.diagnostics.timeseries.plot_gregory.ConfigStyle", lambda style=None: None)
        p = PlotGregory(
            t2m_monthly_data=_FakeData([1.0]),  # len < 2
            net_toa_monthly_data=_FakeData([2.0]),
            t2m_annual_data=_FakeData([1.0], freq="YS"),
            net_toa_annual_data=_FakeData([2.0], freq="YS"),
        )
        with pytest.raises(ValueError, match="Not enough data to plot"):
            p.plot(freq=["monthly", "annual"])


def test_plot_monthly_only_path_calls_monthly_plotter(mocker):
    mocker.patch("aqua.diagnostics.timeseries.plot_gregory.ConfigStyle", side_effect=lambda style=None: None)
    p = PlotGregory(
        t2m_monthly_data=_FakeData([1.0, 2.0]),
        net_toa_monthly_data=_FakeData([2.0, 3.0]),
        t2m_annual_data=None,
        net_toa_annual_data=None,
    )
    m_plot_monthly = mocker.patch(
        "aqua.diagnostics.timeseries.plot_gregory.plot_gregory_monthly", side_effect=lambda **k: (k["fig"], k["ax"])
    )
    mocker.patch("aqua.diagnostics.timeseries.plot_gregory.plot_gregory_annual", side_effect=lambda **k: (k["fig"], k["ax"]))

    fig = p.plot(freq=["monthly"], data_labels=["d1"], ref_label="ref")

    assert fig is not None
    m_plot_monthly.assert_called_once()
    plt.close(fig)


def test_set_ref_label_and_description_with_and_without_reference():
    p_no_ref = PlotGregory(
        t2m_monthly_data=_FakeData([1.0, 2.0]),
        net_toa_monthly_data=_FakeData([2.0, 3.0]),
    )
    assert p_no_ref.set_ref_label() is None
    assert p_no_ref.set_description().endswith(".")

    p_ref = PlotGregory(
        t2m_monthly_data=_FakeData([1.0, 2.0]),
        net_toa_monthly_data=_FakeData([2.0, 3.0]),
        t2m_monthly_ref=_FakeData([1.0, 2.0], model="ERA5", exp="era5"),
        net_toa_monthly_ref=_FakeData([2.0, 3.0], model="CERES", exp="obs"),
    )
    ref_label = p_ref.set_ref_label()
    desc = p_ref.set_description()
    assert ref_label == "ERA5 era5 CERES obs"
    assert "using as a reference ERA5 era5" in desc
    assert "CERES obs (net TOA)." in desc


def test_check_data_length_mismatch_raises():
    with pytest.raises(ValueError, match="Length of .* data is not the same"):
        PlotGregory(
            t2m_monthly_data=[_FakeData([1.0, 2.0]), _FakeData([2.0, 3.0])],
            net_toa_monthly_data=[_FakeData([2.0, 3.0])],  # mismatch in list length
        )
