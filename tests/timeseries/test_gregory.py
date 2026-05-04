import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aqua.diagnostics.timeseries import Gregory, PlotGregory
from tests.shared_constants import APPROX_REL, DPI, LOGLEVEL

# pytest approximation, to bear with different machines
approx_rel = APPROX_REL
loglevel = LOGLEVEL


@pytest.mark.diagnostics
class TestGregory:
    """Test the Gregory class."""

    def setup_method(self):
        """Initialize the variables before each test."""
        self.catalog = "ci"
        self.model = "ERA5"
        self.exp = "era5-hpz3"
        self.source = "monthly"
        self.regrid = "r100"
        self.std_startdate = "1990-01-01"
        self.std_enddate = "1991-12-31"
        self.diagnostic_name = "radiation"

    def test_gregory(self, tmp_path):
        """Test the Gregory class."""
        gp = Gregory(
            diagnostic_name=self.diagnostic_name,
            catalog=self.catalog,
            model=self.model,
            exp=self.exp,
            source=self.source,
            regrid=self.regrid,
            startdate=self.std_startdate,
            enddate=self.std_enddate,
            loglevel=loglevel,
        )

        gp.run(std=True, outputdir=tmp_path)

        assert isinstance(gp.t2m, xr.DataArray)
        assert isinstance(gp.net_toa, xr.DataArray)

        assert gp.t2m_monthly.values[0] == pytest.approx(12.274455935718379, rel=approx_rel)
        assert gp.net_toa_monthly.values[0] == pytest.approx(7.86250579018185, rel=approx_rel)

        assert gp.t2m_std.values == pytest.approx(0.0277312, rel=approx_rel)
        assert gp.net_toa_std.values == pytest.approx(0.52176817, rel=approx_rel)

        filename = f"{self.diagnostic_name}.gregory.{self.catalog}.{self.model}.{self.exp}.r1.2t.annual.nc"
        file = os.path.join(tmp_path, "netcdf", filename)
        assert os.path.exists(file)

        plt = PlotGregory(
            diagnostic_name=self.diagnostic_name,
            t2m_monthly_data=gp.t2m_monthly,
            net_toa_monthly_data=gp.net_toa_monthly,
            t2m_annual_data=gp.t2m_annual,
            net_toa_annual_data=gp.net_toa_annual,
            t2m_monthly_ref=gp.t2m_monthly,
            net_toa_monthly_ref=gp.net_toa_monthly,
            t2m_annual_ref=gp.t2m_annual,
            net_toa_annual_ref=gp.net_toa_annual,
            t2m_annual_std=gp.t2m_std,
            net_toa_annual_std=gp.net_toa_std,
            loglevel=loglevel,
        )

        title = plt.set_title()
        data_labels = plt.set_data_labels()
        ref_label = plt.set_ref_label()
        fig = plt.plot(title=title, data_labels=data_labels, ref_label=ref_label)
        _ = plt.set_description()
        plt.save_plot(fig, outputdir=tmp_path, diagnostic_product="gregory", dpi=DPI)

        filename = f"{self.diagnostic_name}.gregory.{self.catalog}.{self.model}.{self.exp}.r1.multiref.png"
        file = os.path.join(tmp_path, "png", filename)
        assert os.path.exists(file)


@pytest.mark.diagnostics
class TestGregoryUnit:
    """Lightweight unit tests for Gregory internal branches."""

    @staticmethod
    def _da(values, start="2000-01-01", freq="MS", **attrs):
        da = xr.DataArray(values, dims=["time"], coords={"time": pd.date_range(start, periods=len(values), freq=freq)})
        da.attrs.update(attrs)
        return da

    def test_retrieve_sets_default_realization_and_short_names(self, mocker):
        gp = Gregory(model="M", exp="E", source="S")
        ds = xr.Dataset({"2t": self._da([1.0, 2.0])})
        reader = mocker.MagicMock()

        mocker.patch("aqua.diagnostics.timeseries.gregory.Diagnostic._retrieve", return_value=(ds, reader, "ci"))
        formula_eval = mocker.patch("aqua.diagnostics.timeseries.gregory.EvaluateFormula")
        formula_eval.return_value.evaluate.return_value = self._da([3.0, 4.0])

        gp.retrieve(t2m=True, net_toa=True, t2m_name="2t", net_toa_name="a+b", reader_kwargs={})

        assert gp.realization == "r1"
        assert gp.t2m.attrs["short_name"] == "2t"
        assert gp.net_toa.attrs["short_name"] == "net_toa"

    def test_compute_t2m_empty_monthly_and_annual_sets_none(self, mocker):
        gp = Gregory(model="M", exp="E", source="S")
        gp.t2m = self._da([1.0, 2.0, 3.0], var_name="2t")
        gp.reader = mocker.MagicMock()
        gp.reader.fldmean.return_value = gp.t2m
        empty = xr.DataArray([], dims=["time"], coords={"time": pd.DatetimeIndex([])})
        gp.reader.timmean.side_effect = [empty, empty]

        mocker.patch("aqua.diagnostics.timeseries.gregory.convert_data_units", side_effect=lambda data, **_: data)
        gp.compute_t2m(freq=["monthly", "annual"], std=True, units="degC")

        assert gp.t2m_monthly is None
        assert gp.t2m_annual is None
        assert gp.t2m_std is None

    def test_compute_net_toa_with_std(self, mocker):
        gp = Gregory(model="M", exp="E", source="S")
        gp.net_toa = self._da([1.0, 2.0, 3.0, 4.0])
        gp.reader = mocker.MagicMock()
        gp.reader.fldmean.return_value = gp.net_toa
        monthly = self._da([1.0, 2.0, 3.0, 4.0])
        annual = self._da([1.0, 2.0], start="2000-01-01", freq="YS")
        gp.reader.timmean.side_effect = [monthly, annual]

        gp.compute_net_toa(freq=["monthly", "annual"], std=True)

        assert gp.net_toa_monthly is not None
        assert gp.net_toa_annual is not None
        assert gp.net_toa_std is not None

    def test_save_netcdf_dispatches_expected_products(self, mocker):
        gp = Gregory(diagnostic_name="gregory", model="M", exp="E", source="S")
        gp.t2m_monthly = self._da([1.0, 2.0])
        gp.t2m_annual = self._da([1.0, 2.0], start="2000-01-01", freq="YS")
        gp.t2m_std = xr.DataArray(np.array(1.0))
        gp.net_toa_monthly = self._da([3.0, 4.0])
        gp.net_toa_annual = self._da([3.0, 4.0], start="2000-01-01", freq="YS")
        gp.net_toa_std = xr.DataArray(np.array(2.0))

        save_mock = mocker.patch("aqua.diagnostics.timeseries.gregory.Diagnostic.save_netcdf")
        gp.save_netcdf(freq=["monthly", "annual"], std=True, t2m=True, net_toa=True, outputdir=".", rebuild=True)

        # t2m std/monthly/annual + net_toa std/monthly/annual
        assert save_mock.call_count == 6
