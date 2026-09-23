import os

import matplotlib
import numpy as np
import pytest

from aqua.core.exceptions import NotEnoughDataError
from aqua.diagnostics.teleconnections import DMI, PlotDMI
from tests.shared_constants import DPI, LOGLEVEL

loglevel = LOGLEVEL


@pytest.mark.diagnostics
def test_dmi(tmp_path):
    """
    Test that the DMI class works.
    """
    init_dict = {"model": "ERA5", "exp": "era5-hpz3", "source": "monthly", "loglevel": loglevel, "regrid": "r100"}

    dmi = DMI(**init_dict)

    with pytest.raises(NotEnoughDataError):
        dmi.compute_index()

    dmi.retrieve()
    assert dmi.data is not None, "Data should not be None"

    with pytest.raises(ValueError):
        dmi.compute_regression(season="annual")

    dmi.compute_index()
    assert dmi.index is not None, "Index should not be None"
    assert dmi.index.size > 0
    assert dmi.index.AQUA_startdate == dmi.data.AQUA_startdate
    assert dmi.index.AQUA_enddate == dmi.data.AQUA_enddate

    dmi.save_netcdf(dmi.index, diagnostic="dmi", diagnostic_product="index", outputdir=tmp_path)
    netcdf_path = os.path.join(tmp_path, "netcdf")
    filename = "dmi.index.ci.ERA5.era5-hpz3.r1.nc"
    assert os.path.exists(os.path.join(netcdf_path, filename))

    reg = dmi.compute_regression(season="annual")
    cor = dmi.compute_correlation()

    assert np.isfinite(reg.isel(lon=4, lat=23).values)
    assert np.isfinite(cor.isel(lon=4, lat=23).values)

    plot_ref = PlotDMI(loglevel=loglevel, indexes=dmi.index, ref_indexes=dmi.index, outputdir=tmp_path)

    fig, _ = plot_ref.plot_index()
    description = plot_ref.set_index_description()
    assert "index" in description.lower()
    assert isinstance(fig, matplotlib.figure.Figure)
    plot_ref.save_plot(fig, diagnostic_product="index", metadata={"description": description}, dpi=DPI)
    assert os.path.exists(os.path.join(tmp_path, "png", "dmi.index.ci.ERA5.era5-hpz3.r1.ci.ERA5.era5-hpz3.png"))

    reg.load()
    fig_reg = plot_ref.plot_maps(maps=reg, ref_maps=reg, statistic="regression")
    assert isinstance(fig_reg, matplotlib.figure.Figure)
    description = plot_ref.set_map_description(maps=reg, ref_maps=reg, statistic="regression")
    assert "regression" in description.lower()
    plot_ref.save_plot(
        fig_reg, diagnostic_product="regression_annual", metadata={"description": description}, format="pdf", dpi=DPI
    )
    assert os.path.exists(os.path.join(tmp_path, "pdf", "dmi.regression_annual.ci.ERA5.era5-hpz3.r1.ci.ERA5.era5-hpz3.pdf"))

    plot_single = PlotDMI(loglevel=loglevel, indexes=dmi.index, outputdir=tmp_path)
    cor.load()
    fig_cor = plot_single.plot_maps(maps=cor, statistic="correlation")
    assert isinstance(fig_cor, matplotlib.figure.Figure)
    description = plot_single.set_map_description(maps=cor, statistic="correlation")
    assert "correlation" in description.lower()
    plot_single.save_plot(
        fig_cor, diagnostic_product="correlation", metadata={"description": description}, format="pdf", dpi=DPI
    )
    assert os.path.exists(os.path.join(tmp_path, "pdf", "dmi.correlation.ci.ERA5.era5-hpz3.r1.pdf"))

    reg.attrs["AQUA_season"] = "annual"
    cor.attrs["AQUA_season"] = "annual"

    fig_reg_no_ref = plot_ref.plot_maps(maps=[reg, reg], ref_maps=None, statistic="regression")
    assert isinstance(fig_reg_no_ref, matplotlib.figure.Figure)

    fig_reg_single_ref = plot_ref.plot_maps(maps=[reg, reg], ref_maps=reg, statistic="regression")
    assert isinstance(fig_reg_single_ref, matplotlib.figure.Figure)

    reg2 = reg + reg
    fig_reg_list_ref = plot_ref.plot_maps(maps=reg2, ref_maps=[reg, reg], statistic="regression")
    assert isinstance(fig_reg_list_ref, matplotlib.figure.Figure)

    fig_not_implemented = plot_ref.plot_maps(maps=[reg, reg], ref_maps=[reg, reg], statistic="not_implemented")
    assert fig_not_implemented is None, "Plotting with not implemented statistic should return None"
