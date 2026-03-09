import pytest
from pathlib import Path
from aqua.diagnostics.ocean_trends import Trends
from aqua.diagnostics.ocean_trends.plot_trends import PlotTrends
from conftest import APPROX_REL, LOGLEVEL

loglevel = LOGLEVEL
approx_rel = APPROX_REL * 10

# --- Constants ---
EXPECTED_THETAO_TREND = -0.06603967
EXPECTED_SO_TREND     =  0.02622599

# --- Fixtures ---
@pytest.fixture(scope="session")
def trends_result(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("trends")
    trend = Trends(catalog='ci', model='FESOM',
                   exp='hpz3', source='monthly-3d',
                   regrid='r100', loglevel=loglevel)
    trend.run(var=['thetao', 'so'], region='go', outputdir=tmp_path)
    return trend, tmp_path

@pytest.fixture(scope="module")
def trends_plots(trends_result):
    """Run both plot types once, saving PNG and PDF."""
    trend, tmp_path = trends_result

    PlotTrends(data=trend.trend_coef,
               outputdir=tmp_path,
               loglevel=loglevel).plot_multilevel(save_png=True, save_pdf=True)

    PlotTrends(data=trend.trend_coef.mean('lon'),
               outputdir=tmp_path,
               loglevel=loglevel).plot_zonal(save_png=True, save_pdf=True)

    return tmp_path

# --- Tests ---

@pytest.mark.diagnostics
def test_trends_not_none(trends_result):
    trend, _ = trends_result
    assert trend is not None

@pytest.mark.diagnostics
def test_thetao_trend_coef(trends_result):
    trend, _ = trends_result
    actual = trend.trend_coef["thetao"].isel(level=1).mean('lat').mean('lon').values
    assert actual == pytest.approx(EXPECTED_THETAO_TREND, rel=approx_rel)

@pytest.mark.diagnostics
def test_so_trend_coef(trends_result):
    trend, _ = trends_result
    actual = trend.trend_coef["so"].isel(level=1).mean('lat').mean('lon').values
    assert actual == pytest.approx(EXPECTED_SO_TREND, rel=approx_rel)

@pytest.mark.diagnostics
def test_multilevel_png(trends_plots):
    png = Path(trends_plots) / "png" / "trends.multilevel_trend.ci.FESOM.hpz3.r1.global_ocean.png"
    assert png.is_file(), f"PNG not found: {png}"
    assert png.stat().st_size > 0

@pytest.mark.diagnostics
def test_multilevel_pdf(trends_plots):
    pdf = Path(trends_plots) / "pdf" / "trends.multilevel_trend.ci.FESOM.hpz3.r1.global_ocean.pdf"
    assert pdf.is_file(), f"PDF not found: {pdf}"
    assert pdf.stat().st_size > 0

@pytest.mark.diagnostics
def test_zonal_png(trends_plots):
    png = Path(trends_plots) / "png" / "trends.zonal_mean.ci.FESOM.hpz3.r1.global_ocean.png"
    assert png.is_file(), f"PNG not found: {png}"
    assert png.stat().st_size > 0

@pytest.mark.diagnostics
def test_zonal_pdf(trends_plots):
    pdf = Path(trends_plots) / "pdf" / "trends.zonal_mean.ci.FESOM.hpz3.r1.global_ocean.pdf"
    assert pdf.is_file(), f"PDF not found: {pdf}"
    assert pdf.stat().st_size > 0