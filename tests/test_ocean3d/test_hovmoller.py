import pytest
from pathlib import Path
from aqua.diagnostics.ocean_drift import Hovmoller
from aqua.diagnostics.ocean_drift.plot_hovmoller import PlotHovmoller
from conftest import LOGLEVEL

loglevel = LOGLEVEL

# --- Constants ---
EXPECTED_THETAO = [22.2086629652034, -0.6924832430820729, -2.07305172]
EXPECTED_SO     = [36.57638045014168, 0.02545398252818387, 2.35781597]

# --- Fixtures ---

@pytest.fixture(scope="session")
def hovmoller_config():
    return {
        'catalog': 'ci',
        'model': 'FESOM',
        'exp': 'hpz3',
        'source': 'monthly-3d',
        'startdate': '1990-01-01',
        'enddate': '1990-03-31',
        'regrid': 'r200',
        'loglevel': loglevel
    }

@pytest.fixture(scope="module")
def hovmoller_result(tmp_path_factory, hovmoller_config):
    """Run the Hovmoller pipeline once for the entire test session."""
    tmp_path = tmp_path_factory.mktemp("hovmoller")
    hov = Hovmoller(**hovmoller_config)
    hov.run(anomaly_ref="t0", outputdir=tmp_path, region='sss')
    return hov, tmp_path

@pytest.fixture(scope="module")
def hovmoller_plot(hovmoller_result):
    """Run PlotHovmoller once, saving both formats."""
    hov, tmp_path = hovmoller_result
    hov_plot = PlotHovmoller(data=hov.processed_data_list,
                             loglevel=loglevel,
                             outputdir=tmp_path)
    hov_plot.plot_hovmoller(save_png=True, save_pdf=True)
    return hov_plot, tmp_path

# --- Tests ---

@pytest.mark.diagnostics
def test_hovmoller_not_none(hovmoller_result):
    hov, _ = hovmoller_result
    assert hov is not None

@pytest.mark.diagnostics
def test_processed_data_list_length(hovmoller_result):
    hov, _ = hovmoller_result
    assert len(hov.processed_data_list) == 3, "Should have 3 datasets (raw, anomaly, drift)"

@pytest.mark.diagnostics
@pytest.mark.parametrize("dataset_idx, expected", enumerate(EXPECTED_THETAO))
def test_thetao_values(hovmoller_result, dataset_idx, expected):
    hov, _ = hovmoller_result
    actual = hov.processed_data_list[dataset_idx].thetao.isel(level=1, time=1).values
    assert actual == pytest.approx(expected, abs=1e-4), \
        f"thetao mismatch at dataset {dataset_idx}"

@pytest.mark.diagnostics
@pytest.mark.parametrize("dataset_idx, expected", enumerate(EXPECTED_SO))
def test_so_values(hovmoller_result, dataset_idx, expected):
    hov, _ = hovmoller_result
    actual = hov.processed_data_list[dataset_idx].so.isel(level=1, time=1).values
    assert actual == pytest.approx(expected, abs=1e-4), \
        f"so mismatch at dataset {dataset_idx}"

@pytest.mark.diagnostics
def test_png_output(hovmoller_plot):
    _, tmp_path = hovmoller_plot
    png = Path(tmp_path) / "png" / "oceandrift.hovmoller.ci.FESOM.hpz3.r1.sargasso_sea.png"
    assert png.is_file(), f"PNG not found: {png}"
    assert png.stat().st_size > 0

@pytest.mark.diagnostics
def test_pdf_output(hovmoller_plot):
    _, tmp_path = hovmoller_plot
    pdf = Path(tmp_path) / "pdf" / "oceandrift.hovmoller.ci.FESOM.hpz3.r1.sargasso_sea.pdf"
    assert pdf.is_file(), f"PDF not found: {pdf}"
    assert pdf.stat().st_size > 0