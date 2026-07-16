"""
Shared fixtures for AQUA-diagnostics test suite.
These fixtures use scope="session" to retrieve data once and share across all tests.
Reference: https://docs.pytest.org/en/stable/reference/fixtures.html
"""

import os
import shutil
import tempfile

import matplotlib
import pytest

from aqua import Reader  # type: ignore
from tests.shared_constants import LOGLEVEL

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

plt.ioff()  # Turn off interactive mode explicitly

_WORKER_TMPDIR_ATTR = "_worker_tmpdir"


def pytest_configure(config):
    """Set per-worker TMPDIR to avoid CDO/temp contention under pytest-xdist."""
    workerinput = getattr(config, "workerinput", None)
    if workerinput is None:
        return

    worker_id = workerinput.get("workerid", "master")
    worker_tmpdir = tempfile.mkdtemp(prefix=f"aqua_diag_pytest_{worker_id}_")
    os.environ["TMPDIR"] = worker_tmpdir
    setattr(config, _WORKER_TMPDIR_ATTR, worker_tmpdir)


def pytest_sessionfinish(session, exitstatus):
    """Remove the per-worker TMPDIR created in pytest_configure."""
    worker_tmpdir = getattr(session.config, _WORKER_TMPDIR_ATTR, None)
    if worker_tmpdir:
        shutil.rmtree(worker_tmpdir, ignore_errors=True)


# ======================================================================
# IFS fixtures
# ======================================================================
@pytest.fixture(scope="session")
def ifs_tco79_short_reader():
    return Reader(model="IFS", exp="test-tco79", source="short", loglevel=LOGLEVEL)


@pytest.fixture(scope="session")
def ifs_tco79_short_data(ifs_tco79_short_reader):
    return ifs_tco79_short_reader.retrieve()


@pytest.fixture(scope="session")
def ifs_tco79_short_data_2t(ifs_tco79_short_reader):
    return ifs_tco79_short_reader.retrieve(var="2t")


@pytest.fixture(scope="session")
def ifs_tco79_short_r100_reader():
    return Reader(model="IFS", exp="test-tco79", source="short", loglevel=LOGLEVEL, regrid="r100")


@pytest.fixture(scope="session")
def ifs_tco79_short_r100_data(ifs_tco79_short_r100_reader):
    return ifs_tco79_short_r100_reader.retrieve()


@pytest.fixture(scope="session")
def ifs_tco79_short_r200_reader():
    return Reader(model="IFS", exp="test-tco79", source="short", loglevel=LOGLEVEL, regrid="r200")


@pytest.fixture(scope="session")
def ifs_tco79_short_r200_data(ifs_tco79_short_r200_reader):
    return ifs_tco79_short_r200_reader.retrieve()


@pytest.fixture(scope="session")
def ifs_tco79_long_fixfalse_reader():
    return Reader(model="IFS", exp="test-tco79", source="long", fix=False, loglevel=LOGLEVEL)


@pytest.fixture(scope="session")
def ifs_tco79_long_fixfalse_data(ifs_tco79_long_fixfalse_reader):
    return ifs_tco79_long_fixfalse_reader.retrieve(var=["2t", "ttr"])


@pytest.fixture(scope="session")
def ifs_tco79_long_reader():
    return Reader(model="IFS", exp="test-tco79", source="long", loglevel=LOGLEVEL)


@pytest.fixture(scope="session")
def ifs_tco79_long_data(ifs_tco79_long_reader):
    return ifs_tco79_long_reader.retrieve()


# ======================================================================
# FESOM fixtures
# ======================================================================
@pytest.fixture(scope="session")
def fesom_test_pi_original_2d_reader():
    return Reader(model="FESOM", exp="test-pi", source="original_2d", loglevel=LOGLEVEL)


@pytest.fixture(scope="session")
def fesom_test_pi_original_2d_data(fesom_test_pi_original_2d_reader):
    return fesom_test_pi_original_2d_reader.retrieve(var="tos")


@pytest.fixture(scope="session")
def fesom_test_pi_original_2d_r200_fixfalse_reader():
    return Reader(model="FESOM", exp="test-pi", source="original_2d", regrid="r200", fix=False, loglevel=LOGLEVEL)


@pytest.fixture(scope="session")
def fesom_test_pi_original_2d_r200_fixfalse_data(fesom_test_pi_original_2d_r200_fixfalse_reader):
    return fesom_test_pi_original_2d_r200_fixfalse_reader.retrieve()


# ======================================================================
# ICON fixtures
# ======================================================================
@pytest.fixture(scope="session")
def icon_test_healpix_short_reader():
    return Reader(model="ICON", exp="test-healpix", source="short", loglevel=LOGLEVEL)


@pytest.fixture(scope="session")
def icon_test_healpix_short_data(icon_test_healpix_short_reader):
    return icon_test_healpix_short_reader.retrieve(var="2t")


@pytest.fixture(scope="session")
def icon_test_r2b0_short_reader():
    return Reader(model="ICON", exp="test-r2b0", source="short", loglevel=LOGLEVEL)


@pytest.fixture(scope="session")
def icon_test_r2b0_short_data(icon_test_r2b0_short_reader):
    return icon_test_r2b0_short_reader.retrieve(var="t")


# ======================================================================
# NEMO fixtures
# ======================================================================
@pytest.fixture(scope="session")
def nemo_test_e_orca1_long_2d_reader():
    return Reader(model="NEMO", exp="test-e_orca1", source="long-2d", loglevel=LOGLEVEL)


@pytest.fixture(scope="session")
def nemo_test_e_orca1_long_2d_data(nemo_test_e_orca1_long_2d_reader):
    return nemo_test_e_orca1_long_2d_reader.retrieve(var="tos")


@pytest.fixture(scope="session")
def nemo_test_e_orca1_short_3d_reader():
    return Reader(model="NEMO", exp="test-e_orca1", source="short-3d", loglevel=LOGLEVEL)


@pytest.fixture(scope="session")
def nemo_test_e_orca1_short_3d_data(nemo_test_e_orca1_short_3d_reader):
    return nemo_test_e_orca1_short_3d_reader.retrieve(var="so")


# ======================================================================
# ERA5 fixtures
# ======================================================================
@pytest.fixture(scope="session")
def era5_hpz3_monthly_reader():
    return Reader(model="ERA5", exp="era5-hpz3", source="monthly", loglevel=LOGLEVEL)


@pytest.fixture(scope="session")
def era5_hpz3_monthly_data(era5_hpz3_monthly_reader):
    return era5_hpz3_monthly_reader.retrieve(var=["2t", "tprate", "q"])


@pytest.fixture(scope="session")
def era5_hpz3_monthly_r100_reader():
    return Reader(model="ERA5", exp="era5-hpz3", source="monthly", regrid="r100", loglevel=LOGLEVEL)


@pytest.fixture(scope="session")
def era5_hpz3_monthly_r100_data(era5_hpz3_monthly_r100_reader):
    return era5_hpz3_monthly_r100_reader.retrieve(var=["q"])
