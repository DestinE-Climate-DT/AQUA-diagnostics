import pandas as pd
import pytest

from aqua.diagnostics.base import Diagnostic
from tests.shared_constants import LOGLEVEL

loglevel = LOGLEVEL


@pytest.mark.aqua
def test_class_diagnostic(tmp_path):
    """
    Test for the Diagnostic class
    """
    catalog = None
    model = "ERA5"
    exp = "era5-hpz3"
    source = "monthly"
    var = "tcc"
    regrid = "r100"
    startdate = "19900101"
    enddate = "19910101"
    std_startdate = "19900601"
    std_enddate = "19900901"

    diag = Diagnostic(
        catalog=catalog,
        model=model,
        exp=exp,
        source=source,
        regrid=regrid,
        startdate=startdate,
        enddate=enddate,
        std_startdate=std_startdate,
        std_enddate=std_enddate,
        loglevel=loglevel,
    )

    assert diag.model == model
    assert diag.exp == exp
    assert diag.source == source
    assert diag.regrid == regrid
    assert diag.data is None
    assert diag.std_data is None

    diag.retrieve(var=var)

    assert diag.catalog == "ci"
    assert diag.data is not None
    assert diag.std_data is not None

    # Windows respect requested bounds
    assert pd.Timestamp(diag.data.time.values[0]) >= pd.Timestamp(startdate)
    assert pd.Timestamp(diag.data.time.values[-1]) <= pd.Timestamp(enddate)
    assert pd.Timestamp(diag.std_data.time.values[0]) >= pd.Timestamp(std_startdate)
    assert pd.Timestamp(diag.std_data.time.values[-1]) <= pd.Timestamp(std_enddate)

    assert "AQUA_startdate" in diag.data.attrs
    assert "AQUA_enddate" in diag.data.attrs
    assert "AQUA_std_startdate" in diag.std_data.attrs
    assert "AQUA_std_enddate" in diag.std_data.attrs

    # save_netcdf still works
    data_sel = diag.data.isel(time=0)
    diag.save_netcdf(data=data_sel, diagnostic="test", diagnostic_product="save", outputdir=tmp_path, rebuild=True)
    assert tmp_path.joinpath("netcdf/test.save.ci.ERA5.era5-hpz3.r1.nc").exists()


@pytest.mark.aqua
def test_retrieve_clip_to_bounds():
    """Out-of-range startdate/enddate are clipped to the catalog's effective bounds."""
    diag = Diagnostic(
        model="ERA5",
        exp="era5-hpz3",
        source="monthly",
        regrid="r100",
        startdate="18000101",
        enddate="21000101",
        loglevel=loglevel,
    )
    diag.retrieve(var="tcc")

    assert pd.Timestamp(diag.startdate) > pd.Timestamp("18000101")
    assert pd.Timestamp(diag.enddate) < pd.Timestamp("21000101")


@pytest.mark.aqua
def test_retrieve_without_std():
    """Without std window, std_data stays None and no AQUA_std_* attrs are attached."""
    diag = Diagnostic(
        model="ERA5",
        exp="era5-hpz3",
        source="monthly",
        regrid="r100",
        startdate="19900101",
        enddate="19910101",
        loglevel=loglevel,
    )
    diag.retrieve(var="tcc")

    assert diag.std_data is None
    assert "AQUA_std_startdate" not in diag.data.attrs
    assert "AQUA_std_enddate" not in diag.data.attrs


@pytest.mark.aqua
def test_load_netcdf_without_retrieve(tmp_path):
    """A fresh Diagnostic reads back what a previous run wrote, without retrieving any data.

    This is the plot only path: no Reader is instantiated, so the catalog is resolved from the
    triplet and the realization falls back to the default, as they were when the file was written.
    """
    producer = Diagnostic(
        model="ERA5",
        exp="era5-hpz3",
        source="monthly",
        regrid="r100",
        startdate="19900101",
        enddate="19910101",
        loglevel=loglevel,
    )
    producer.retrieve(var="tcc")
    producer.save_netcdf(data=producer.data.isel(time=0), diagnostic="test", diagnostic_product="save", outputdir=tmp_path)

    consumer = Diagnostic(model="ERA5", exp="era5-hpz3", source="monthly", loglevel=loglevel)
    assert consumer.catalog is None

    loaded = consumer.load_netcdf(diagnostic="test", diagnostic_product="save", outputdir=tmp_path, as_dataarray=True)

    assert consumer.catalog == "ci"
    # The attributes the plot classes rely on survive the round trip
    assert loaded.attrs["AQUA_model"] == "ERA5"
    assert loaded.attrs["AQUA_exp"] == "era5-hpz3"
    assert consumer.load_netcdf(diagnostic="test", diagnostic_product="never-computed", outputdir=tmp_path) is None


@pytest.mark.aqua
def test_resolve_catalog():
    """The catalog taking part in the filenames is found from the triplet, with no data access."""
    diag = Diagnostic(model="ERA5", exp="era5-hpz3", source="monthly", loglevel=loglevel)
    diag._resolve_catalog()
    assert diag.catalog == "ci"

    unknown = Diagnostic(model="ERA5", exp="not-an-experiment", source="monthly", loglevel=loglevel)
    with pytest.raises(KeyError, match="Cannot find"):
        unknown._resolve_catalog()
