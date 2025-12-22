import numpy as np
import pandas as pd
import xarray as xr
import pytest

from ensemble.util import merge_from_data_files


@pytest.mark.ensemble
def test_merge_from_data_files_timeseries(tmp_path):
    """
    Test merging multiple NetCDF time-series files along ensemble dimension
    with temporal slicing.
    """
    var = "tas"
    ens_dim = "ensemble"

    # Create synthetic timeseries data
    time = pd.date_range("2000-01-01", periods=10, freq="MS")

    ds1 = xr.Dataset(
        {var: (("time",), np.random.rand(len(time)))},
        coords={"time": time},
    )

    ds2 = xr.Dataset(
        {var: (("time",), np.random.rand(len(time)))},
        coords={"time": time},
    )

    # Write datasets to NetCDF files
    f1 = tmp_path / "model1.nc"
    f2 = tmp_path / "model2.nc"
    ds1.to_netcdf(f1)
    ds2.to_netcdf(f2)

    model_names = ["ModelA", "ModelA"]

    merged = merge_from_data_files(
        variable=var,
        ens_dim=ens_dim,
        model_names=model_names,
        data_path_list=[str(f1), str(f2)],
        startdate="2000-03-01",
        enddate="2000-08-01",
        loglevel="WARNING",
    )

    # -----------------------
    # Assertions
    # -----------------------
    assert merged is not None
    assert ens_dim in merged.dims
    assert merged.dims[ens_dim] == 2

    assert "time" in merged.dims
    assert merged.time.values[0] >= np.datetime64("2000-03-01")
    assert merged.time.values[-1] <= np.datetime64("2000-08-01")

    assert var in merged.data_vars
    assert "model" in merged.coords
    assert list(merged.coords["model"].values) == model_names

    assert "description" in merged.attrs
    assert merged.attrs["model"] == model_names

@pytest.mark.ensemble
def test_merge_from_data_files_non_timeseries(tmp_path):
    """
    Test merging non-timeseries NetCDF files.
    """
    var = "psl"

    ds1 = xr.Dataset(
        {var: (("lat", "lon"), np.random.rand(5, 5))},
        coords={"lat": np.linspace(-90, 90, 5), "lon": np.linspace(0, 360, 5)},
    )

    ds2 = xr.Dataset(
        {var: (("lat", "lon"), np.random.rand(5, 5))},
        coords={"lat": np.linspace(-90, 90, 5), "lon": np.linspace(0, 360, 5)},
    )

    f1 = tmp_path / "file1.nc"
    f2 = tmp_path / "file2.nc"
    ds1.to_netcdf(f1)
    ds2.to_netcdf(f2)

    merged = merge_from_data_files(
        variable=var,
        data_path_list=[str(f1), str(f2)],
        model_names=["M1", "M2"],
    )

    assert "ensemble" in merged.dims
    assert merged.dims["ensemble"] == 2
    assert "time" not in merged.dims
    assert merged.attrs["model"] == ["M1", "M2"]

def test_merge_from_data_files_no_model_names(tmp_path):
    var = "tas"

    time = pd.date_range("2001-01-01", periods=5, freq="D")
    ds = xr.Dataset({var: (("time",), np.random.rand(5))}, coords={"time": time})

    f = tmp_path / "single.nc"
    ds.to_netcdf(f)

    merged = merge_from_data_files(
        variable=var,
        data_path_list=[str(f)],
    )

    assert merged.coords["model"].values.tolist() == ["model_name"]

