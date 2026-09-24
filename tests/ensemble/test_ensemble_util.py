from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aqua.core.exceptions import NoDataError
from aqua.diagnostics.ensemble.util import (
    center_timestamp,
    compute_statistics,
    extract_realizations,
    extract_realizations_list,
    generate_realizations_path,
    merge_from_data_files,
    reader_retrieve_and_merge,
)

pytestmark = [pytest.mark.diagnostics, pytest.mark.ensemble]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_timeseries_dataset(variable="tas", value=1.0, periods=4):
    """Create a small in-memory timeseries dataset."""
    time = pd.date_range("2000-01-01", periods=periods, freq="MS")
    return xr.Dataset(
        {variable: (("time",), np.full(periods, value))},
        coords={"time": time},
    )


def make_latlon_dataset(variable="tas", value=1.0):
    """Create a small in-memory lat/lon dataset."""
    lat = np.linspace(-90, 90, 3)
    lon = np.linspace(0, 360, 4)

    return xr.Dataset(
        {variable: (("lat", "lon"), np.full((3, 4), value))},
        coords={"lat": lat, "lon": lon},
    )


# ---------------------------------------------------------------------------
# merge_from_data_files
# ---------------------------------------------------------------------------


@pytest.mark.ensemble
def test_merge_from_data_files_timeseries(tmp_path):
    """Test merging multiple NetCDF time-series files with temporal slicing."""
    var = "tas"
    ens_dim = "ensemble"

    time = pd.date_range("2000-01-01", periods=10, freq="MS")

    ds1 = xr.Dataset(
        {var: (("time",), np.random.rand(len(time)))},
        coords={"time": time},
    )

    ds2 = xr.Dataset(
        {var: (("time",), np.random.rand(len(time)))},
        coords={"time": time},
    )

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

    assert merged is not None
    assert ens_dim in merged.dims
    assert merged.sizes[ens_dim] == 2
    assert "time" in merged.dims

    assert merged.time.values[0] >= np.datetime64("2000-03-01")
    assert merged.time.values[-1] <= np.datetime64("2000-08-01")

    assert var in merged.data_vars
    assert "model" in merged.coords
    assert list(merged.coords["model"].values) == model_names

    assert "description" in merged.attrs
    assert merged.attrs["model"] == model_names

    merged.close()


@pytest.mark.ensemble
def test_merge_from_data_files_non_timeseries(tmp_path):
    """Test merging non-timeseries NetCDF files."""
    var = "psl"

    ds1 = xr.Dataset(
        {var: (("lat", "lon"), np.random.rand(5, 5))},
        coords={
            "lat": np.linspace(-90, 90, 5),
            "lon": np.linspace(0, 360, 5),
        },
    )

    ds2 = xr.Dataset(
        {var: (("lat", "lon"), np.random.rand(5, 5))},
        coords={
            "lat": np.linspace(-90, 90, 5),
            "lon": np.linspace(0, 360, 5),
        },
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
    assert merged.sizes["ensemble"] == 2
    assert "time" not in merged.dims
    assert merged.attrs["model"] == ["M1", "M2"]

    merged.close()


@pytest.mark.ensemble
def test_merge_from_data_files_no_model_names(tmp_path):
    """Test default model name generation."""
    var = "tas"

    time = pd.date_range("2001-01-01", periods=5, freq="D")

    ds = xr.Dataset(
        {var: (("time",), np.random.rand(5))},
        coords={"time": time},
    )

    f = tmp_path / "single.nc"
    ds.to_netcdf(f)

    merged = merge_from_data_files(
        variable=var,
        data_path_list=[str(f)],
    )

    assert merged.coords["model"].values.tolist() == ["model_name"]

    merged.close()


@pytest.mark.ensemble
def test_merge_from_data_files_empty_paths():
    """An empty data path list should raise ValueError."""
    with pytest.raises(ValueError, match="data_path_list"):
        merge_from_data_files(data_path_list=[])


@pytest.mark.ensemble
def test_merge_from_data_files_none_paths():
    """A missing data path list should raise ValueError."""
    with pytest.raises(ValueError, match="data_path_list"):
        merge_from_data_files(data_path_list=None)


@pytest.mark.ensemble
def test_merge_from_data_files_variable_not_present(tmp_path):
    """Test preprocessing when requested variable is not present."""
    ds = xr.Dataset(
        {
            "tas": (("time",), np.ones(3)),
            "psl": (("time",), np.ones(3) * 2),
        },
        coords={"time": pd.date_range("2000-01-01", periods=3)},
    )

    f = tmp_path / "data.nc"
    ds.to_netcdf(f)

    merged = merge_from_data_files(
        variable="not_present",
        data_path_list=[str(f)],
    )

    # The preprocessing function should leave the dataset untouched.
    assert "tas" in merged.data_vars
    assert "psl" in merged.data_vars

    merged.close()


@pytest.mark.ensemble
def test_merge_from_data_files_variable_none(tmp_path):
    """Test merging when no variable filter is requested."""
    ds = xr.Dataset(
        {
            "tas": (("time",), np.ones(3)),
            "psl": (("time",), np.ones(3) * 2),
        },
        coords={"time": pd.date_range("2000-01-01", periods=3)},
    )

    f = tmp_path / "data.nc"
    ds.to_netcdf(f)

    merged = merge_from_data_files(
        variable=None,
        data_path_list=[str(f)],
    )

    assert "tas" in merged.data_vars
    assert "psl" in merged.data_vars

    merged.close()


@pytest.mark.ensemble
def test_merge_from_data_files_single_model_explicit_name(tmp_path):
    """Test the single-model branch when model_names are explicitly supplied."""
    ds = make_timeseries_dataset()

    f1 = tmp_path / "one.nc"
    f2 = tmp_path / "two.nc"

    ds.to_netcdf(f1)
    ds.to_netcdf(f2)

    merged = merge_from_data_files(
        variable="tas",
        data_path_list=[str(f1), str(f2)],
        model_names=["ModelA", "ModelA"],
    )

    assert merged.attrs["model"] == ["ModelA", "ModelA"]
    assert list(merged["model"].values) == ["ModelA", "ModelA"]

    merged.close()


# ---------------------------------------------------------------------------
# compute_statistics
# ---------------------------------------------------------------------------


@pytest.mark.ensemble
def test_compute_statistics_no_model_coordinate():
    """Test ordinary mean/std for an ensemble without a model coordinate."""
    var = "tas"

    ds = xr.Dataset(
        {
            var: (
                ("ensemble", "time"),
                np.array(
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                        [5.0, 6.0],
                    ]
                ),
            )
        },
        coords={
            "ensemble": ["r1", "r2", "r3"],
            "time": [0, 1],
        },
    )

    mean, std = compute_statistics(variable=var, ds=ds)

    np.testing.assert_allclose(mean.values, [3.0, 4.0])
    np.testing.assert_allclose(std.values, np.std([1.0, 3.0, 5.0]))
    np.testing.assert_allclose(std.values[1], np.std([2.0, 4.0, 6.0]))


@pytest.mark.ensemble
def test_compute_statistics_none_data():
    """Test that missing data raises NoDataError."""
    with pytest.raises(NoDataError, match="No data is given"):
        compute_statistics(variable="tas", ds=None)


@pytest.mark.ensemble
def test_compute_statistics_single_model_coordinate():
    """Test model-coordinate branch when only one model is present."""
    var = "tas"

    values = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    ds = xr.Dataset(
        {var: (("ensemble", "time"), values)},
        coords={
            "ensemble": ["r1", "r2", "r3"],
            "time": [0, 1],
            "model": ("ensemble", ["ModelA", "ModelA", "ModelA"]),
        },
    )

    mean, std = compute_statistics(variable=var, ds=ds)

    np.testing.assert_allclose(mean.values, [3.0, 4.0])
    np.testing.assert_allclose(std.values, np.std(values, axis=0))


@pytest.mark.ensemble
def test_compute_statistics_multi_model_weighted():
    """
    Test the weighted multi-model branch with non-identical values.

    ModelA has two realizations and ModelB has one. The implementation
    assigns a weight proportional to the number of realizations to each
    member, so ModelA receives twice the total weight of ModelB.
    """
    var = "tas"

    values = np.array(
        [
            [1.0],
            [3.0],
            [10.0],
        ]
    )

    ds = xr.Dataset(
        {var: (("ensemble", "time"), values)},
        coords={
            "ensemble": ["r1", "r2", "r3"],
            "time": [0],
            "model": ("ensemble", ["ModelA", "ModelA", "ModelB"]),
        },
    )

    mean, std = compute_statistics(variable=var, ds=ds)

    # weights are [2, 2, 1], normalized to [0.4, 0.4, 0.2]
    expected_mean = (1 * 0.4) + (3 * 0.4) + (10 * 0.2)

    expected_variance = ((1 - expected_mean) ** 2) * 0.4 + ((3 - expected_mean) ** 2) * 0.4 + ((10 - expected_mean) ** 2) * 0.2

    expected_std = np.sqrt(expected_variance)

    np.testing.assert_allclose(mean.values, [expected_mean])
    np.testing.assert_allclose(std.values, [expected_std])

    assert "description" in mean.attrs
    assert "Weighted mean" in mean.attrs["description"]

    assert "description" in std.attrs
    assert "Weighted std" in std.attrs["description"]


# ---------------------------------------------------------------------------
# center_timestamp
# ---------------------------------------------------------------------------


@pytest.mark.ensemble
def test_center_timestamp():
    """Test monthly, annual and invalid timestamp frequencies."""
    time = pd.Timestamp("2000-01-01")

    centered_monthly = center_timestamp(time, "monthly")
    assert centered_monthly == pd.Timestamp("2000-01-16")

    centered_annual = center_timestamp(time, "annual")
    assert centered_annual == pd.Timestamp("2000-07-01")

    with pytest.raises(ValueError, match="not supported"):
        center_timestamp(time, "daily")


# ---------------------------------------------------------------------------
# reader_retrieve_and_merge
# ---------------------------------------------------------------------------


@pytest.mark.ensemble
def test_reader_retrieve_and_merge_no_inputs():
    """Return None when no source information is provided."""
    result = reader_retrieve_and_merge()

    assert result is None


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.Reader")
def test_reader_retrieve_and_merge_default_realization(mock_reader):
    """Test successful retrieval using the default r1 realization."""
    data = make_timeseries_dataset(value=2.0)

    mock_instance = MagicMock()
    mock_instance.retrieve.return_value = data
    mock_reader.return_value = mock_instance

    result = reader_retrieve_and_merge(
        catalog_list=["catalog"],
        model_list=["ModelA"],
        exp_list=["exp"],
        source_list=["source"],
        variable="tas",
    )

    assert result is not None
    assert "ensemble" in result.dims
    assert result.sizes["ensemble"] == 1
    assert result["ensemble"].values.tolist() == ["ModelA_exp_r1"]

    assert result.attrs["variable"] == "tas"
    assert result.attrs["ensemble_members"] == ["ModelA_exp_r1"]

    mock_reader.assert_called_once()
    mock_instance.retrieve.assert_called_once_with(var="tas")

    result.close()


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.Reader")
def test_reader_retrieve_and_merge_explicit_realizations(mock_reader):
    """Test multiple explicit realizations."""
    data1 = make_timeseries_dataset(value=1.0)
    data2 = make_timeseries_dataset(value=2.0)

    mock_instance = MagicMock()
    mock_instance.retrieve.side_effect = [data1, data2]
    mock_reader.return_value = mock_instance

    result = reader_retrieve_and_merge(
        catalog_list=["catalog"],
        model_list=["ModelA"],
        exp_list=["exp"],
        source_list=["source"],
        realizations={"ModelA": ["r1", "r2"]},
        variable="tas",
    )

    assert result.sizes["ensemble"] == 2
    assert result["ensemble"].values.tolist() == [
        "ModelA_exp_r1",
        "ModelA_exp_r2",
    ]

    assert mock_instance.retrieve.call_count == 2

    result.close()


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.Reader")
def test_reader_retrieve_and_merge_missing_model_realizations_uses_default(mock_reader):
    """Missing model entry in realizations should fall back to r1."""
    data = make_timeseries_dataset()

    mock_instance = MagicMock()
    mock_instance.retrieve.return_value = data
    mock_reader.return_value = mock_instance

    result = reader_retrieve_and_merge(
        catalog_list=["catalog"],
        model_list=["ModelA"],
        exp_list=["exp"],
        source_list=["source"],
        realizations={"OtherModel": ["r2"]},
        variable="tas",
    )

    assert result["ensemble"].values.tolist() == ["ModelA_exp_r1"]

    result.close()


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.Reader")
def test_reader_retrieve_and_merge_string_inputs(mock_reader):
    """Test conversion of string inputs to one-element lists."""
    data = make_timeseries_dataset()

    mock_instance = MagicMock()
    mock_instance.retrieve.return_value = data
    mock_reader.return_value = mock_instance

    result = reader_retrieve_and_merge(
        catalog_list="catalog",
        model_list="ModelA",
        exp_list="exp",
        source_list="source",
        variable="tas",
    )

    assert result.sizes["ensemble"] == 1
    assert result["ensemble"].values.tolist() == ["ModelA_exp_r1"]

    result.close()


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.Reader")
def test_reader_retrieve_and_merge_spatial_selection(mock_reader):
    """Test lon/lat spatial subsetting."""
    data = make_latlon_dataset()

    mock_instance = MagicMock()
    mock_instance.retrieve.return_value = data
    mock_reader.return_value = mock_instance

    result = reader_retrieve_and_merge(
        catalog_list=["catalog"],
        model_list=["ModelA"],
        exp_list=["exp"],
        source_list=["source"],
        variable="tas",
        lon_limits=[100, 200],
        lat_limits=[-45, 45],
    )

    assert result.sizes["lon"] == 1
    assert result.sizes["lat"] == 1
    assert result.lon.values[0] >= 100
    assert result.lon.values[0] <= 200
    assert result.lat.values[0] >= -45
    assert result.lat.values[0] <= 45
    result.close()


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.Reader")
def test_reader_retrieve_and_merge_temporal_selection(mock_reader):
    """Test temporal subsetting."""
    data = make_timeseries_dataset(periods=10)

    mock_instance = MagicMock()
    mock_instance.retrieve.return_value = data
    mock_reader.return_value = mock_instance

    result = reader_retrieve_and_merge(
        catalog_list=["catalog"],
        model_list=["ModelA"],
        exp_list=["exp"],
        source_list=["source"],
        variable="tas",
        startdate="2000-03-01",
        enddate="2000-06-01",
    )

    assert result.time.values[0] >= np.datetime64("2000-03-01")
    assert result.time.values[-1] <= np.datetime64("2000-06-01")

    result.close()


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.Reader")
def test_reader_retrieve_and_merge_no_lon_lat(mock_reader):
    """Spatial limits should be skipped for datasets without lon/lat."""
    data = make_timeseries_dataset()

    mock_instance = MagicMock()
    mock_instance.retrieve.return_value = data
    mock_reader.return_value = mock_instance

    result = reader_retrieve_and_merge(
        catalog_list=["catalog"],
        model_list=["ModelA"],
        exp_list=["exp"],
        source_list=["source"],
        variable="tas",
        lon_limits=[0, 10],
        lat_limits=[0, 10],
    )

    assert "time" in result.dims
    assert "lon" not in result.dims
    assert "lat" not in result.dims

    result.close()


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.Reader")
def test_reader_retrieve_and_merge_no_time(mock_reader):
    """Temporal limits should be skipped for datasets without time."""
    data = make_latlon_dataset()

    mock_instance = MagicMock()
    mock_instance.retrieve.return_value = data
    mock_reader.return_value = mock_instance

    result = reader_retrieve_and_merge(
        catalog_list=["catalog"],
        model_list=["ModelA"],
        exp_list=["exp"],
        source_list=["source"],
        variable="tas",
        startdate="2000-01-01",
        enddate="2000-02-01",
    )

    assert "time" not in result.dims

    result.close()


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.Reader")
def test_reader_retrieve_and_merge_nodata_error(mock_reader):
    """NoDataError should skip a realization."""
    good_data = make_timeseries_dataset(value=3.0)

    mock_instance = MagicMock()
    mock_instance.retrieve.side_effect = [
        NoDataError("missing data"),
        good_data,
    ]
    mock_reader.return_value = mock_instance

    result = reader_retrieve_and_merge(
        catalog_list=["catalog"],
        model_list=["ModelA"],
        exp_list=["exp"],
        source_list=["source"],
        realizations={"ModelA": ["r1", "r2"]},
        variable="tas",
    )

    assert result.sizes["ensemble"] == 1
    assert result["ensemble"].values.tolist() == ["ModelA_exp_r2"]

    result.close()


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.Reader")
def test_reader_retrieve_and_merge_all_realizations_missing(mock_reader):
    """If every realization raises NoDataError, return None."""
    mock_instance = MagicMock()
    mock_instance.retrieve.side_effect = NoDataError("missing data")
    mock_reader.return_value = mock_instance

    result = reader_retrieve_and_merge(
        catalog_list=["catalog"],
        model_list=["ModelA"],
        exp_list=["exp"],
        source_list=["source"],
        realizations={"ModelA": ["r1", "r2"]},
        variable="tas",
    )

    assert result is None


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.Reader")
def test_reader_retrieve_and_merge_filename_branch(mock_reader, tmp_path):
    """
    Test filename-based Reader retrieval.

    The current implementation expects realizations to be iterable in
    the filename branch, so provide one realization per filename.
    """
    data = make_timeseries_dataset()

    mock_instance = MagicMock()
    mock_instance.retrieve.return_value = data
    mock_reader.return_value = mock_instance

    filename = str(tmp_path / "data.nc")

    result = reader_retrieve_and_merge(
        filenames=[filename],
        realizations=["r1"],
        variable="tas",
    )

    assert result is not None
    assert result.sizes["ensemble"] == 1
    assert result["ensemble"].values.tolist() == ["r1"]

    mock_reader.assert_called_once()

    result.close()


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.Reader")
def test_reader_retrieve_and_merge_filename_error(mock_reader, tmp_path):
    """Errors in filename-based retrieval should skip the file."""
    mock_instance = MagicMock()
    mock_instance.retrieve.side_effect = RuntimeError("broken file")
    mock_reader.return_value = mock_instance

    filename = str(tmp_path / "broken.nc")

    result = reader_retrieve_and_merge(
        filenames=[filename],
        realizations=["r1"],
        variable="tas",
    )

    assert result is None


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.Reader")
def test_reader_retrieve_and_merge_reader_kwargs(mock_reader):
    """Additional Reader keyword arguments should be forwarded."""
    data = make_timeseries_dataset()

    mock_instance = MagicMock()
    mock_instance.retrieve.return_value = data
    mock_reader.return_value = mock_instance

    reader_retrieve_and_merge(
        catalog_list=["catalog"],
        model_list=["ModelA"],
        exp_list=["exp"],
        source_list=["source"],
        variable="tas",
        reader_kwargs={"foo": "bar"},
    )

    _, kwargs = mock_reader.call_args

    assert kwargs["foo"] == "bar"

    result = mock_instance.retrieve.return_value
    result.close()


# ---------------------------------------------------------------------------
# extract_realizations
# ---------------------------------------------------------------------------


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.ConfigPath")
def test_extract_realizations_returns_realizations(mock_config_path):
    """Return allowed realizations from the catalog."""
    source_entry = MagicMock()

    source_entry.describe.return_value = {
        "user_parameters": [
            {
                "name": "realization",
                "allowed": ["r1", "r2", "r3"],
            }
        ]
    }

    exp_entry = {"source": source_entry}
    model_entry = {"exp": exp_entry}
    catalog = {"ModelA": model_entry}

    cat_callable = MagicMock(return_value=catalog)

    configurer = MagicMock()
    configurer.deliver_intake_catalog.return_value = (
        cat_callable,
        "catalog.yaml",
        "machine.yaml",
    )

    mock_config_path.return_value = configurer

    result = extract_realizations(
        catalog="catalog",
        model="ModelA",
        exp="exp",
        source="source",
    )

    assert result == ["r1", "r2", "r3"]


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.ConfigPath")
def test_extract_realizations_no_realization(mock_config_path):
    """Return None if no realization parameter exists."""
    source_entry = MagicMock()

    source_entry.describe.return_value = {
        "user_parameters": [
            {
                "name": "member",
                "allowed": ["foo"],
            }
        ]
    }

    cat_callable = MagicMock(
        return_value={
            "ModelA": {
                "exp": {
                    "source": source_entry,
                }
            }
        }
    )

    configurer = MagicMock()
    configurer.deliver_intake_catalog.return_value = (
        cat_callable,
        "catalog.yaml",
        "machine.yaml",
    )

    mock_config_path.return_value = configurer

    result = extract_realizations(
        catalog="catalog",
        model="ModelA",
        exp="exp",
        source="source",
    )

    assert result is None


# ---------------------------------------------------------------------------
# extract_realizations_list
# ---------------------------------------------------------------------------


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.ConfigPath")
def test_extract_realizations_list_returns_realizations(mock_config_path):
    """Return realizations from an uninstantiated catalog entry."""
    source_entry = MagicMock()

    source_entry.describe.return_value = {
        "user_parameters": [
            {
                "name": "realization",
                "allowed": ["r1", "r2"],
            }
        ]
    }

    exp_entry = MagicMock()
    exp_entry._entries = {"source": source_entry}

    cat_callable = MagicMock(
        return_value={
            "ModelA": {
                "exp": exp_entry,
            }
        }
    )

    configurer = MagicMock()
    configurer.deliver_intake_catalog.return_value = (
        cat_callable,
        "catalog.yaml",
        "machine.yaml",
    )

    mock_config_path.return_value = configurer

    result = extract_realizations_list(
        catalog="catalog",
        model="ModelA",
        exp="exp",
        source="source",
    )

    assert result == ["r1", "r2"]


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.ConfigPath")
def test_extract_realizations_list_source_missing(mock_config_path):
    """Return None when the requested source is absent."""
    exp_entry = MagicMock()
    exp_entry._entries = {}

    cat_callable = MagicMock(
        return_value={
            "ModelA": {
                "exp": exp_entry,
            }
        }
    )

    configurer = MagicMock()
    configurer.deliver_intake_catalog.return_value = (
        cat_callable,
        "catalog.yaml",
        "machine.yaml",
    )

    mock_config_path.return_value = configurer

    result = extract_realizations_list(
        catalog="catalog",
        model="ModelA",
        exp="exp",
        source="missing_source",
    )

    assert result is None


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.ConfigPath")
def test_extract_realizations_list_no_realization(mock_config_path):
    """Return None when no realization parameter is available."""
    source_entry = MagicMock()

    source_entry.describe.return_value = {
        "user_parameters": [
            {
                "name": "other_parameter",
                "allowed": ["foo"],
            }
        ]
    }

    exp_entry = MagicMock()
    exp_entry._entries = {"source": source_entry}

    cat_callable = MagicMock(
        return_value={
            "ModelA": {
                "exp": exp_entry,
            }
        }
    )

    configurer = MagicMock()
    configurer.deliver_intake_catalog.return_value = (
        cat_callable,
        "catalog.yaml",
        "machine.yaml",
    )

    mock_config_path.return_value = configurer

    result = extract_realizations_list(
        catalog="catalog",
        model="ModelA",
        exp="exp",
        source="source",
    )

    assert result is None


# ---------------------------------------------------------------------------
# generate_realizations_path
# ---------------------------------------------------------------------------


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.OutputSaver")
def test_generate_realizations_path_with_realizations(mock_output_saver):
    """Generate one output path per realization."""
    saver1 = MagicMock()
    saver2 = MagicMock()

    saver1.generate_name.return_value = "output_r1"
    saver2.generate_name.return_value = "output_r2"

    mock_output_saver.side_effect = [saver1, saver2]

    result = generate_realizations_path(
        catalog="catalog",
        model="ModelA",
        exp="exp",
        diagnostic_name="ensemble",
        diagnostic_product="timeseries",
        variable="tas",
        file_dir="/tmp/output",
        realization_list=["r1", "r2"],
    )

    assert result == [
        "/tmp/output/output_r1.nc",
        "/tmp/output/output_r2.nc",
    ]

    assert mock_output_saver.call_count == 2
    saver1.generate_name.assert_called_once_with(
        diagnostic_product="timeseries",
        extra_keys=None,
    )
    saver2.generate_name.assert_called_once_with(
        diagnostic_product="timeseries",
        extra_keys=None,
    )


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.OutputSaver")
def test_generate_realizations_path_without_realizations(mock_output_saver):
    """Generate a single output path when no realization list is supplied."""
    saver = MagicMock()
    saver.generate_name.return_value = "output"
    mock_output_saver.return_value = saver

    result = generate_realizations_path(
        catalog="catalog",
        model="ModelA",
        exp="exp",
        diagnostic_name="ensemble",
        diagnostic_product="maps",
        variable="tas",
        file_dir="/tmp/output",
    )

    assert result == ["/tmp/output/output.nc"]

    mock_output_saver.assert_called_once()
    saver.generate_name.assert_called_once_with(
        diagnostic_product="maps",
        extra_keys=None,
    )


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.OutputSaver")
def test_generate_realizations_path_extra_keys_and_format(mock_output_saver):
    """Test custom extra keys and file format."""
    saver = MagicMock()
    saver.generate_name.return_value = "output_r1"
    mock_output_saver.return_value = saver

    result = generate_realizations_path(
        catalog="catalog",
        model="ModelA",
        exp="exp",
        diagnostic_name="ensemble",
        diagnostic_product="maps",
        variable="tas",
        file_dir="/tmp/output",
        realization_list=["r1"],
        extra_keys={"region": "global"},
        file_format=".zarr",
    )

    assert result == ["/tmp/output/output_r1.zarr"]

    saver.generate_name.assert_called_once_with(
        diagnostic_product="maps",
        extra_keys={"region": "global"},
    )


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.Reader")
def test_reader_retrieve_and_merge_filenames_spatial_temporal(mock_reader, tmp_path):
    """Test spatial and temporal subsetting in the filenames branch."""
    lat = np.linspace(-90, 90, 3)
    lon = np.linspace(0, 360, 4)
    time = pd.date_range("2000-01-01", periods=10, freq="MS")

    data = xr.Dataset(
        {"tas": (("time", "lat", "lon"), np.random.rand(10, 3, 4))},
        coords={"time": time, "lat": lat, "lon": lon},
    )

    mock_instance = MagicMock()
    mock_instance.retrieve.return_value = data
    mock_reader.return_value = mock_instance

    filename = str(tmp_path / "data.nc")

    result = reader_retrieve_and_merge(
        filenames=[filename],
        realizations=[["r1"]],
        variable="tas",
        lon_limits=[100, 200],
        lat_limits=[-45, 45],
        startdate="2000-03-01",
        enddate="2000-06-01",
    )

    assert result is not None
    # Spatial subsetting check
    assert result.lon.values[0] >= 100
    assert result.lon.values[-1] <= 200
    assert result.lat.values[0] >= -45
    assert result.lat.values[-1] <= 45
    # Temporal subsetting check
    assert result.time.values[0] >= np.datetime64("2000-03-01")
    assert result.time.values[-1] <= np.datetime64("2000-06-01")

    result.close()


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.Reader")
def test_reader_retrieve_and_merge_filenames_missing_dims(mock_reader, tmp_path):
    """Test missing lon/lat and time dims behavior in the filenames branch."""
    # Data missing both lat/lon and time coordinates
    data = xr.Dataset({"tas": np.array(1.0)})

    mock_instance = MagicMock()
    mock_instance.retrieve.return_value = data
    mock_reader.return_value = mock_instance

    filename = str(tmp_path / "data.nc")

    result = reader_retrieve_and_merge(
        filenames=[filename],
        realizations=[["r1"]],
        variable="tas",
        lon_limits=[100, 200],
        lat_limits=[-45, 45],
        startdate="2000-03-01",
        enddate="2000-06-01",
    )

    assert result is not None
    assert "lon" not in result.dims
    assert "lat" not in result.dims
    assert "time" not in result.dims

    result.close()


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.Reader")
def test_reader_retrieve_and_merge_filenames_reader_kwargs(mock_reader, tmp_path):
    """Test that reader_kwargs are correctly propagated when using the filenames backend."""
    data = make_timeseries_dataset()

    mock_instance = MagicMock()
    mock_instance.retrieve.return_value = data
    mock_reader.return_value = mock_instance

    filename = str(tmp_path / "data.nc")
    kwargs_to_pass = {"engine": "netcdf4", "chunks": "auto"}

    reader_retrieve_and_merge(
        filenames=[filename],
        realizations=[["r1"]],
        variable="tas",
        reader_kwargs=kwargs_to_pass,
    )

    # Extract kwargs passed to the Reader constructor
    _, called_kwargs = mock_reader.call_args
    assert called_kwargs.get("reader_kwargs") == kwargs_to_pass


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.Reader")
def test_reader_retrieve_and_merge_filenames_all_fail(mock_reader, tmp_path):
    """Test behavior when all files fail in the filename branch."""
    mock_instance = MagicMock()
    # Trigger a generic Exception as caught by the filename branch
    mock_instance.retrieve.side_effect = Exception("General file failure")
    mock_reader.return_value = mock_instance

    filename = str(tmp_path / "corrupted_data.nc")

    result = reader_retrieve_and_merge(
        filenames=[filename],
        realizations=[["r1"]],
        variable="tas",
    )

    # Should return None and exit gracefully via the `if not model_data_list:` check
    assert result is None


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.Reader")
def test_reader_retrieve_and_merge_catalog_error_propagation(mock_reader):
    """
    Test that non-NoDataError exceptions (like ValueError from config issues)
    are strictly propagated and NOT swallowed by the catalog loop.
    """
    mock_instance = MagicMock()
    mock_instance.retrieve.side_effect = ValueError("Severe configuration error")
    mock_reader.return_value = mock_instance

    with pytest.raises(ValueError, match="Severe configuration error"):
        reader_retrieve_and_merge(
            catalog_list=["catalog"],
            model_list=["ModelA"],
            exp_list=["exp"],
            source_list=["source"],
            variable="tas",
        )


@pytest.mark.ensemble
@patch("aqua.diagnostics.ensemble.util.xr.concat")
@patch("aqua.diagnostics.ensemble.util.Reader")
def test_reader_retrieve_and_merge_no_close_attr(mock_reader, mock_concat):
    """
    Test the garbage collection branch where the data object has no close() method.
    This ensures the `hasattr(data, "close")` line gets full branch coverage.
    """

    # Create a mock object that acts like data but lacks a close() method
    class MockDataWithoutClose:
        def __init__(self):
            self.dims = {"lon": 1, "lat": 1}

        def expand_dims(self, *args, **kwargs):
            return self

    mock_instance = MagicMock()
    mock_instance.retrieve.return_value = MockDataWithoutClose()
    mock_reader.return_value = mock_instance

    # Mock the concat function to return a Dataset that has an 'ensemble' coordinate!
    mock_concat.return_value = xr.Dataset(coords={"ensemble": ["ModelA_exp_r1"]}, attrs={})

    result = reader_retrieve_and_merge(
        catalog_list=["catalog"],
        model_list=["ModelA"],
        exp_list=["exp"],
        source_list=["source"],
        variable="tas",
    )

    assert result is not None
    assert "description" in result.attrs

    result.close()


@pytest.mark.ensemble
def test_merge_from_data_files_partial_dates(tmp_path):
    """
    Test the time slicing logic in merge_from_data_files when one date is missing.
    Ensures the `if startdate is not None and enddate is not None:` branch evaluates to False safely.
    """
    var = "tas"
    time = pd.date_range("2000-01-01", periods=3)
    ds = xr.Dataset({var: (("time",), np.ones(3))}, coords={"time": time})

    f1 = tmp_path / "model_a.nc"
    ds.to_netcdf(f1)

    # Provide startdate but leave enddate as None
    merged = merge_from_data_files(
        variable=var,
        data_path_list=[str(f1)],
        startdate="2000-01-01",
        enddate=None,
    )

    assert "time" in merged.dims
    assert len(merged.time) == 3  # Slicing should not have occurred

    merged.close()
