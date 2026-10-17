"""Regression coverage for lazy, finite Tropical Cyclones processing windows."""

import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest
from dask.callbacks import Callback
from tropical_cyclones import TCs

from aqua.core.util import load_yaml

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics, pytest.mark.frontier]
CONFIG_FILE = Path(__file__).resolve().parents[2] / "frontier-diagnostics/tropical_cyclones/config/config_tcs.yaml"


@pytest.fixture(scope="module")
def atmospheric_fields(icon_test_r2b0_short_data):
    """Broadcast an existing catalog fixture lazily over the regression period."""
    data3d = (
        icon_test_r2b0_short_data.isel(time=0, height=slice(0, 2), cell=slice(0, 1), drop=True)
        .rename({"height": "plev", "t": "z"})
        .assign_coords(plev=[30000, 50000])
        .expand_dims(time=pd.date_range("1989-12-31", "1991-01-02", freq="h"))
    )
    data3d.attrs = {**data3d.attrs, "AQUA_test_marker": "preserve me"}
    data2d = data3d.isel(plev=0, drop=True).rename({"z": "msl"})
    data2d = data2d.assign({"10u": data2d.msl, "10v": data2d.msl})
    return data2d, data3d, data2d.assign(tprate=data2d.msl)


@pytest.fixture
def tc_setup(tmp_path, mocker, atmospheric_fields):
    """Configure real TCs control flow with catalog-backed fields and mocked I/O."""
    config = load_yaml(str(CONFIG_FILE))
    config["paths"] = {key: str(tmp_path / key) for key in config["paths"]}
    config["orography"] = None
    config["time"].update(startdate="19900101", enddate="19901231T18")
    readers = [mocker.MagicMock() for _ in atmospheric_fields]
    for reader, data in zip(readers, atmospheric_fields):
        reader.retrieve.return_value = data
    reader_factory = mocker.patch("tropical_cyclones.tropical_cyclones.Reader", side_effect=readers)
    processed = []
    mocker.patch.object(TCs, "readwrite_from_intake", side_effect=processed.append)
    mocker.patch.object(TCs, "run_detect_nodes")
    mocker.patch.object(TCs, "read_lonlat_nodes")
    mocker.patch.object(TCs, "store_detect_nodes")
    return config, readers, reader_factory, processed


def forbid_computation(*args):
    """Fail if window selection or mocked detection executes any Dask graph."""
    pytest.fail("Atmospheric fields must remain lazy until output")


@pytest.mark.parametrize("write_fullres", [False, True])
@pytest.mark.parametrize(
    "start, end, step, expected_windows",
    [
        ("19900101", "19901231T18", "1D", 365),
        ("19900101", "19901231T18", "10D", 37),
        ("19900101", "19901231T18", "400D", 1),
        ("19891231T06", "19900102T12", 1, 3),
        ("19900101", "19900102T00", "1D", 2),
        ("19900101", "19900102", "1D", 2),
    ],
)
def test_detect_each_timestep_once(tc_setup, start, end, step, expected_windows, write_fullres):
    """Daily and multi-day windows terminate without overlaps, eager work or metadata loss."""
    config, readers, reader_factory, processed = tc_setup
    config["time"].update(startdate=start, enddate=end)
    config["detect"]["write_fullres"] = write_fullres
    tropical = TCs(tdict=config, streaming=True, stream_step=step, loglevel="ERROR")
    expected_times = readers[0].retrieve.return_value.sel(time=slice(start, end)).indexes["time"]
    expected_steps = pd.date_range(expected_times[0], expected_times[-1], freq="6h").strftime("%Y%m%dT%H").tolist()
    windows = 0
    previous_end = None

    with Callback(start=forbid_computation):
        while tropical.data_retrieve():
            windows += 1
            assert windows <= len(expected_times), "Stream never terminates"
            assert previous_end is None or tropical.stream_startdate > previous_end
            previous_end = tropical.stream_enddate
            fields = [(tropical.data2d, readers[0]), (tropical.data3d, readers[1])]
            if write_fullres:
                fields.append((tropical.fullres, readers[2]))
            for data, reader in fields:
                assert data.indexes["time"].equals(tropical.data2d.indexes["time"])
                assert data.chunks
                assert data.aqua.instance is reader
                assert data.attrs["AQUA_test_marker"] == "preserve me"
                assert reader.retrieve.return_value.attrs["history"] in data.attrs["history"]
            tropical.detect_nodes_zoomin()
        assert tropical.data_retrieve() is False

    assert processed == expected_steps
    assert windows == expected_windows
    assert tropical.stream_enddate == expected_times[-1]
    for reader in readers[: 3 if write_fullres else 2]:
        reader.retrieve.assert_called_once()
    assert readers[2].retrieve.call_count == int(write_fullres)
    for call in reader_factory.call_args_list:
        assert "streaming" not in call.kwargs
        assert "aggregation" not in call.kwargs


def test_reset_reuses_cached_fields(tc_setup):
    """Reset restarts at the original stream start without retrieving data again."""
    config, readers, _, _ = tc_setup
    config["time"].update(enddate="19900103T18")
    tropical = TCs(tdict=config, streaming=True, stream_step="1D", stream_startdate="19900102", loglevel="ERROR")
    assert tropical.data_retrieve()
    first_times = tropical.data2d.indexes["time"]
    assert first_times[0] == pd.Timestamp("1990-01-02")
    assert tropical.data_retrieve()
    assert not tropical.data_retrieve()
    assert tropical.data_retrieve(reset_stream=True)
    assert tropical.data2d.indexes["time"].equals(first_times)
    for reader in readers[:2]:
        reader.retrieve.assert_called_once()


@pytest.mark.parametrize("missing", [None, "empty", "outside"])
def test_no_data_terminates_without_detection(tc_setup, missing, mocker):
    """Missing, empty or out-of-range data cannot enter the detection loop."""
    config, readers, _, processed = tc_setup
    if missing == "outside":
        config["time"].update(startdate="19920101", enddate="19920102")
    else:
        readers[0].retrieve.return_value = None if missing is None else readers[0].retrieve.return_value.isel(time=slice(0, 0))
    tropical = TCs(tdict=config, streaming=True, stream_step="1D", loglevel="ERROR")
    stitch = mocker.patch.object(tropical, "stitch_nodes_zoomin")
    tropical.loop_streaming(config)
    assert tropical.data_retrieve() is False
    assert processed == []
    stitch.assert_not_called()
    for reader in readers[:2]:
        reader.retrieve.assert_called_once()


@pytest.mark.parametrize("step", ["0D", "-1D", "6h", "NaT"])
def test_invalid_stream_step(tc_setup, step):
    """Reject windows that cannot make forward progress in whole days."""
    config, _, reader_factory, _ = tc_setup
    with pytest.raises(ValueError, match="positive whole number of days"):
        TCs(tdict=config, streaming=True, stream_step=step)
    reader_factory.assert_not_called()


def test_nonstreaming_retrieves_once(tc_setup):
    """Non-streaming use exposes the complete lazy fields without legacy Reader kwargs."""
    config, readers, _, _ = tc_setup
    tropical = TCs(tdict=config, loglevel="ERROR")
    with Callback(start=forbid_computation):
        assert tropical.data_retrieve()
        assert tropical.data_retrieve()
    assert tropical.data2d is readers[0].retrieve.return_value
    for reader in readers[:2]:
        reader.retrieve.assert_called_once()


@pytest.mark.parametrize("mode", ["--detect-only", "combined"])
def test_cli_stops_at_overridden_enddate(tc_setup, mocker, mode):
    """Both CLI paths process the overridden interval once and close the cluster."""
    config, readers, _, processed = tc_setup
    config["time"]["enddate"] = "20141231"
    config["stitch"].update(n_days_freq=1, n_days_ext=0)
    cli = mocker.MagicMock(config_dict=config, loglevel="ERROR")
    mocker.patch("aqua.diagnostics.base.DiagnosticCLI", return_value=cli)
    stitch = mocker.patch.object(TCs, "stitch_nodes_zoomin")
    args = ["tropical_cyclones", "-c", "unused.yaml", "--startdate", "19900101", "--enddate", "19900103T18"]
    if mode == "--detect-only":
        args.append(mode)
    mocker.patch.object(sys, "argv", args)
    with Callback(start=forbid_computation):
        runpy.run_module("tropical_cyclones.cli_tropical_cyclones", run_name="__main__")
    assert processed == pd.date_range("1990-01-01", "1990-01-03T18", freq="6h").strftime("%Y%m%dT%H").tolist()
    assert stitch.call_count == (0 if mode == "--detect-only" else 2)
    cli.close_dask_cluster.assert_called_once()
    for reader in readers[:2]:
        reader.retrieve.assert_called_once()
