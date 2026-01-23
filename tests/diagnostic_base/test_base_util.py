import pytest
import argparse
import pandas as pd
from unittest.mock import patch
from aqua import Reader
from aqua.diagnostics.base import template_parse_arguments, load_diagnostic_config
from aqua.diagnostics.base import open_cluster, close_cluster, merge_config_args
from aqua.diagnostics.base import start_end_dates, round_startdate, round_enddate
from conftest import LOGLEVEL

loglevel = LOGLEVEL


@pytest.mark.aqua
def test_template_parse_arguments():
    """Test the template_parse_arguments function"""
    parser = argparse.ArgumentParser()
    parser = template_parse_arguments(parser)
    args = parser.parse_args(["--loglevel", "DEBUG", "--catalog", "test_catalog", "--model", "test_model",
                              "--exp", "test_exp", "--source", "test_source", "--config", "test_config.yaml",
                              "--regrid", "r100", "--outputdir", "test_outputdir", "--cluster", "test_cluster",
                              "--nworkers", "2"])

    assert args.loglevel == "DEBUG"
    assert args.catalog == "test_catalog"
    assert args.model == "test_model"
    assert args.exp == "test_exp"
    assert args.source == "test_source"
    assert args.config == "test_config.yaml"
    assert args.regrid == "r100"
    assert args.outputdir == "test_outputdir"
    assert args.cluster == "test_cluster"
    assert args.nworkers == 2

    with pytest.raises(FileNotFoundError):
        load_diagnostic_config(diagnostic='pippo', config=None, loglevel=loglevel)

@pytest.mark.aqua
@patch("aqua.diagnostics.base.util.Client")
@patch("aqua.diagnostics.base.util.LocalCluster")
def test_cluster(mock_cluster, mock_client):
    """Test the cluster functions with mocking"""

    # Test case 1: No workers specified
    client, cluster, private_cluster = open_cluster(None, None, loglevel)
    assert client is None
    assert cluster is None
    assert private_cluster is False

    # # Test case 2: New cluster creation
    client, cluster, private_cluster = open_cluster(2, None, loglevel)
    assert client is not None
    assert cluster is not None
    assert private_cluster is True

    # Test case 3: Using existing cluster
    previous_cluster = mock_cluster
    client, cluster, private_cluster = open_cluster(2, previous_cluster, loglevel)
    assert client is not None
    assert cluster is previous_cluster
    assert private_cluster is False

    close_cluster(client, cluster, private_cluster)


@pytest.mark.aqua
def test_load_diagnostic_config():
    """Test the load_diagnostic_config function"""
    parser = argparse.ArgumentParser()
    parser = template_parse_arguments(parser)
    args = parser.parse_args(["--loglevel", "DEBUG"])
    ts_dict = load_diagnostic_config(diagnostic='timeseries',
                                     default_config='config-timeseries.yaml',
                                     folder="templates",
                                     config=args.config, loglevel=loglevel)

    assert ts_dict['datasets'] == [{'catalog': None, 'exp': None, 'model': None, 'source': 'lra-r100-monthly',
                                    'regrid': None, 'reader_kwargs': None}]


@pytest.mark.aqua
def test_merge_config_args():
    """Test the merge_config_args function"""
    parser = argparse.ArgumentParser()
    parser = template_parse_arguments(parser)
    args = parser.parse_args(["--loglevel", "DEBUG", "--catalog", "test_catalog", "--model", "test_model",
                              "--exp", "test_exp", "--source", "test_source", "--outputdir", "test_outputdir"])

    ts_dict = {'datasets': [{'catalog': None, 'model': None, 'exp': None, 'source': 'lra-r100-monthly'}],
               'references': [{'catalog': 'obs', 'model': 'ERA5', 'exp': 'era5', 'source': 'monthly'}],
               'output': {'outputdir': './'}}

    merged_config = merge_config_args(config=ts_dict, args=args, loglevel=loglevel)

    assert merged_config['datasets'] == [{'catalog': 'test_catalog', 'exp': 'test_exp',
                                          'model': 'test_model', 'source': 'test_source'}]
    assert merged_config['output']['outputdir'] == 'test_outputdir'


@pytest.mark.aqua
def test_start_end_dates():
    # All None inputs
    assert start_end_dates() == (None, None)

    # Only startdate provided
    assert start_end_dates(startdate="2020-01-01") == (pd.Timestamp("2020-01-01"), None)

    # Two dates provided
    assert start_end_dates(startdate="2020-01-01", enddate="2020-01-02") == (
        pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")
    )
    assert start_end_dates(startdate="20200101", enddate="20200102") == (
        pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")
    )
    assert start_end_dates(startdate="20200101", start_std="20200102") == (
        pd.Timestamp("2020-01-01"), None
    )
    assert start_end_dates(startdate="2020-01-01", enddate="20200102") == (
        pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")
    )

    assert start_end_dates(start_std="2020-01-01", end_std="2020-01-02") == (
        None, None
    )

    assert start_end_dates(startdate="2020-01-01", end_std="2020-01-02") == (
        pd.Timestamp("2020-01-01"), None
    )

@pytest.mark.aqua
@pytest.mark.parametrize("date,freq,expected", [
    ('2020-03-15 14:30:00', 'monthly', '2020-03-01 00:00:00'),
    ('2020-06-15 14:30:00', 'annual', '2020-01-01 00:00:00'),
])
def test_round_startdate(date, freq, expected):
    """Test rounding to start of month/year"""
    rounded = round_startdate(pd.Timestamp(date), freq=freq)
    assert rounded == pd.Timestamp(expected)

@pytest.mark.parametrize("date,freq,expected", [
    ('2020-02-15 14:30:00', 'monthly', '2020-02-29 23:59:59'),
    ('2020-06-15 14:30:00', 'annual', '2020-12-31 23:59:59'),
])
def test_round_enddate(date, freq, expected):
    """Test rounding to end of month/year"""
    rounded = round_enddate(pd.Timestamp(date), freq=freq)
    assert rounded == pd.Timestamp(expected)

@pytest.mark.aqua
def test_round_invalid_freq():
    """Test error handling for invalid frequency"""
    with pytest.raises(ValueError):
        round_startdate(pd.Timestamp('2020-03-15'), freq='weekly')
    with pytest.raises(ValueError):
        round_enddate(pd.Timestamp('2020-03-15'), freq='weekly')