import pytest
import pandas as pd
import xarray as xr
from aqua.diagnostics.timeseries.util import loop_seasonalcycle, center_timestamp


@pytest.mark.diagnostics
@pytest.mark.parametrize("time,freq,expected", [
    ('2020-03-01', 'monthly', '2020-03-15 12:00:00'),
    ('2020-01-01', 'annual', '2020-07-02 12:00:00'),
])
def test_center_timestamp(time, freq, expected):
    """Test centering timestamp"""
    centered = center_timestamp(pd.Timestamp(time), freq=freq)
    assert centered == pd.Timestamp(expected)


@pytest.mark.diagnostics
def test_loop_seasonalcycle_monthly():
    """Test looping a seasonal cycle"""
    data = xr.DataArray(
        [10, 15, 20, 25, 30, 35, 30, 25, 20, 15, 10, 5],
        dims=['time'],
        coords={'time': pd.date_range('2020-01-01', periods=12, freq='MS')}
    )
    
    looped = loop_seasonalcycle(data, '2021-01-01', '2022-12-31', 
                                freq='monthly', center_time=False)
    
    assert len(looped.time) == 24
    assert looped.values[0] == looped.values[12] == 10


@pytest.mark.diagnostics
def test_loop_seasonalcycle_errors():
    """Test error handling in loop_seasonalcycle"""
    data = xr.DataArray([1], dims=['time'], 
                       coords={'time': pd.date_range('2020-01-01', periods=1, freq='MS')})
    
    with pytest.raises(ValueError):
        loop_seasonalcycle(None, '2020-01-01', '2020-12-31', 'monthly')