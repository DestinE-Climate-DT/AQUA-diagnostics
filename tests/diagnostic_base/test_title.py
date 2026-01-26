"""Tests for the TitleBuilder class."""
import pytest
from aqua.diagnostics.base import TitleBuilder

pytestmark = pytest.mark.aqua

@pytest.mark.parametrize("kwargs,expected", [
    ({"title": "Custom Title"}, "Custom Title"),
    ({"diagnostic": "MLD", "regions": "global", "catalog": "ci", "models": "ERA5", 
      "exps": "era5-hpz3", "timeseason": "climatology"}, 
      "MLD [global] for ci ERA5 era5-hpz3 climatology"),
    ({}, ""), # Empty result
    ({"variable": "Temperature"}, "Temperature"),
    ({"diagnostic": "Test", "startyear": 2020}, "Test 2020"),
    ({"diagnostic": "Test", "endyear": 2021}, "Test 2021"),
])
def test_title_basic(kwargs, expected):
    """Test basic title generation and spacing fix."""
    result = TitleBuilder(**kwargs).generate()
    assert result == expected
    assert "  " not in result

def test_title_references():
    """Test reference data handling."""
    result = TitleBuilder(
        diagnostic="Bias",
        variable="Temperature",
        models="IFS",
        exps="test-exp",
        ref_model="ERA5",
        ref_exp="era5",
        ref_startyear=1980,
        ref_endyear="1990",
        comparison="vs",
        conjunction="in"
    ).generate()
    assert "  " not in result
    assert "Bias of Temperature" in result
    assert "in IFS" in result
    assert "vs ERA5 era5" in result
    assert "1980-1990" in result

def test_title_complex():
    """Test complex title with multiple components."""
    result = TitleBuilder(
        diagnostic="Stratification",
        regions="global",
        catalog="ci",
        models="ERA5",
        exps="era5-hpz3",
        realizations="r1",
        startyear=1990,
        endyear="1991",
        timeseason="climatology",
        ref_model="IFS",
        ref_exp="test",
        ref_startyear="1980",
        ref_endyear=1990,
        extra_info="info"
    ).generate()
    assert result == "Stratification [global] for ci ERA5 era5-hpz3 r1 1990-1991 relative to IFS test 1980-1990 climatology info"
    assert "  " not in result

def test_title_realizations():
    """Test realizations handling."""
    result = TitleBuilder(diagnostic="Bias", realizations=["r1", "r2"]).generate()
    assert "Bias Multi-realization" == result

def test_title_models_edge_cases():
    """Test edge cases for models and extra_info."""
    result1 = TitleBuilder(diagnostic="Bias", catalog=["ci", "ci"], models=["IFS", "FESOM"], exps=["exp1", "exp2"]).generate()
    assert "Bias for Multi-model" == result1
    result2 = TitleBuilder(diagnostic="Bias", extra_info=["info1", "info2"]).generate()
    assert "info1 info2" in result2
