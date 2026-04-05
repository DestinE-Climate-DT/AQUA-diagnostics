"""Tests for the TitleBuilder class."""
import pytest

from aqua.diagnostics.base import TitleBuilder

pytestmark = pytest.mark.aqua

@pytest.mark.parametrize("kwargs,expected", [
    ({"title": "Custom Title"}, "Custom Title"),
    ({"diagnostic": "MLD", "regions": "global", "catalog": "ci", "model": "ERA5",
      "exp": "era5-hpz3", "timeseason": "climatology"},
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
        model="IFS",
        exp="test-exp",
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
        model="ERA5",
        exp="era5-hpz3",
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
    assert result == (
        "Stratification [global] for ci ERA5 era5-hpz3 r1 1990-1991 relative to IFS test 1980-1990 climatology info"
    )
    assert "  " not in result


def test_title_realizations():
    """Test realizations handling."""
    result = TitleBuilder(diagnostic="Bias", realizations=["r1", "r2"]).generate()
    assert "Bias Multi-realization" == result


def test_title_models_edge_cases():
    """Test edge cases for model and extra_info."""
    result1 = TitleBuilder(diagnostic="Bias", catalog=["ci", "ci"], model=["IFS", "FESOM"], exp=["exp1", "exp2"]).generate()
    assert "Bias for Multi-model" == result1
    result2 = TitleBuilder(diagnostic="Bias", extra_info=["info1", "info2"]).generate()
    assert "info1 info2" in result2
    assert TitleBuilder(diagnostic="Bias", regions=[""]).generate() == "Bias"


def test_title_wrap_not_triggered():
    """Wrapping is not triggered when title is short or no marker matches."""
    assert "\n" not in TitleBuilder(diagnostic="Bias", model="IFS").generate(max_chars=100)
    assert "\n" not in TitleBuilder(title="Supercalifragilisticexpialidocious").generate(max_chars=10, split_on=["for"])


def test_title_wrap_triggered():
    """Wrapping produces multiple lines all within max_chars."""
    for result, limit in [
        (TitleBuilder(diagnostic="Bias", variable="Temperature",
                      model="IFS", exp="historical",
                      ref_model="ERA5", ref_exp="era5",
                      comparison="vs", conjunction="in"
                      ).generate(max_chars=30, split_on=["vs", "in"]), 30),
        (TitleBuilder(title="Bias in IFS historical vs ERA5 era5").generate(max_chars=20, split_on=["vs", "in"]), 20),
    ]:
        lines = result.split("\n")
        assert len(lines) > 1
        assert all(len(line) <= limit for line in lines)


def test_title_marker_multi_times():
    """Same marker repeated: keep splitting until each segment fits ``max_chars``.

    One ``partition`` per input line would leave a tail such as ``for BBB for CCC``
    that can still exceed ``max_chars``; wrapping must apply the same marker again
    to that remainder in the same marker pass.
    """
    result = TitleBuilder(title="AAA for BBB for CCC").generate(max_chars=10, split_on=["for"])
    assert result == "AAA\nfor BBB\nfor CCC"
    lines = result.split("\n")
    assert len(lines) == 3
    assert all(len(line) <= 10 for line in lines)


def test_title_marker_multi_segments():
    """Several occurrences of one marker on a single long line all get split."""
    result = TitleBuilder(title="A for B for C for D").generate(max_chars=8, split_on=["for"])
    assert result == "A\nfor B\nfor C\nfor D"
    assert all(len(line) <= 8 for line in result.split("\n"))


def test_title_wrap_empty_title():
    """_wrap_title early-return on empty title."""
    assert TitleBuilder().generate(max_chars=50) == ""


def test_title_model_only():
    """models_part added without a 'for' prefix when title is empty."""
    assert TitleBuilder(model="IFS").generate() == "IFS"


def test_title_ref_only():
    """refs_part added without 'relative to' prefix when title is empty."""
    assert TitleBuilder(ref_model="ERA5").generate() == "ERA5"
    assert TitleBuilder(ref_catalog="ci").generate() == "ci"
