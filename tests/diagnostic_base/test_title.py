"""Tests for the TitleBuilder class."""

import pytest

from aqua.diagnostics.base import TitleBuilder

pytestmark = pytest.mark.aqua


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"title": "Custom Title"}, "Custom Title"),
        (
            {
                "diagnostic": "MLD",
                "regions": "global",
                "catalog": "ci",
                "model": "ERA5",
                "exp": "era5-hpz3",
                "timeseason": "climatology",
            },
            "MLD [global] for ci ERA5 era5-hpz3 climatology",
        ),
        ({}, ""),  # Empty result
        ({"variable": "Temperature"}, "Temperature"),
        ({"diagnostic": "Test", "startyear": 2020}, "Test 2020"),
        ({"diagnostic": "Test", "endyear": 2021}, "Test 2021"),
        ({"diagnostic": "Bias", "realizations": ["r1", "r2"]}, "Bias Multi-realization"),
        ({"model": "IFS"}, "IFS"),
        ({"ref_model": "ERA5"}, "ERA5"),
        ({"ref_catalog": "ci"}, "ci"),
    ],
)
def test_title_basic(kwargs, expected):
    """Test basic title generation."""
    result = TitleBuilder(**kwargs).generate()
    assert result == expected
    assert "  " not in result


def test_title_references():
    """Test reference components with custom comparison and conjunction."""
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
        conjunction="in",
    ).generate()
    assert "  " not in result
    assert "Bias of Temperature" in result
    assert "in IFS" in result
    assert "vs ERA5 era5" in result
    assert "1980-1990" in result


def test_title_complex():
    """Test full title assembly with every component."""
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
        extra_info="info",
    ).generate()
    assert result == (
        "Stratification [global] for ci ERA5 era5-hpz3 r1 1990-1991 relative to IFS test 1980-1990 climatology info"
    )
    assert "  " not in result


def test_title_models_edge_cases():
    """Test multi-model, extra_info list, and empty region."""
    result1 = TitleBuilder(diagnostic="Bias", catalog=["ci", "ci"], model=["IFS", "FESOM"], exp=["exp1", "exp2"]).generate()
    assert result1 == "Bias for Multi-model"

    result2 = TitleBuilder(diagnostic="Bias", extra_info=["info1", "info2"]).generate()
    assert "info1 info2" in result2

    assert TitleBuilder(diagnostic="Bias", regions=[""]).generate() == "Bias"


def test_title_wrap_not_triggered():
    """Wrapping is skipped when title fits, no marker matches, or marker is not surrounded by spaces."""
    assert "\n" not in TitleBuilder(diagnostic="Bias", model="IFS").generate(max_chars=100)
    # No matching marker in split_on
    assert "\n" not in TitleBuilder(title="Supercalifragilisticexpialidocious").generate(max_chars=10, split_on=["for"])
    # Marker present but only at the very start (no leading space → separator not found)
    assert "\n" not in TitleBuilder(title="for very long tail").generate(max_chars=8, split_on=["for"])


@pytest.mark.parametrize(
    "title,max_chars,split_on,expected",
    [
        # Two different markers used in sequence
        ("Bias in IFS historical vs ERA5 era5", 20, ["vs", "in"], None),
        # Tail short enough after a single split – loop must stop
        ("AAAA for BBB", 9, ["for"], "AAAA\nfor BBB"),
        # Later marker handles what the earlier marker could not
        ("LongAlpha in Beta", 10, ["for", "in"], "LongAlpha\nin Beta"),
    ],
)
def test_title_wrap_triggered(title, max_chars, split_on, expected):
    """Wrapping produces lines all within max_chars; exact output checked when expected is given."""
    result = TitleBuilder(title=title).generate(max_chars=max_chars, split_on=split_on)
    lines = result.split("\n")
    assert len(lines) > 1
    assert all(len(line) <= max_chars for line in lines)
    if expected is not None:
        assert result == expected


@pytest.mark.parametrize(
    "title,max_chars,expected",
    [
        # Two occurrences → three lines
        ("AAA for BBB for CCC", 10, "AAA\nfor BBB\nfor CCC"),
        # Three occurrences → four lines
        ("A for B for C for D", 8, "A\nfor B\nfor C\nfor D"),
    ],
)
def test_title_wrap_repeated_marker(title, max_chars, expected):
    """Same marker repeated: the while-loop keeps splitting until every segment fits."""
    result = TitleBuilder(title=title).generate(max_chars=max_chars, split_on=["for"])
    assert result == expected
    assert all(len(line) <= max_chars for line in result.split("\n"))
