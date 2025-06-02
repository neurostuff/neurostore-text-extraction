"""Tests for normalize.py functions using parameterized test cases."""

import pytest
from unittest import mock

from ns_extract.pipelines.normalize import (
    normalize_string,
    load_abbreviations,
    resolve_abbreviations,
    find_and_remove_definitions,
)


@pytest.fixture
def test_sentences():
    """Collection of test sentences covering various cases for all functions."""
    return {
        "basic": "Magnetic Resonance Imaging (MRI) revealed structural changes",
        "multiple": (
            "Using Positron Emission Tomography (PET) and "
            "Magnetic Resonance Imaging (MRI) techniques"
        ),
        "mixed_case": (
            "FUNCTIONAL magnetic resonance imaging (fMRI) and "
            "electroencephalogram (EEG) Data"
        ),
        "nested": "Computed Tomography (CT (Computed Tomography)) scans",
        "apostrophe": (
            "Patient's functional magnetic resonance imaging (fMRI) results"
        ),
        "empty": "",
        "whitespace": "   ",
        "none_str": "None",
        "na_upper": "N/A",
        "na_lower": "n/a",
        "null": "NULL",
        "nan": "nan",
        "non_abbrev": "The study (published in 2023) showed results",
        "complex": (
            "The Diffusion Tensor Imaging (DTI) and "
            "Blood Oxygen Level Dependent (BOLD) signals"
        ),
        "repeated": (
            "The MRI was conducted. After reviewing the MRI results, "
            "another MRI was scheduled."
        ),
    }


@pytest.fixture
def test_words():
    """Individual words for testing normalize_string function."""
    return {
        "empty": "",
        "whitespace": "   ",
        "none_str": "None",
        "na_upper": "N/A",
        "na_lower": "n/a",
        "null": "NULL",
        "nan": "nan",
        "upper": "HELLO",
        "lower": "world",
        "mixed": "HeLLo",
        "apostrophe": "patient's",
    }


@pytest.mark.parametrize(
    "key,expected",
    [
        ("empty", None),
        ("whitespace", None),
        ("none_str", None),
        ("na_upper", "N/a"),
        ("na_lower", "N/a"),
        ("null", None),
        ("nan", None),
        ("upper", "Hello"),
        ("lower", "World"),
        ("mixed", "Hello"),
        ("apostrophe", "Patient's"),
    ],
)
def test_normalize_string(test_words, key, expected):
    """Test string normalization with various inputs."""
    assert normalize_string(test_words[key]) == expected


@pytest.mark.parametrize(
    "key,expected_abbrevs",
    [
        ("basic", [("MRI", "Magnetic Resonance Imaging")]),
        (
            "multiple",
            [
                ("PET", "Positron Emission Tomography"),
                ("MRI", "Magnetic Resonance Imaging"),
            ],
        ),
        (
            "mixed_case",
            [
                ("fMRI", "FUNCTIONAL magnetic resonance imaging"),
                ("EEG", "electroencephalogram"),
            ],
        ),
        ("empty", []),
        (
            "complex",
            [
                ("DTI", "Diffusion Tensor Imaging"),
                ("BOLD", "Blood Oxygen Level Dependent"),
            ],
        ),
    ],
)
def test_load_abbreviations(test_sentences, key, expected_abbrevs):
    """Test abbreviation extraction from sentences."""
    result = load_abbreviations(test_sentences[key])
    if not expected_abbrevs:
        assert not result
        return

    for short, long in expected_abbrevs:
        found = False
        for abbrev in result:
            if (
                abbrev["short_text"] == short
                and abbrev["long_text"].lower() == long.lower()
            ):
                found = True
                break
        assert found, f"Abbreviation {short} ({long}) not found in results"


@mock.patch.dict("sys.modules", {"scispacy": None, "scispacy.abbreviation": None})
def test_load_abbreviations_missing_scispacy(test_sentences):
    """Test error handling when scispacy is missing."""
    result = load_abbreviations(test_sentences["basic"])
    assert result == []


@pytest.mark.parametrize(
    "key,expected_abbrev_pairs,expected_output",
    [
        (
            "basic",
            [("MRI", "Magnetic Resonance Imaging")],
            "Magnetic Resonance Imaging (Magnetic Resonance Imaging) revealed structural changes",
        ),
        (
            "multiple",
            [
                ("PET", "Positron Emission Tomography"),
                ("MRI", "Magnetic Resonance Imaging"),
            ],
            (
                "Using Positron Emission Tomography (Positron Emission Tomography) and "
                "Magnetic Resonance Imaging (Magnetic Resonance Imaging) techniques"
            ),
        ),
        ("empty", [], ""),
        ("non_abbrev", [], "The study (published in 2023) showed results"),
        (
            "repeated",
            [("MRI", "Magnetic Resonance Imaging")],
            (
                "The Magnetic Resonance Imaging was conducted. "
                "After reviewing the Magnetic Resonance Imaging results, "
                "another Magnetic Resonance Imaging was scheduled."
            ),
        ),
    ],
)
def test_resolve_abbreviations(
    test_sentences, key, expected_abbrev_pairs, expected_output
):
    """Test abbreviation resolution in sentences."""
    abbrevs = [
        {"short_text": short, "long_text": long}
        for short, long in expected_abbrev_pairs
    ]
    result = resolve_abbreviations(test_sentences[key], abbrevs)
    assert result == expected_output


@pytest.mark.parametrize(
    "key,abbrev_pairs,expected",
    [
        (
            "basic",
            [("MRI", "Magnetic Resonance Imaging")],
            "Magnetic Resonance Imaging revealed structural changes",
        ),
        (
            "multiple",
            [
                ("PET", "Positron Emission Tomography"),
                ("MRI", "Magnetic Resonance Imaging"),
            ],
            "Using Positron Emission Tomography and Magnetic Resonance Imaging techniques",
        ),
        ("non_abbrev", [], "The study (published in 2023) showed results"),
        ("empty", [], ""),
    ],
)
def test_find_and_remove_definitions(test_sentences, key, abbrev_pairs, expected):
    """Test removal of parenthetical definitions."""
    abbrevs = [{"short_text": short, "long_text": long} for short, long in abbrev_pairs]
    result = find_and_remove_definitions(test_sentences[key], abbrevs)
    assert result.strip() == expected.strip()


def test_integration(test_sentences):
    """Test integration between all normalize functions."""
    # Mock the load_abbreviations call to return expected abbreviations
    # This avoids dependency on spacy/scispacy in tests
    abbrevs = [
        {
            "short_text": "MRI",
            "long_text": "Magnetic Resonance Imaging",
            "short_start": 28,
            "short_end": 31,
            "long_start": 0,
            "long_end": 25,
        }
    ]

    text = test_sentences["basic"]  # "Magnetic Resonance Imaging (MRI) revealed..."

    # Resolve abbreviations
    resolved = resolve_abbreviations(text, abbrevs)
    assert "Magnetic Resonance Imaging" in resolved
    assert "(Magnetic Resonance Imaging)" in resolved

    # Remove definitions
    cleaned = find_and_remove_definitions(resolved, abbrevs)
    assert "(MRI)" not in cleaned
    assert "Magnetic Resonance Imaging" in cleaned
    assert "revealed structural changes" in cleaned
