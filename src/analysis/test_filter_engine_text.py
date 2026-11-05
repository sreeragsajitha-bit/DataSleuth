
import pandas as pd

from filter_engine import apply_filters


def test_string_equality_is_case_insensitive():
    df = pd.DataFrame({
        "category": ["Accessories", "Electronics", "accessories"]
    })

    result = apply_filters(
        df,
        [{"column": "category", "operator": "eq", "value": "accessories"}],
    )

    assert len(result) == 2


def test_string_equality_ignores_outer_spaces():
    df = pd.DataFrame({
        "category": [" Accessories ", "Electronics"]
    })

    result = apply_filters(
        df,
        [{"column": "category", "operator": "eq", "value": "accessories"}],
    )

    assert len(result) == 1


def test_string_not_equal_is_case_insensitive():
    df = pd.DataFrame({
        "category": ["Accessories", "Electronics", "accessories"]
    })

    result = apply_filters(
        df,
        [{"column": "category", "operator": "ne", "value": "ACCESSORIES"}],
    )

    assert result["category"].tolist() == ["Electronics"]


def test_string_in_is_case_insensitive():
    df = pd.DataFrame({
        "category": ["Accessories", "Electronics", "Furniture"]
    })

    result = apply_filters(
        df,
        [{
            "column": "category",
            "operator": "in",
            "value": ["ACCESSORIES", "electronics"],
        }],
    )

    assert len(result) == 2


def test_string_not_in_is_case_insensitive():
    df = pd.DataFrame({
        "category": ["Accessories", "Electronics", "Furniture"]
    })

    result = apply_filters(
        df,
        [{
            "column": "category",
            "operator": "not_in",
            "value": ["accessories"],
        }],
    )

    assert set(result["category"]) == {"Electronics", "Furniture"}
