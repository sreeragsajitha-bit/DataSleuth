
import pandas as pd
import pytest

from data_pipeline import analyze_dataset


def test_pipeline_returns_all_analysis_sections(tmp_path):
    df = pd.DataFrame({
        "age": [20, 25, 30, 35, 40],
        "city": ["A", "B", "C", "D", "E"]
    })

    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)

    result = analyze_dataset(file_path)

    assert "profile" in result
    assert "quality_issues" in result
    assert "insights" in result
    assert "dataset_name" in result


def test_pipeline_detects_missing_values(tmp_path):
    df = pd.DataFrame({
        "age": [20, None, 30, 40],
        "city": ["A", "B", "C", "D"]
    })

    file_path = tmp_path / "missing_data.csv"
    df.to_csv(file_path, index=False)

    result = analyze_dataset(file_path)

    assert result["profile"]["total_missing_values"] == 1
    assert len(result["quality_issues"]) > 0

    missing_insights = [
        insight for insight in result["insights"]
        if insight["type"] == "missing_values"
    ]

    assert len(missing_insights) == 1


def test_pipeline_rejects_missing_file(tmp_path):
    file_path = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError):
        analyze_dataset(file_path)
