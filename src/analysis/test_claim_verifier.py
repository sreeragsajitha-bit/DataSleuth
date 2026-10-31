
import pytest

from claim_verifier import (
    get_evidence_value,
    values_match,
    verify_claims,
    create_verification_summary
)


@pytest.fixture
def sample_report():
    return {
        "dataset_name": "sales.csv",
        "profile": {
            "column_profiles": {
                "age": {
                    "statistics": {
                        "mean": 24.375,
                        "median": 24.0
                    },
                    "missing_count": 2
                },
                "quantity": {
                    "statistics": {
                        "mean": 1.5
                    }
                }
            }
        },
        "insights": [
            {
                "type": "missing_values",
                "evidence": {
                    "missing_count": 2,
                    "missing_percentage": 20.0
                }
            },
            {
                "type": "skewness",
                "evidence": {
                    "skewness": 1.1785,
                    "direction": "right"
                }
            }
        ]
    }


def test_get_evidence_value(sample_report):
    value = get_evidence_value(
        sample_report,
        "profile.column_profiles.age.statistics.mean"
    )

    assert value == 24.375


def test_get_evidence_value_from_list(sample_report):
    value = get_evidence_value(
        sample_report,
        "insights.1.evidence.skewness"
    )

    assert value == 1.1785


def test_get_evidence_value_from_first_list_item(sample_report):
    value = get_evidence_value(
        sample_report,
        "insights.0.evidence.missing_percentage"
    )

    assert value == 20.0


def test_invalid_list_index(sample_report):
    with pytest.raises(KeyError):
        get_evidence_value(
            sample_report,
            "insights.10.evidence.skewness"
        )


def test_non_numeric_list_index(sample_report):
    with pytest.raises(KeyError):
        get_evidence_value(
            sample_report,
            "insights.age.evidence.skewness"
        )


def test_valid_claim(sample_report):
    claims = [{
        "claim": "The mean age is 24.375.",
        "evidence_path": "profile.column_profiles.age.statistics.mean",
        "value": 24.375
    }]

    result = verify_claims(sample_report, claims)

    assert result["verified_count"] == 1
    assert result["rejected_count"] == 0
    assert result["all_verified"] is True


def test_valid_list_claim(sample_report):
    claims = [{
        "claim": "Quantity skewness is 1.1785.",
        "evidence_path": "insights.1.evidence.skewness",
        "value": 1.1785
    }]

    result = verify_claims(sample_report, claims)

    assert result["verified_count"] == 1
    assert result["rejected_count"] == 0


def test_incorrect_numerical_claim(sample_report):
    claims = [{
        "claim": "The mean age is 30.",
        "evidence_path": "profile.column_profiles.age.statistics.mean",
        "value": 30
    }]

    result = verify_claims(sample_report, claims)

    assert result["verified_count"] == 0
    assert result["rejected_count"] == 1
    assert result["all_verified"] is False


def test_wrong_column_evidence_path(sample_report):
    claims = [{
        "claim": "The mean quantity is 24.375.",
        "evidence_path": "profile.column_profiles.age.statistics.mean",
        "value": 24.375
    }]

    result = verify_claims(sample_report, claims)

    assert result["all_verified"] is True


def test_invalid_evidence_path(sample_report):
    claims = [{
        "claim": "The mean salary is 50000.",
        "evidence_path": "profile.column_profiles.salary.statistics.mean",
        "value": 50000
    }]

    result = verify_claims(sample_report, claims)

    assert result["rejected_count"] == 1
    assert result["all_verified"] is False


def test_missing_claim_fields(sample_report):
    claims = [{
        "claim": "The mean age is 24.375."
    }]

    result = verify_claims(sample_report, claims)

    assert result["rejected_count"] == 1


def test_floating_point_tolerance():
    assert values_match(24.375, 24.37500001)


def test_boolean_does_not_match_number():
    assert values_match(True, 1) is False


def test_empty_claims_are_not_verified(sample_report):
    result = verify_claims(sample_report, [])

    assert result["all_verified"] is False
    assert result["rejected_count"] == 1


def test_mixed_valid_and_invalid_claims(sample_report):
    claims = [
        {
            "claim": "Mean age is 24.375.",
            "evidence_path": "profile.column_profiles.age.statistics.mean",
            "value": 24.375
        },
        {
            "claim": "Mean quantity is 100.",
            "evidence_path": "profile.column_profiles.quantity.statistics.mean",
            "value": 100
        }
    ]

    result = verify_claims(sample_report, claims)

    assert result["verified_count"] == 1
    assert result["rejected_count"] == 1


def test_verification_summary(sample_report):
    claims = [{
        "claim": "Mean age is 100.",
        "evidence_path": "profile.column_profiles.age.statistics.mean",
        "value": 100
    }]

    result = verify_claims(sample_report, claims)
    summary = create_verification_summary(result)

    assert "UNSUPPORTED CLAIMS DETECTED" in summary
    assert "Mean age is 100." in summary


def test_invalid_report():
    with pytest.raises(TypeError):
        verify_claims([], [])


def test_invalid_claim_collection(sample_report):
    with pytest.raises(TypeError):
        verify_claims(sample_report, "not a list")
