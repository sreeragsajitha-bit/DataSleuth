
import math
from numbers import Real


def get_evidence_value(report, evidence_path):
    """
    Retrieve evidence using dot-separated dictionary keys
    and numeric list indexes.

    Examples:
        profile.column_profiles.age.statistics.mean
        insights.1.evidence.skewness
    """

    if not isinstance(evidence_path, str) or not evidence_path.strip():
        raise ValueError("Evidence path must be a non-empty string.")

    current = report

    for part in evidence_path.split("."):

        if isinstance(current, dict):
            if part not in current:
                raise KeyError(
                    f"Evidence path not found: {evidence_path}"
                )

            current = current[part]

        elif isinstance(current, list):
            if not part.isdigit():
                raise KeyError(
                    f"Expected a numeric list index at '{part}' "
                    f"in path: {evidence_path}"
                )

            index = int(part)

            if index >= len(current):
                raise KeyError(
                    f"List index {index} is out of range "
                    f"in path: {evidence_path}"
                )

            current = current[index]

        else:
            raise KeyError(
                f"Cannot navigate through value at '{part}' "
                f"in path: {evidence_path}"
            )

    return current


def values_match(expected, supplied, tolerance=1e-6):
    """Compare AI-supplied values with Python evidence."""

    if isinstance(expected, bool) or isinstance(supplied, bool):
        return (
            isinstance(expected, bool)
            and isinstance(supplied, bool)
            and expected == supplied
        )

    if isinstance(expected, Real) and isinstance(supplied, Real):
        return math.isclose(
            float(expected),
            float(supplied),
            rel_tol=tolerance,
            abs_tol=tolerance
        )

    return expected == supplied


def verify_claims(report, claims):
    """
    Verify structured AI claims against the original Python report.

    Each claim must contain:
        claim
        evidence_path
        value
    """

    if not isinstance(report, dict):
        raise TypeError("Report must be a dictionary.")

    if not isinstance(claims, list):
        raise TypeError("Claims must be a list.")

    verified_claims = []
    rejected_claims = []

    if not claims:
        rejected_claims.append({
            "claim": "No auditable claims supplied",
            "reason": (
                "The AI returned no evidence-backed claims. "
                "A dataset analysis response must contain claims."
            )
        })

    for claim in claims:

        if not isinstance(claim, dict):
            rejected_claims.append({
                "claim": str(claim),
                "reason": "Claim must be a dictionary."
            })
            continue

        claim_text = claim.get("claim", "Unnamed claim")
        evidence_path = claim.get("evidence_path")
        supplied_value = claim.get("value")

        if evidence_path is None or "value" not in claim:
            rejected_claims.append({
                "claim": claim_text,
                "reason": (
                    "Claim must provide both evidence_path and value."
                )
            })
            continue

        try:
            expected_value = get_evidence_value(
                report,
                evidence_path
            )

        except (KeyError, ValueError) as error:
            rejected_claims.append({
                "claim": claim_text,
                "reason": str(error),
                "evidence_path": evidence_path
            })
            continue

        if not values_match(expected_value, supplied_value):
            rejected_claims.append({
                "claim": claim_text,
                "reason": (
                    f"Value mismatch. Python evidence contains "
                    f"{expected_value!r}, but AI supplied "
                    f"{supplied_value!r}."
                ),
                "evidence_path": evidence_path,
                "expected_value": expected_value,
                "supplied_value": supplied_value
            })
            continue

        verified_claims.append({
            "claim": claim_text,
            "evidence_path": evidence_path,
            "verified_value": expected_value
        })

    return {
        "verified": verified_claims,
        "rejected": rejected_claims,
        "total_claims": len(claims),
        "verified_count": len(verified_claims),
        "rejected_count": len(rejected_claims),
        "all_verified": (
            len(rejected_claims) == 0 and len(claims) > 0
        )
    }


def create_verification_summary(result):
    """Create a readable verification summary."""

    lines = [
        "CLAIM VERIFICATION SUMMARY",
        "=" * 40,
        f"Total claims: {result['total_claims']}",
        f"Verified: {result['verified_count']}",
        f"Rejected: {result['rejected_count']}"
    ]

    if result["all_verified"]:
        lines.append(
            "\nAll submitted claims passed evidence verification."
        )

    else:
        lines.append("\nUNSUPPORTED CLAIMS DETECTED:")

        for item in result["rejected"]:
            lines.append(f"\n- {item['claim']}")
            lines.append(f"  Reason: {item['reason']}")

    return "\n".join(lines)
