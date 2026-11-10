
from filter_engine import apply_filters
from query_engine import (
    prepare_dataset,
    validate_query_plan,
    execute_query,
)


QUERY_FIELDS = {
    "operation",
    "metric",
    "aggregation",
    "group_by",
    "sort_by",
    "sort_order",
    "limit",
}

ALLOWED_FIELDS = QUERY_FIELDS | {
    "filters",
    "logical_operator",
}


def validate_unified_plan(df, plan):
    """Validate a unified filtering and aggregation plan."""

    if not isinstance(plan, dict):
        raise ValueError("Unified plan must be a dictionary.")

    unknown_fields = set(plan) - ALLOWED_FIELDS

    if unknown_fields:
        raise ValueError(
            f"Unsupported unified plan fields: {sorted(unknown_fields)}"
        )

    prepared_df = prepare_dataset(df)

    filters = plan.get("filters", [])
    logical_operator = plan.get("logical_operator", "and")

    # Reuse the existing filter validator.
    from filter_engine import validate_filters

    validate_filters(
        prepared_df,
        filters,
        logical_operator,
    )

    query_plan = {
        key: value
        for key, value in plan.items()
        if key in QUERY_FIELDS
    }

    validate_query_plan(prepared_df, query_plan)

    return True


def execute_unified_plan(df, plan):
    """
    Execute a validated unified analytics plan.

    Python performs all filtering and calculations.
    The original dataset is not modified.
    """

    validate_unified_plan(df, plan)

    prepared_df = prepare_dataset(df)

    filters = plan.get("filters", [])
    logical_operator = plan.get("logical_operator", "and")

    filtered_df = apply_filters(
        prepared_df,
        filters,
        logical_operator,
    )

    query_plan = {
        key: value
        for key, value in plan.items()
        if key in QUERY_FIELDS
    }

    query_result = execute_query(
        filtered_df,
        query_plan,
    )

    return {
        "success": True,
        "unified_plan": plan,
        "original_row_count": len(prepared_df),
        "filtered_row_count": len(filtered_df),
        "query_result": query_result,
    }
