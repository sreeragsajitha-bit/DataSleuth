
import pandas as pd


ALLOWED_AGGREGATIONS = {
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "count",
}

ALLOWED_METRICS = {"revenue"}

ALLOWED_OPERATIONS = {"aggregate"}

ALLOWED_SORT_ORDERS = {"asc", "desc"}


def prepare_dataset(df):
    """
    Create a safe working copy of the dataset.
    Add a revenue column when quantity and unit_price are available.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    prepared_df = df.copy()

    if "quantity" in prepared_df.columns and "unit_price" in prepared_df.columns:
        if (
            pd.api.types.is_numeric_dtype(prepared_df["quantity"])
            and pd.api.types.is_numeric_dtype(prepared_df["unit_price"])
        ):
            prepared_df["revenue"] = (
                prepared_df["quantity"] * prepared_df["unit_price"]
            )

    return prepared_df


def validate_query_plan(df, plan):
    """
    Validate a structured query plan before execution.
    Only explicitly allowed operations are accepted.
    """
    if not isinstance(plan, dict):
        raise ValueError("Query plan must be a dictionary.")

    allowed_keys = {
        "operation",
        "metric",
        "aggregation",
        "group_by",
        "sort_by",
        "sort_order",
        "limit",
    }

    unknown_keys = set(plan.keys()) - allowed_keys

    if unknown_keys:
        raise ValueError(f"Unsupported query fields: {unknown_keys}")

    operation = plan.get("operation")

    if operation not in ALLOWED_OPERATIONS:
        raise ValueError("Unsupported operation.")

    metric = plan.get("metric")

    if not isinstance(metric, str):
        raise ValueError("Metric must be a string.")

    if metric not in df.columns and metric not in ALLOWED_METRICS:
        raise ValueError(f"Unknown metric: {metric}")

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' is unavailable in this dataset.")

    aggregation = plan.get("aggregation")

    if aggregation not in ALLOWED_AGGREGATIONS:
        raise ValueError("Unsupported aggregation.")

    if aggregation != "count":
        if not pd.api.types.is_numeric_dtype(df[metric]):
            raise ValueError(
                f"Aggregation '{aggregation}' requires a numeric column."
            )

    group_by = plan.get("group_by")

    if group_by is not None:
        if not isinstance(group_by, str):
            raise ValueError("group_by must be a column name.")

        if group_by not in df.columns:
            raise ValueError(f"Unknown grouping column: {group_by}")

    sort_by = plan.get("sort_by")

    if sort_by is not None:
        if not isinstance(sort_by, str):
            raise ValueError("sort_by must be a string.")

        valid_sort_columns = {metric}

        if group_by is not None:
            valid_sort_columns.add(group_by)

        if sort_by not in valid_sort_columns:
            raise ValueError(f"Invalid sort column: {sort_by}")

    sort_order = plan.get("sort_order", "desc")

    if sort_order not in ALLOWED_SORT_ORDERS:
        raise ValueError("sort_order must be 'asc' or 'desc'.")

    limit = plan.get("limit")

    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer.")

    return True


def execute_query(df, plan):
    """
    Validate and execute a structured query plan.
    Returns a structured result dictionary.
    """
    prepared_df = prepare_dataset(df)

    validate_query_plan(prepared_df, plan)

    metric = plan["metric"]
    aggregation = plan["aggregation"]
    group_by = plan.get("group_by")
    sort_by = plan.get("sort_by")
    sort_order = plan.get("sort_order", "desc")
    limit = plan.get("limit")

    if group_by:
        grouped = prepared_df.groupby(
            group_by,
            dropna=False
        )[metric]

        result = grouped.agg(aggregation).reset_index()

        if sort_by:
            result = result.sort_values(
                by=sort_by,
                ascending=(sort_order == "asc")
            )

        if limit is not None:
            result = result.head(limit)

        return {
            "success": True,
            "operation": "aggregate",
            "metric": metric,
            "aggregation": aggregation,
            "group_by": group_by,
            "result": result.to_dict(orient="records"),
            "row_count": len(result),
        }

    value = prepared_df[metric].agg(aggregation)

    return {
        "success": True,
        "operation": "aggregate",
        "metric": metric,
        "aggregation": aggregation,
        "group_by": None,
        "result": value.item() if hasattr(value, "item") else value,
    }


if __name__ == "__main__":
    from dataset_loader import load_dataset

    df = load_dataset("data/raw/sales.csv")

    query_plan = {
        "operation": "aggregate",
        "metric": "revenue",
        "aggregation": "sum",
        "group_by": "product",
        "sort_by": "revenue",
        "sort_order": "desc",
        "limit": 3,
    }

    output = execute_query(df, query_plan)

    print("\nDATASLEUTH QUERY ENGINE")
    print("=" * 40)
    print(output)
