
import pandas as pd

from data_quality import check_data_quality


# Create a deliberately problematic dataset
data = {
    "customer_id": [101, 102, 102, 104],
    "age": [21, None, None, 25],
    "quantity": [2, 1, 1, -3],
    "unit_price": [500, 1000, 1000, -200]
}

df = pd.DataFrame(data)

# Run quality checks
issues = check_data_quality(df)

print("\n========== TEST RESULTS ==========")

for issue in issues:
    print("-", issue)

print("==================================")
