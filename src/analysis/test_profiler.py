
from dataset_loader import load_dataset
from dataset_profiler import profile_dataset

# Load the dataset
df = load_dataset("data/raw/sales.csv")

# Generate the profile
result = profile_dataset(df)

# Access individual results
print("Total rows:", result["total_rows"])
print("Total columns:", result["total_columns"])
print("Missing values:", result["total_missing_values"])
print("Duplicate rows:", result["duplicate_rows"])

# Access age-specific missing values
print("Missing ages:", result["missing_values"]["age"])
