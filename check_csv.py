import pandas as pd

# Path to CSV
csv_path = "data/creditcard_sample.csv"

# Load CSV
df = pd.read_csv(csv_path)

# Print info
print("Columns:", df.columns.tolist())
print("Number of rows:", len(df))
if "Class" in df.columns:
    print("NaN in Class:", df['Class'].isna().sum())
    print("Class value counts:\n", df['Class'].value_counts())
else:
    print("No column named 'Class' in CSV")
