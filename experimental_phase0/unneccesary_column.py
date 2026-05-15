import pandas as pd
df = pd.read_csv("data/train_normal.csv")

# Identify columns where EVERY row is 0.0 or constant
empty_info_cols = [col for col in df.columns if df[col].nunique() <= 1]

print(f"Found {len(empty_info_cols)} constant/empty columns:")
print(empty_info_cols)