import pandas as pd

df = pd.read_parquet("../../data/processed/cleaned.parquet")
print(df.dtypes)
print(df.head())
print(df["text_len"].describe())
