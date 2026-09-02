import pandas as pd

df = pd.read_csv("data/prompts.csv")

print("Dataset loaded successfully!")
print()

print("First 5 rows:")
print(df.head())

print()
print("Dataset shape:")
print(df.shape)

print()
print("Label distribution:")
print(df["label"].value_counts())