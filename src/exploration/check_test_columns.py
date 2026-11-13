import pandas as pd

df = pd.read_csv('data/application_train.csv')

text_cols = df.select_dtypes(include=["object", "string"]).columns

# print(f"text cols: {text_cols}")

for col in text_cols:
    print(f"{col}: {df[col].nunique()} unique values")