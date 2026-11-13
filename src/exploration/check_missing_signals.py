import pandas as pd

df = pd.read_csv("data/application_train.csv")

for col in ["EXT_SOURCE_1", "OCCUPATION_TYPE", "APARTMENTS_AVG"]:
    missing = df[col].isna()
    print(col)
    print("  default rate when missing:    ", df.loc[missing, "TARGET"].mean())
    print("  default rate when not missing:", df.loc[~missing, "TARGET"].mean())


print(f"NAME_INCOME_TYPE\n {df["NAME_INCOME_TYPE"].value_counts()}")

print(f"OCCUPATION_TYPE\n {df["OCCUPATION_TYPE"].value_counts()}")
print(f"NA      {df["OCCUPATION_TYPE"].isna().sum()}")

print(pd.crosstab(df["NAME_INCOME_TYPE"], df['OCCUPATION_TYPE'].isna()))

print("rate of default of pensioners: " , df.loc[df["NAME_INCOME_TYPE"] == "Pensioner", "TARGET"].mean())
print("rate of default of non-pensioners: " , df.loc[df["NAME_INCOME_TYPE"] != "Pensioner", "TARGET"].mean())