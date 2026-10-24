import pandas as pd

df = pd.read_csv("data/application_train.csv")

non_pensioners = df[df["NAME_INCOME_TYPE"] != "Pensioner"]
missing = non_pensioners["OCCUPATION_TYPE"].isna()

print("default rate when missing:    ", non_pensioners.loc[missing, "TARGET"].mean())
print("default rate when not missing:", non_pensioners.loc[~missing, "TARGET"].mean())