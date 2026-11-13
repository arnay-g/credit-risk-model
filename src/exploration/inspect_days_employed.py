import pandas as pd
from preprocessing import fix_days_employed
df = pd.read_csv('data/application_train.csv')

anomaly = df["DAYS_EMPLOYED"] > 10000 

print(f"Number of Anomalies: {anomaly.sum()}")
print(f"Share of all rows: {anomaly.mean()}")

print(f"Default rate, for the rows with anomaly in DAYS_EMPLOYED: {df.loc[anomaly, 'TARGET'].mean()}")
print(f"Default rate, for the rows without anomaly in DAYS_EMPLOYED: {df.loc[~anomaly, 'TARGET'].mean()}")

# normal days_employed summary
print("Normal days employed summary: ")
print(df.loc[~anomaly, "DAYS_EMPLOYED"].describe())

print("\nIncome Types among placeholder rows: ")
print(df.loc[anomaly, 'NAME_INCOME_TYPE'].value_counts()[0:2])

clean = fix_days_employed(df)
print(clean["DAYS_EMPLOYED"].isna().sum())
print(clean["DAYS_EMPLOYED_ANOM"].sum())
