import pandas as pd

df = pd.read_csv('data/application_train.csv')

print(pd.crosstab(df['FLAG_OWN_CAR'], df['OWN_CAR_AGE'].isna()))