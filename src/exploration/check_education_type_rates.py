import pandas as pd

df = pd.read_csv("data/application_train.csv")

print(df.groupby("NAME_EDUCATION_TYPE")["TARGET"].agg(["count", "mean"]))