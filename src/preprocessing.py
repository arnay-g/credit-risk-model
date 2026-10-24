import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def fix_days_employed(df: pd.DataFrame):
    # work on copy so that we dont change original data
    df = df.copy()

    # create a column with the anomalies for days_employed as 1 and the others as 0 
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan) 

    return df

def get_splits(path = "data/application_train.csv"):
    df = pd.read_csv(path)
    df = fix_days_employed(df)

    X = df.drop(columns = ["TARGET", "SK_ID_CURR"])
    y = df["TARGET"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size = 0.3, stratify = y, random_state=40
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify = y_temp, random_state=40
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


