import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

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


# build preprocessor for handling missing values, one hot encoding of text and num columns
def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(exclude = ["object", "string"]).columns.to_list()
    text_cols = X.select_dtypes(include = ["object", "string"]).columns.to_list()

    # creating required pipelines for num/text columns
    # add_indicator creates a column that indicates whether the row had a missing value or not for a given column

    num_pipe = SimpleImputer(strategy = "median", add_indicator = True)

    text_pipe = Pipeline([("fill", SimpleImputer(strategy = 'constant', fill_value = "missing")),
                          ("one-hot-encode", OneHotEncoder(handle_unknown="ignore"))])

    return ColumnTransformer([("num", num_pipe, num_cols),
                              ("text", text_pipe, text_cols)])



