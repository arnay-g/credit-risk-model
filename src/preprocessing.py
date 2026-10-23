import numpy as np
import pandas as pd

def fix_days_employed(df: pd.DataFrame):
    # work on copy so that we dont change original data
    df = df.copy()

    # create a column with the anomalies for days_employed as 1 and the others as 0 
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan) 

    return df



