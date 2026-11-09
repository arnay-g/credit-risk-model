import shap
import pandas as pd
from lightgbm import LGBMClassifier
from preprocessing import get_splits, build_preprocessor
import re

X_train, X_val, X_test, y_train, y_val, y_test = get_splits("data/application_train.csv")

prep = build_preprocessor(X_train)

prep.set_output(transform = 'pandas')

X_train_t = prep.fit_transform(X_train)

X_val_t = prep.transform(X_val)

def clean_names(df):
    df.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", c) for c in df.columns]
    return df

X_train_t = clean_names(X_train_t)
X_val_t = clean_names(X_val_t)
assert X_train_t.columns.is_unique

clf = LGBMClassifier(n_estimators = 300, learning_rate = 0.05, random_state = 42, verbose = -1)

clf.fit(X_train_t, y_train)

print(X_val_t.shape)
print(X_val_t.columns[:10].tolist())
