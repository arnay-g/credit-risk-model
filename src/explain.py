import shap
import pandas as pd
from lightgbm import LGBMClassifier
from preprocessing import get_splits, build_preprocessor
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

# print(X_val_t.shape)
# print(X_val_t.columns[:10].tolist())

Path("reports/figures").mkdir(parents=True, exist_ok=True)

sample = X_val_t.sample(5000, random_state=42)
explainer = shap.TreeExplainer(clf)
shap_values = explainer(sample)

vals = shap_values.values
if vals.ndim == 3:
    vals = vals[:, :, 1]

importance = pd.Series(np.abs(vals).mean(axis=0), index=sample.columns)
print(importance.sort_values(ascending=False).head(15))

shap.summary_plot(vals, sample, show=False, max_display=15)
plt.savefig("reports/figures/shap_summary.png", bbox_inches="tight")
plt.close()


base = shap_values.base_values
if base.ndim == 2:
    base = base[:, 1]

raw = base + vals.sum(axis=1)
prob = 1 / (1 + np.exp(-raw))

print("max difference:", np.abs(prob - clf.predict_proba(sample)[:, 1]).max())