import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from preprocessing import get_splits, build_preprocessor

X_train, X_val, X_test, y_train, y_val, y_test = get_splits()

model = Pipeline([
    ("prep", build_preprocessor(X_train)),
    ("clf", LGBMClassifier(n_estimators=300, learning_rate=0.05,
                           random_state=42, verbose=-1)),
])
model.fit(X_train, y_train)

proba = model.predict_proba(X_val)[:, 1]
pred = (proba >= 0.15).astype(int)
actual = y_val.values

results = X_val.copy()
results["TARGET"] = actual
results["proba"] = proba
results["outcome"] = "TN"
results.loc[(pred == 1) & (actual == 1), "outcome"] = "TP"
results.loc[(pred == 1) & (actual == 0), "outcome"] = "FP"
results.loc[(pred == 0) & (actual == 1), "outcome"] = "FN"

print(results["outcome"].value_counts())


fn_rows = results[results["outcome"] == "FN"]
tp_rows = results[results["outcome"] == "TP"]

for col in ["NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_CONTRACT_TYPE"]:
    print(f"\n--- {col} ---")
    fn_share = fn_rows[col].value_counts(normalize=True).round(3)
    tp_share = tp_rows[col].value_counts(normalize=True).round(3)
    print(pd.DataFrame({"FN share": fn_share, "TP share": tp_share}))

for col in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
    print(f"\n{col}: FN mean {fn_rows[col].mean():.2f}, TP mean {tp_rows[col].mean():.2f}")