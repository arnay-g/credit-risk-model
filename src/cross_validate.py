import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from preprocessing import get_splits, build_preprocessor

X_train, X_val, X_test, y_train, y_val, y_test = get_splits()

models = {
    "logreg": LogisticRegression(max_iter=1000),
    "lightgbm": LGBMClassifier(n_estimators=300, learning_rate=0.05,
                               random_state=42, verbose=-1),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in models.items():
    pipe = Pipeline([("prep", build_preprocessor(X_train)), ("clf", clf)])
    res = cross_validate(pipe, X_train, y_train, cv=cv,
                         scoring=["roc_auc", "average_precision"])
    print(name)
    print("  ROC-AUC:", res["test_roc_auc"].round(4),
          "mean", res["test_roc_auc"].mean().round(4),
          "std", res["test_roc_auc"].std().round(4))
    print("  PR-AUC: ", res["test_average_precision"].round(4),
          "mean", res["test_average_precision"].mean().round(4))
          