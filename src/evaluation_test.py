from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
)
from preprocessing import get_splits, build_preprocessor


X_train, X_val, X_test, y_train, y_val, y_test = get_splits()

# Same model settings used during validation.
model = Pipeline([
    ("prep", build_preprocessor(X_train)),
    ("clf", LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
    )),
])

model.fit(X_train, y_train)

# Threshold chosen using validation data.
THRESHOLD = 0.15
COST_FN, COST_FP = 5, 1

proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= THRESHOLD).astype(int)
actual = y_test.values

tn = ((actual == 0) & (pred == 0)).sum()
fp = ((actual == 0) & (pred == 1)).sum()
fn = ((actual == 1) & (pred == 0)).sum()
tp = ((actual == 1) & (pred == 1)).sum()

model_cost = COST_FN * fn + COST_FP * fp
approve_all_cost = COST_FN * (actual == 1).sum()
cost_reduction = (approve_all_cost - model_cost) / approve_all_cost

print("Test ROC-AUC:", roc_auc_score(actual, proba))
print("Test average precision:", average_precision_score(actual, proba))
print("Threshold:", THRESHOLD)
print("Precision:", precision_score(actual, pred, zero_division=0))
print("Recall:", recall_score(actual, pred, zero_division=0))
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print("Flagged applicants:", pred.sum())
print("Model cost:", model_cost)
print("Approve-everyone cost:", approve_all_cost)
print(f"Cost reduction: {cost_reduction:.2%}")