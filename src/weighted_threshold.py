import numpy as np
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from preprocessing import get_splits, build_preprocessor

X_train, X_val, X_test, y_train, y_val, y_test = get_splits()

model = Pipeline([
    ("prep", build_preprocessor(X_train)),
    ("clf", LGBMClassifier(n_estimators=300, learning_rate=0.05,
                           class_weight="balanced",
                           random_state=42, verbose=-1)),
])
model.fit(X_train, y_train)
proba = model.predict_proba(X_val)[:, 1]

print("mean predicted probability:", proba.mean().round(3))
print("share above 0.5:", (proba >= 0.5).mean().round(3))

COST_FN, COST_FP = 5, 1
print("Approve everyone:", COST_FN * (y_val == 1).sum())
for t in np.arange(0.55, 0.95, 0.025):
    pred = (proba >= t).astype(int)
    fn = ((pred == 0) & (y_val == 1)).sum()
    fp = ((pred == 1) & (y_val == 0)).sum()
    print(f"threshold {t:.2f}: cost {COST_FN * fn + COST_FP * fp}")