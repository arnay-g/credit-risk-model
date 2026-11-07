from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score
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

for t in [0.5, 0.3, 0.2, 0.1, 0.08]:
    pred = (proba >= t).astype(int)
    print(f"threshold {t}: flagged {pred.sum()}, "
          f"precision {precision_score(y_val, pred, zero_division=0):.3f}, "
          f"recall {recall_score(y_val, pred):.3f}")