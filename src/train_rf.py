from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from preprocessing import get_splits, build_preprocessor

X_train, X_val, X_test, y_train, y_val, y_test = get_splits()

model = Pipeline([
    ("prep", build_preprocessor(X_train)),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=50,
        n_jobs=-1,
        random_state=42,
    )),
])
model.fit(X_train, y_train)

for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val)]:
    proba = model.predict_proba(X)[:, 1]
    print(name, "ROC-AUC:", roc_auc_score(y, proba))
    print(name, "PR-AUC: ", average_precision_score(y, proba))