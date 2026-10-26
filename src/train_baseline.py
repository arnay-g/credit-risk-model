from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from preprocessing import build_preprocessor, get_splits


X_train, X_val, X_test, y_train, y_val, y_test = get_splits("data/application_train.csv")


preprocessor = build_preprocessor(X_train)

# preprocessor.fit_transform(X_train)
# preprocessor.transform(X_val)
# preprocessor.transform(X_test)

model = Pipeline([("prep", preprocessor),
                  ("classifier", LogisticRegression(max_iter=1000))])

model.fit(X_train, y_train)

# predict_proba return a numpy 2D array in which the first column is of class 0 and second column is of class 1
probability = model.predict_proba(X_val)[:, 1]

print("ROC-AUC:", roc_auc_score(y_val, probability))
print("PR-AUC: ", average_precision_score(y_val, probability))
