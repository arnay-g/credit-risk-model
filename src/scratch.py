from preprocessing import get_splits

path = "data/application_train.csv"

X_train, X_val, X_test, y_train, y_val, y_test = get_splits(path)

text_cols = X_train.select_dtypes(include = ["object", "string"]).columns.tolist()
num_cols = X_train.select_dtypes(exclude = ["object", "string"]).columns.tolist()
print(len(text_cols), len(num_cols))
print(f"X_train shape: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

