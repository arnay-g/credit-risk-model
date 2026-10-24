from preprocessing import get_splits

path = "data/application_train.csv"

X_train, X_val, X_test, y_train, y_val, y_test = get_splits(path)

print(f"X_train shape: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
