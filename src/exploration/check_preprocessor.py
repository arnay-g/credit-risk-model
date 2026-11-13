from preprocessing import get_splits, build_preprocessor


X_train, X_val, X_test, y_train, y_val, y_test = get_splits("data/application_train.csv")

processor = build_preprocessor(X_train)

# processor.set_output(transform = 'pandas')
# first we fit the model to the X_train data, then we transform it
X_train_t = processor.fit_transform(X_train)

X_val_t = processor.transform(X_val)

# Remember: X_train_t and X_val_t are numpy arrays and not pandas dataframes

print(f"X_train's shape: {X_train_t.shape}\nX_val's shape: {X_val_t.shape}")

# new_col_names = processor.get_feature_names_out()
# idx = list(new_col_names).index("text__NAME_CONTRACT_TYPE_Cash loans")
# print(idx)
# print(X_train_t["text__NAME_CONTRACT_TYPE_Cash loans"])

