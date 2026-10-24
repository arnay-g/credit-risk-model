import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import fix_days_employed


df = pd.read_csv('data/application_train.csv')
df = fix_days_employed(df)

X = df.drop(labels = ["TARGET", "SK_ID_CURR"], axis = 1)
y = df["TARGET"]

# get training data and the data which will further be split into test and validation
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = 40)


# get test data and the validation data 
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size = 0.5, stratify = y_temp, random_state = 40)

for name, labels in [("train: ", y_train), ("validation: ", y_val), ("test", y_test)]:
    print(name, len(labels), labels.mean())

print(X_train.shape)