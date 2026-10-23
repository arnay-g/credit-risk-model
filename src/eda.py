import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# folder for storing the plots
Path("reports/figures").mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/application_train.csv")

print("Shape:", df.shape)

# we see that data is heavily imbalanced - class imbalance
print("\nTarget balance:")
print(df["TARGET"].value_counts(normalize=True))

print("\nTop 20 missing columns:")
print(df.isna().mean().sort_values(ascending=False).head(20))

print("\nColumn types:")
print(df.dtypes.value_counts())


print(df["DAYS_EMPLOYED"])
df["TARGET"].value_counts().plot(kind="bar")
plt.title("Default vs non-default")
plt.ylabel("Number of training instances")
plt.savefig("reports/figures/target_balance.png", bbox_inches="tight")
plt.close()