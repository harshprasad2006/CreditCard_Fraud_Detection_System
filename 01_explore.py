import pandas as pd

df = pd.read_csv("creditcard.csv")

print("Shape:", df.shape)
print("\nColumn info:")
print(df.info())
print("\nMissing values:", df.isnull().sum().sum())
print("\nFirst 5 rows:")
print(df.head())
class_counts = df["Class"].value_counts()
class_pct = df["Class"].value_counts(normalize=True) * 100

print("Class counts:\n", class_counts)
print("\nClass percentage:\n", class_pct.round(4))
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.figure(figsize=(6, 5))
sns.countplot(data=df, x="Class")
plt.title("Class Distribution: Normal (0) vs Fraud (1)")
plt.yscale("log")  # log scale — otherwise the fraud bar is invisible
plt.ylabel("Count (log scale)")
plt.savefig("chart_class_imbalance.png", dpi=150)
plt.show()
print(df.groupby("Class")["Amount"].describe())

plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="Class", y="Amount", hue="Class", palette="Set2", legend=False)
plt.title("Transaction Amount: Normal vs Fraud")
plt.ylim(0, 500)  # zoom in — a few huge outliers otherwise squash the plot
plt.savefig("chart_amount_by_class.png", dpi=150)
plt.show()
df["Hour"] = (df["Time"] // 3600) % 24  # convert seconds -> hour of day

fraud_by_hour = df[df["Class"] == 1].groupby("Hour").size()
normal_by_hour = df[df["Class"] == 0].groupby("Hour").size()

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(normal_by_hour.index, normal_by_hour.values, color="steelblue", label="Normal")
ax1.set_ylabel("Normal transaction count", color="steelblue")
ax2 = ax1.twinx()
ax2.plot(fraud_by_hour.index, fraud_by_hour.values, color="red", marker="o", label="Fraud")
ax2.set_ylabel("Fraud transaction count", color="red")
plt.title("Transactions by Hour of Day: Normal vs Fraud")
plt.savefig("chart_fraud_by_hour.png", dpi=150)
plt.show()
correlations = df.corr()["Class"].sort_values(ascending=False)
print("Top features correlated with fraud:\n", correlations.head(6))
print("\nBottom (most negatively correlated):\n", correlations.tail(6))