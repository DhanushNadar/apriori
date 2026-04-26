import os
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "Smart_Home_Dataset.csv"))

transactions = df.groupby("UserID")["DevicesUsed"].apply(list)
top_devices = df["DevicesUsed"].value_counts().head(10)

print("Most used devices:\n", top_devices)

plt.figure(figsize=(10, 5))
top_devices.plot(kind="bar", color="lightslategray")
plt.title("Top Smart Home Devices")
plt.xlabel("Device")
plt.ylabel("Frequency")
plt.xticks(rotation=35)
plt.tight_layout()
plt.show()

te = TransactionEncoder()
basket = pd.DataFrame(te.fit(transactions).transform(
    transactions), columns=te.columns_)

freq_items = apriori(basket, min_support=0.02, use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.36)
rules = rules.sort_values(["lift", "confidence"], ascending=False)

print("\nDevice usage combinations:\n", rules[[
      "antecedents", "consequents", "support", "confidence", "lift"]].head(10))
print("\nUsage pattern interpretation:")
for i, row in rules.head(5).iterrows():
    a = ", ".join(list(row["antecedents"]))
    c = ", ".join(list(row["consequents"]))
    print(f"Rule {i}: Homes using [{a}] are likely to use [{c}] as well.")

plt.figure(figsize=(8, 5))
plt.scatter(rules["support"], rules["confidence"],
            s=rules["lift"] * 30, alpha=0.65, c="dimgray")
plt.title("Smart Home Rule Distribution")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.tight_layout()
plt.show()
