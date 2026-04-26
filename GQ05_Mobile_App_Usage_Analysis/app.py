import os
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "App_Usage_Dataset.csv"))

transactions = df.groupby("UserID")["AppsUsed"].apply(list)
top_apps = df["AppsUsed"].value_counts().head(10)

print("Most used apps:\n", top_apps)

plt.figure(figsize=(10, 5))
top_apps.plot(kind="bar", color="olive")
plt.title("Top Mobile Apps")
plt.xlabel("App")
plt.ylabel("Frequency")
plt.xticks(rotation=35)
plt.tight_layout()
plt.show()

te = TransactionEncoder()
basket = pd.DataFrame(te.fit(transactions).transform(
    transactions), columns=te.columns_)

freq_items = apriori(basket, min_support=0.015, use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.30)
rules = rules.sort_values(["lift", "confidence"], ascending=False)

print("\nApp combination rules:\n", rules[[
      "antecedents", "consequents", "support", "confidence", "lift"]].head(10))
print("\nUser behavior interpretation:")
for i, row in rules.head(5).iterrows():
    a = ", ".join(list(row["antecedents"]))
    c = ", ".join(list(row["consequents"]))
    print(f"Rule {i}: Users of [{a}] are likely to use [{c}] too.")

plt.figure(figsize=(8, 5))
plt.scatter(rules["support"], rules["confidence"],
            s=rules["lift"] * 25, alpha=0.65, c="darkolivegreen")
plt.title("App Usage Rules")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.tight_layout()
plt.show()
