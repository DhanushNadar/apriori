import os
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "Restaurant_Dataset.csv"))

transactions = df.groupby("OrderID")["MenuItem"].apply(list)
top_items = df["MenuItem"].value_counts().head(10)

print("Popular dishes:\n", top_items)

plt.figure(figsize=(9, 5))
top_items.plot(kind="bar", color="tomato")
plt.title("Top Menu Items")
plt.xlabel("Menu Item")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

te = TransactionEncoder()
basket = pd.DataFrame(te.fit(transactions).transform(
    transactions), columns=te.columns_)

freq_items = apriori(basket, min_support=0.02, use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.50)
rules = rules.sort_values(["lift", "confidence"], ascending=False)

print("\nFood combination rules:\n", rules[[
      "antecedents", "consequents", "support", "confidence", "lift"]].head(10))
print("\nCustomer preference interpretation:")
for i, row in rules.head(5).iterrows():
    a = ", ".join(list(row["antecedents"]))
    c = ", ".join(list(row["consequents"]))
    print(f"Rule {i}: Customers ordering [{a}] frequently add [{c}] too.")

plt.figure(figsize=(8, 5))
plt.scatter(rules["support"], rules["confidence"],
            s=rules["lift"] * 25, alpha=0.65, c="brown")
plt.title("Restaurant Rule Visualization")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.tight_layout()
plt.show()
