import os
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "Supermarket_Simulation_Dataset.csv"))

transactions = df.groupby("TransactionID")["Items"].apply(list)
top_items = df["Items"].value_counts().head(10)

print("Top products:\n", top_items)

plt.figure(figsize=(10, 5))
top_items.plot(kind="bar", color="cornflowerblue")
plt.title("Top Supermarket Products")
plt.xlabel("Product")
plt.ylabel("Frequency")
plt.xticks(rotation=35)
plt.tight_layout()
plt.show()

te = TransactionEncoder()
basket = pd.DataFrame(te.fit(transactions).transform(
    transactions), columns=te.columns_)

freq_items = apriori(basket, min_support=0.04, use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.55)
rules = rules.sort_values(["lift", "confidence"], ascending=False)

print("\nStrong supermarket rules:\n", rules[[
      "antecedents", "consequents", "support", "confidence", "lift"]].head(10))
print("\nPurchasing pattern interpretation:")
for i, row in rules.head(5).iterrows():
    a = ", ".join(list(row["antecedents"]))
    c = ", ".join(list(row["consequents"]))
    print(f"Rule {i}: If customers buy [{a}], they tend to buy [{c}] too.")

plt.figure(figsize=(8, 5))
plt.scatter(rules["support"], rules["confidence"],
            s=rules["lift"] * 30, alpha=0.65, c="royalblue")
plt.title("Supermarket Simulation Rules")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.tight_layout()
plt.show()
