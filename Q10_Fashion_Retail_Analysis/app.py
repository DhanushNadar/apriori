import os
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "Fashion_Retail_Dataset.csv"))

transactions = df.groupby("TransactionID")["ClothingItem"].apply(list)
top_items = df["ClothingItem"].value_counts().head(10)

print("Most purchased clothing items:\n", top_items)

plt.figure(figsize=(10, 5))
top_items.plot(kind="bar", color="mediumorchid")
plt.title("Top 10 Clothing Items")
plt.xlabel("Clothing Item")
plt.ylabel("Count")
plt.xticks(rotation=40)
plt.tight_layout()
plt.show()

te = TransactionEncoder()
basket = pd.DataFrame(te.fit(transactions).transform(
    transactions), columns=te.columns_)

freq_items = apriori(basket, min_support=0.018, use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.42)
rules = rules.sort_values(["lift", "confidence"], ascending=False)

print("\nStrong fashion rules:\n", rules[[
      "antecedents", "consequents", "support", "confidence", "lift"]].head(10))
print("\nFashion buying behavior:")
for i, row in rules.head(5).iterrows():
    a = ", ".join(list(row["antecedents"]))
    c = ", ".join(list(row["consequents"]))
    print(
        f"Rule {i}: Buying [{a}] is associated with [{c}] (lift={row['lift']:.2f}).")

plt.figure(figsize=(8, 5))
plt.scatter(rules["support"], rules["confidence"],
            s=rules["lift"] * 25, alpha=0.6, c="deeppink")
plt.title("Fashion Rules Scatter")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.tight_layout()
plt.show()
