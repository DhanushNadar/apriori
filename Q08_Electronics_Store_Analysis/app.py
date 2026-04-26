import os
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "Electronics_Dataset.csv"))
df = df.dropna().drop_duplicates()

transactions = df.groupby("OrderID")["Product"].apply(list)
top_products = df["Product"].value_counts().head(10)

print("Top-selling electronics:\n", top_products)

plt.figure(figsize=(10, 5))
top_products.plot(kind="bar", color="cadetblue")
plt.title("Top Selling Electronics")
plt.xlabel("Product")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

te = TransactionEncoder()
basket = pd.DataFrame(te.fit(transactions).transform(
    transactions), columns=te.columns_)

freq_items = apriori(basket, min_support=0.03, use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.60)
rules = rules.sort_values(["lift", "confidence"], ascending=False)

print("\nHigh-lift rules:\n",
      rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))
print("\nCross-selling interpretation:")
for i, row in rules.head(5).iterrows():
    a = ", ".join(list(row["antecedents"]))
    c = ", ".join(list(row["consequents"]))
    print(
        f"Rule {i}: Bundling [{a}] with [{c}] is promising (lift={row['lift']:.2f}).")

plt.figure(figsize=(8, 5))
plt.scatter(rules["support"], rules["confidence"],
            s=rules["lift"] * 25, alpha=0.6, c="darkslategray")
plt.title("Electronics Rule Pattern")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.tight_layout()
plt.show()
