import os
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def interpret_rules(rules_df, top_n=5):
    print("\nInterpretation of strong rules:")
    top_rules = rules_df.head(top_n)
    for i, row in top_rules.iterrows():
        lhs = ", ".join(list(row["antecedents"]))
        rhs = ", ".join(list(row["consequents"]))
        print(
            f"Rule {i}: If a basket has [{lhs}], it often also has [{rhs}] "
            f"(confidence={row['confidence']:.2f}, lift={row['lift']:.2f})."
        )


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "E_Commerce_Data.csv"),
                 encoding="ISO-8859-1")
df = df.dropna(subset=["InvoiceNo", "Description"])
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

transactions = df.groupby("InvoiceNo")["Description"].apply(list)
top_products = df["Description"].value_counts().head(10)

print("Top-selling products:\n", top_products)

plt.figure(figsize=(10, 5))
top_products.plot(kind="bar", color="teal")
plt.title("Top 10 Products - E-Commerce")
plt.xlabel("Product")
plt.ylabel("Frequency")
plt.xticks(rotation=85)
plt.tight_layout()
plt.show()

te = TransactionEncoder()
encoded = te.fit(transactions).transform(transactions)
basket = pd.DataFrame(encoded, columns=te.columns_)

freq_items = apriori(basket, min_support=0.015, use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.35)
rules = rules.sort_values(["confidence", "lift"], ascending=False)

print("\nFrequent itemsets (top 10):\n", freq_items.sort_values(
    "support", ascending=False).head(10))
print("\nHigh-confidence/high-lift rules (top 10):\n",
      rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))
interpret_rules(rules, top_n=5)

plt.figure(figsize=(8, 5))
plt.scatter(rules["support"], rules["confidence"],
            s=rules["lift"] * 20, alpha=0.6, c="darkorange")
plt.title("Rule Distribution - E-Commerce")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.tight_layout()
plt.show()
