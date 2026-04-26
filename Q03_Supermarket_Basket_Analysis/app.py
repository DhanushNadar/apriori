import os
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def interpret_rules(rules_df, top_n=5):
    print("\nStrongest rules interpretation:")
    for idx, row in rules_df.head(top_n).iterrows():
        lhs = ", ".join(list(row["antecedents"]))
        rhs = ", ".join(list(row["consequents"]))
        print(
            f"Rule {idx}: [{lhs}] -> [{rhs}] | confidence={row['confidence']:.2f}, lift={row['lift']:.2f}")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "SuperMarket_Analysis.csv"))
df = df.dropna(subset=["Invoice ID", "Product line"])

transactions = df.groupby("Invoice ID")["Product line"].apply(list)
top_categories = df["Product line"].value_counts().head(10)

print("Top product categories:\n", top_categories)

plt.figure(figsize=(9, 5))
top_categories.plot(kind="bar", color="slateblue")
plt.title("Top Product Categories - Supermarket")
plt.xlabel("Product Line")
plt.ylabel("Frequency")
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()

te = TransactionEncoder()
encoded = te.fit(transactions).transform(transactions)
basket = pd.DataFrame(encoded, columns=te.columns_)

freq_items = apriori(basket, min_support=0.03, use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.50)
rules = rules.sort_values(["lift", "confidence"], ascending=False)

print("\nFrequent itemsets:\n", freq_items.sort_values(
    "support", ascending=False).head(10))

if rules.empty:
    print("\nNo association rules were found for this dataset with the current thresholds.")
else:
    print("\nRules by lift:\n", rules[[
          "antecedents", "consequents", "support", "confidence", "lift"]].head(10))
    interpret_rules(rules, top_n=5)

    plt.figure(figsize=(8, 5))
    plt.scatter(rules["support"], rules["confidence"],
                s=rules["lift"] * 25, alpha=0.65, c="crimson")
    plt.title("Supermarket Rules Scatter")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.tight_layout()
    plt.show()
