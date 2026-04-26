import os
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def interpret_rules(rules_df, top_n=5):
    print("\nFood combination interpretations:")
    for i, row in rules_df.head(top_n).iterrows():
        lhs = ", ".join(list(row["antecedents"]))
        rhs = ", ".join(list(row["consequents"]))
        print(
            f"Rule {i}: Customers who choose [{lhs}] also choose [{rhs}] often (lift={row['lift']:.2f}).")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "Food_Delivery_train.csv"))

# Data Cleaning
df = df.dropna(subset=["ID", "Type_of_order"])
df = df.drop_duplicates(subset=["ID", "Type_of_order"])

# Create Transactions
transactions = df.groupby("ID")["Type_of_order"].apply(list)

# Top Items
top_items = df["Type_of_order"].value_counts().head(10)
print("Most ordered food-item types:\n", top_items)

# Bar Chart
plt.figure(figsize=(8, 5))
top_items.plot(kind="bar", color="seagreen")
plt.title("Top 10 Ordered Item Types - Food Delivery")
plt.xlabel("Type of Order")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Encoding
te = TransactionEncoder()
encoded = te.fit(transactions).transform(transactions)
basket = pd.DataFrame(encoded, columns=te.columns_)

# Apriori
freq_items = apriori(basket, min_support=0.01, use_colnames=True)

# Association Rules
rules = association_rules(freq_items, metric="confidence", min_threshold=0.30)
rules = rules.sort_values(["lift", "confidence"], ascending=False)

# Print Frequent Itemsets
print("\nFrequent itemsets (top):\n",
      freq_items.sort_values("support", ascending=False).head(10))

# ✅ ADDING IF-ELSE HERE
if rules.empty:
    print("\nNo association rules were found for this dataset with the current thresholds.")
else:
    print("\nTop rules by lift:\n", rules[[
          "antecedents", "consequents", "support", "confidence", "lift"]].head(10))

    interpret_rules(rules, top_n=5)

    # Scatter Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(rules["support"], rules["confidence"],
                s=rules["lift"] * 30, alpha=0.6, c="darkred")
    plt.title("Food Delivery Rule Distribution")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.tight_layout()
    plt.show()