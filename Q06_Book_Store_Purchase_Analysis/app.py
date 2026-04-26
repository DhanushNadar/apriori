import os
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "Book_Sales_Dataset.csv"))

transactions = df.groupby("TransactionID")["BookTitle"].apply(list)
top_books = df["BookTitle"].value_counts().head(10)

print("Top books:\n", top_books)

plt.figure(figsize=(10, 5))
top_books.plot(kind="bar", color="darkcyan")
plt.title("Top 10 Books")
plt.xlabel("Book")
plt.ylabel("Count")
plt.xticks(rotation=40)
plt.tight_layout()
plt.show()

te = TransactionEncoder()
basket = pd.DataFrame(te.fit(transactions).transform(
    transactions), columns=te.columns_)

freq_items = apriori(basket, min_support=0.02, use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.55)
rules = rules.sort_values(["lift", "confidence"], ascending=False)

print("\nTop rules by lift:\n", rules[[
      "antecedents", "consequents", "support", "confidence", "lift"]].head(10))
print("\nReading pattern interpretation:")
for i, row in rules.head(5).iterrows():
    a = ", ".join(list(row["antecedents"]))
    c = ", ".join(list(row["consequents"]))
    print(
        f"Rule {i}: Readers buying [{a}] also buy [{c}] (lift={row['lift']:.2f}).")

plt.figure(figsize=(8, 5))
plt.scatter(rules["support"], rules["confidence"],
            s=rules["lift"] * 25, alpha=0.7, c="purple")
plt.title("Book Association Rules")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.tight_layout()
plt.show()
