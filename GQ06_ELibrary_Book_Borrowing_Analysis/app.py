import os
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "ELibrary_Dataset.csv"))

transactions = df.groupby("StudentID")["BooksBorrowed"].apply(list)
top_books = df["BooksBorrowed"].value_counts().head(10)

print("Most borrowed books:\n", top_books)

plt.figure(figsize=(10, 5))
top_books.plot(kind="bar", color="darkgoldenrod")
plt.title("Top Borrowed Books")
plt.xlabel("Book")
plt.ylabel("Frequency")
plt.xticks(rotation=35)
plt.tight_layout()
plt.show()

te = TransactionEncoder()
basket = pd.DataFrame(te.fit(transactions).transform(
    transactions), columns=te.columns_)

freq_items = apriori(basket, min_support=0.02, use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.38)
rules = rules.sort_values(["lift", "confidence"], ascending=False)

print("\nStrong book associations:\n", rules[[
      "antecedents", "consequents", "support", "confidence", "lift"]].head(10))
print("\nReading trend interpretation:")
for i, row in rules.head(5).iterrows():
    a = ", ".join(list(row["antecedents"]))
    c = ", ".join(list(row["consequents"]))
    print(f"Rule {i}: Students borrowing [{a}] also borrow [{c}] frequently.")

plt.figure(figsize=(8, 5))
plt.scatter(rules["support"], rules["confidence"],
            s=rules["lift"] * 30, alpha=0.65, c="goldenrod")
plt.title("E-Library Association Rules")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.tight_layout()
plt.show()
