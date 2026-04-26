import os
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "Pharmacy_Dataset.csv"))
df = df.dropna().drop_duplicates()

transactions = df.groupby("BillNo")["MedicineName"].apply(list)
top_medicines = df["MedicineName"].value_counts().head(10)

print("Frequently purchased medicines:\n", top_medicines)

plt.figure(figsize=(10, 5))
top_medicines.plot(kind="bar", color="royalblue")
plt.title("Top 10 Medicines")
plt.xlabel("Medicine")
plt.ylabel("Count")
plt.xticks(rotation=50)
plt.tight_layout()
plt.show()

te = TransactionEncoder()
basket = pd.DataFrame(te.fit(transactions).transform(
    transactions), columns=te.columns_)

freq_items = apriori(basket, min_support=0.025, use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.45)
rules = rules.sort_values(["lift", "confidence"], ascending=False)

print("\nStrong medicine rules:\n", rules[[
      "antecedents", "consequents", "support", "confidence", "lift"]].head(10))
print("\nInterpretation of 5 rules:")
for i, row in rules.head(5).iterrows():
    lhs = ", ".join(list(row["antecedents"]))
    rhs = ", ".join(list(row["consequents"]))
    print(
        f"Rule {i}: [{lhs}] => [{rhs}] with confidence {row['confidence']:.2f} and lift {row['lift']:.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(rules["support"], rules["confidence"],
            s=rules["lift"] * 25, alpha=0.65, c="navy")
plt.title("Pharmacy Rules Scatter")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.tight_layout()
plt.show()
