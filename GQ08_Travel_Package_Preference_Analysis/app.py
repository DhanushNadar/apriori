import os
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "Travel_Package_Dataset.csv"))

transactions = df.groupby("CustomerID")["SelectedServices"].apply(list)
top_services = df["SelectedServices"].value_counts().head(10)

print("Frequently selected services:\n", top_services)

plt.figure(figsize=(10, 5))
top_services.plot(kind="bar", color="mediumseagreen")
plt.title("Top Travel Services")
plt.xlabel("Service")
plt.ylabel("Frequency")
plt.xticks(rotation=35)
plt.tight_layout()
plt.show()

te = TransactionEncoder()
basket = pd.DataFrame(te.fit(transactions).transform(
    transactions), columns=te.columns_)

freq_items = apriori(basket, min_support=0.025, use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.42)
rules = rules.sort_values(["lift", "confidence"], ascending=False)

print("\nService combination rules:\n", rules[[
      "antecedents", "consequents", "support", "confidence", "lift"]].head(10))
print("\nTravel preference interpretation:")
for i, row in rules.head(5).iterrows():
    a = ", ".join(list(row["antecedents"]))
    c = ", ".join(list(row["consequents"]))
    print(f"Rule {i}: Customers choosing [{a}] often also choose [{c}].")

plt.figure(figsize=(8, 5))
plt.scatter(rules["support"], rules["confidence"],
            s=rules["lift"] * 30, alpha=0.65, c="seagreen")
plt.title("Travel Rules Scatter")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.tight_layout()
plt.show()
