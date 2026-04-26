import os
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "Learning_Module_Dataset.csv"))

transactions = df.groupby("UserID")["ModulesAccessed"].apply(list)
top_modules = df["ModulesAccessed"].value_counts().head(10)

print("Frequently accessed modules:\n", top_modules)

plt.figure(figsize=(10, 5))
top_modules.plot(kind="bar", color="darkgreen")
plt.title("Top Learning Modules")
plt.xlabel("Module")
plt.ylabel("Frequency")
plt.xticks(rotation=40)
plt.tight_layout()
plt.show()

te = TransactionEncoder()
basket = pd.DataFrame(te.fit(transactions).transform(
    transactions), columns=te.columns_)

freq_items = apriori(basket, min_support=0.025, use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.45)
rules = rules.sort_values(["lift", "confidence"], ascending=False)

print("\nStrong module rules:\n", rules[[
      "antecedents", "consequents", "support", "confidence", "lift"]].head(10))
print("\nLearning behavior interpretation:")
for i, row in rules.head(5).iterrows():
    a = ", ".join(list(row["antecedents"]))
    c = ", ".join(list(row["consequents"]))
    print(f"Rule {i}: Users who access [{a}] often continue to [{c}].")

plt.figure(figsize=(8, 5))
plt.scatter(rules["support"], rules["confidence"],
            s=rules["lift"] * 30, alpha=0.65, c="forestgreen")
plt.title("Module Rule Relationships")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.tight_layout()
plt.show()
