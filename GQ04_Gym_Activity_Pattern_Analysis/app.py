import os
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "Gym_Activity_Dataset.csv"))

transactions = df.groupby("UserID")["Activities"].apply(list)
top_activities = df["Activities"].value_counts().head(10)

print("Most common activities:\n", top_activities)

plt.figure(figsize=(10, 5))
top_activities.plot(kind="bar", color="darkturquoise")
plt.title("Top Gym Activities")
plt.xlabel("Activity")
plt.ylabel("Frequency")
plt.xticks(rotation=40)
plt.tight_layout()
plt.show()

te = TransactionEncoder()
basket = pd.DataFrame(te.fit(transactions).transform(
    transactions), columns=te.columns_)

freq_items = apriori(basket, min_support=0.03, use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.35)
rules = rules.sort_values(["lift", "confidence"], ascending=False)

print("\nTop activity combinations:\n", rules[[
      "antecedents", "consequents", "support", "confidence", "lift"]].head(10))
print("\nFitness pattern interpretation:")
for i, row in rules.head(5).iterrows():
    a = ", ".join(list(row["antecedents"]))
    c = ", ".join(list(row["consequents"]))
    print(f"Rule {i}: Members doing [{a}] also tend to do [{c}].")

plt.figure(figsize=(8, 5))
plt.scatter(rules["support"], rules["confidence"],
            s=rules["lift"] * 30, alpha=0.65, c="teal")
plt.title("Gym Activity Rules")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.tight_layout()
plt.show()
