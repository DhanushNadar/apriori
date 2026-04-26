import os
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "Movie_Dataset.csv"))

transactions = df.groupby("UserID")["MovieTitle"].apply(list)
top_movies = df["MovieTitle"].value_counts().head(10)

print("Most watched movies:\n", top_movies)

plt.figure(figsize=(10, 5))
top_movies.plot(kind="bar", color="darkorange")
plt.title("Top Watched Movies")
plt.xlabel("Movie")
plt.ylabel("Views")
plt.xticks(rotation=35)
plt.tight_layout()
plt.show()

te = TransactionEncoder()
basket = pd.DataFrame(te.fit(transactions).transform(
    transactions), columns=te.columns_)

freq_items = apriori(basket, min_support=0.005, use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.25)
rules = rules.sort_values(["lift", "confidence"], ascending=False)

print("\nStrong movie associations:\n", rules[[
      "antecedents", "consequents", "support", "confidence", "lift"]].head(10))
print("\nRecommendation pattern interpretation:")
for i, row in rules.head(5).iterrows():
    a = ", ".join(list(row["antecedents"]))
    c = ", ".join(list(row["consequents"]))
    print(
        f"Rule {i}: Users who watched [{a}] often watched [{c}] (confidence={row['confidence']:.2f}).")

plt.figure(figsize=(8, 5))
plt.scatter(rules["support"], rules["confidence"],
            s=rules["lift"] * 12, alpha=0.55, c="firebrick")
plt.title("Movie Rules Scatter")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.tight_layout()
plt.show()
