import os
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "Student_Course_Dataset.csv"))

transactions = df.groupby("StudentID")["Courses"].apply(list)
top_courses = df["Courses"].value_counts().head(10)

print("Top courses:\n", top_courses)

plt.figure(figsize=(10, 5))
top_courses.plot(kind="bar", color="steelblue")
plt.title("Top 10 Selected Courses")
plt.xlabel("Course")
plt.ylabel("Frequency")
plt.xticks(rotation=40)
plt.tight_layout()
plt.show()

te = TransactionEncoder()
basket = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)

freq_items = apriori(basket, min_support=0.03, use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.50)
rules = rules.sort_values(["confidence", "lift"], ascending=False)

print("\nTop rules:\n", rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))
print("\nStrong course combination interpretation:")
for i, row in rules.head(5).iterrows():
    a = ", ".join(list(row["antecedents"])) # Courses already selected
    c = ", ".join(list(row["consequents"])) # Course likely to be selected next
    print(f"Rule {i}: Students selecting [{a}] frequently also select [{c}].") # How often this combination appears in dataset

plt.figure(figsize=(8, 5)) # confidence: If student takes LEFT courses, how likely they take RIGHT course
plt.scatter(rules["support"], rules["confidence"], s=rules["lift"] * 35, alpha=0.65, c="darkblue")
plt.title("Course Association Rules")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.tight_layout()
plt.show()
