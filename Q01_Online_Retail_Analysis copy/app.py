# ============================
import os

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ----------------------------
# PART A: PREPROCESSING
# ----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load dataset from the question folder
df = pd.read_csv(os.path.join(BASE_DIR, "OnlineRetail.csv"),
                 encoding="ISO-8859-1")

# Remove cancelled invoices
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# Remove null values
df = df.dropna(subset=['InvoiceNo', 'Description'])

# Convert into transaction format
transactions = df.groupby('InvoiceNo')['Description'].apply(list)

# Top 10 products
top_items = df['Description'].value_counts().head(10)

print("Top 10 Products:\n", top_items)

# ----------------------------
# FIGURE 1 (Bar Chart)
# ----------------------------

plt.figure()
top_items.plot(kind='bar')
plt.title("Fig 1: Top 10 Products")
plt.xlabel("Products")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# ----------------------------
# PART B: MODEL BUILDING
# ----------------------------

# One-hot encoding
te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)
df_final = pd.DataFrame(te_data, columns=te.columns_)

# Apriori
freq_items = apriori(df_final, min_support=0.02, use_colnames=True)

# Association rules
rules = association_rules(freq_items, metric="confidence", min_threshold=0.40)

print("\nRules:\n", rules[['antecedents', 'consequents',
      'support', 'confidence', 'lift']].head())

# ----------------------------
# PART C: EVALUATION
# ----------------------------

# Sort rules
rules = rules.sort_values(by='confidence', ascending=False)

# ----------------------------
# FIGURE 2 (Scatter Plot)
# ----------------------------

plt.figure()
plt.scatter(rules['support'], rules['confidence'])
plt.title("Fig 2: Support vs Confidence")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.tight_layout()
plt.show()
