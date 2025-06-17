### Description:

The Apriori algorithm is a classic algorithm for frequent itemset mining and association rule learning. It identifies items that frequently co-occur in transactional datasets, such as products bought together. This project uses the Apriori algorithm (via mlxtend) to mine frequent itemsets and generate interpretable association rules from sample transactions.

- How to convert transactional data into a format suitable for rule mining
- Use of Apriori to extract frequent itemsets
- Generates interpretable association rules with confidence and lift metrics

## Apriori Algorithm and Association Rule Mining Using `mlxtend`

This script demonstrates how to perform **frequent itemset mining** and generate **association rules** using the Apriori algorithm with the help of the `mlxtend` library.

### 🧰 Prerequisites

Install `mlxtend` if not already installed:

```bash
pip install mlxtend
```

### 📦 Sample Data

We define a small list of transactions where each transaction is a list of items purchased together:

```python
transactions = [
    ['milk', 'bread', 'butter'],
    ['milk', 'bread'],
    ['milk', 'butter'],
    ['bread', 'butter'],
    ['milk', 'bread', 'butter'],
    ['bread', 'jam'],
    ['milk', 'jam'],
    ['bread', 'butter', 'jam'],
    ['milk', 'bread', 'jam'],
    ['butter']
]
```

### 🧼 Step 1: One-Hot Encoding

Convert the transaction list into a one-hot encoded DataFrame using `TransactionEncoder`:

```python
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
```

Each column in the resulting DataFrame represents an item, and each row corresponds to a transaction, with `True/False` indicating the presence/absence of the item.

### 📈 Step 2: Apply Apriori Algorithm

Use the Apriori algorithm to find frequent itemsets with a minimum support threshold:

```python
from mlxtend.frequent_patterns import apriori

frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
print("\ud83d\udd0d Frequent Itemsets:\n")
print(frequent_itemsets)
```

This will output itemsets that occur in at least 30% of the transactions.

### 🤝 Step 3: Generate Association Rules

Extract association rules from the frequent itemsets:

```python
from mlxtend.frequent_patterns import association_rules

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
```

### 📊 Output Association Rules

Print the discovered rules in an interpretable format:

```python
for _, row in rules.iterrows():
    ant = ', '.join(list(row['antecedents']))
    con = ', '.join(list(row['consequents']))
    print(f"If a user buys [{ant}], they are likely to buy [{con}] — "
          f"support: {row['support']:.2f}, confidence: {row['confidence']:.2f}, lift: {row['lift']:.2f}")
```

### ✅ Summary

* `TransactionEncoder` transforms transaction data to a suitable format.
* `apriori` finds frequent itemsets based on support.
* `association_rules` derives actionable rules with confidence and lift metrics.

This is a foundational technique in **market basket analysis** and can be expanded to much larger datasets in retail, recommendation systems, and more.
