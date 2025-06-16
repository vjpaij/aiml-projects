### Description:

Feature engineering is the process of transforming raw data into meaningful features that improve model performance. In this project, we’ll build a toolkit that demonstrates key feature engineering techniques including polynomial features, interaction terms, binarization, discretization, and label encoding—all in one place using a synthetic dataset.

What This Toolkit Includes:
- PolynomialFeatures: Adds squared and interaction terms
- Binarizer: Converts numerical values to 0/1
- KBinsDiscretizer: Buckets continuous features
- LabelEncoder: Converts categorical text to numeric values

## Data Preprocessing Demonstration

This script demonstrates various data preprocessing techniques using a sample dataset. These preprocessing steps are essential in preparing data for machine learning models.

### 1. Importing Libraries

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, Binarizer, KBinsDiscretizer, LabelEncoder
```

We import the following libraries:

* `pandas` and `numpy` for data manipulation.
* Preprocessing tools from `sklearn.preprocessing` for feature transformation.

### 2. Sample Dataset

```python
data = {
    'Age': [22, 25, 47, 52, 46],
    'Salary': [25000, 30000, 52000, 70000, 62000],
    'Purchased': ['No', 'Yes', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)
print("Original Data:\n", df)
```

We create a DataFrame with three features:

* `Age`: Numerical feature
* `Salary`: Numerical feature
* `Purchased`: Categorical feature indicating purchase behavior

### 3. Polynomial Features

```python
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['Age', 'Salary']])
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['Age', 'Salary']))
print("\nPolynomial Features (Degree 2):\n", poly_df)
```

This expands `Age` and `Salary` into polynomial combinations up to degree 2, helping models capture nonlinear relationships.

### 4. Binarization

```python
binarizer = Binarizer(threshold=50000)
binarized_salary = binarizer.fit_transform(df[['Salary']])
df['High_Salary'] = binarized_salary
print("\nBinarized 'Salary' (1 if > 50,000):\n", df[['Salary', 'High_Salary']])
```

The `Salary` feature is binarized: values greater than 50,000 are marked as 1, else 0. Useful for converting continuous data into binary flags.

### 5. Discretization (Bucketing)

```python
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
df['Age_Bin'] = discretizer.fit_transform(df[['Age']])
print("\nDiscretized 'Age' into 3 bins:\n", df[['Age', 'Age_Bin']])
```

`Age` is discretized into 3 uniform-width bins. Each bin is assigned an ordinal value (0, 1, 2).

### 6. Label Encoding

```python
label_encoder = LabelEncoder()
df['Purchased_Encoded'] = label_encoder.fit_transform(df['Purchased'])
print("\nLabel Encoded 'Purchased':\n", df[['Purchased', 'Purchased_Encoded']])
```

Categorical labels (`Purchased`) are encoded into integers: typically 0 for 'No' and 1 for 'Yes'.

---

These transformations are foundational steps when preparing data for machine learning algorithms, especially those that require numeric input.
