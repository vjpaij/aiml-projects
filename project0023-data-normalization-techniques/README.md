### Description:

Data normalization transforms features to a common scale without distorting differences in the ranges of values. It is critical for models that are sensitive to scale (e.g., KNN, SVM, Neural Networks). In this project, we’ll demonstrate and compare Min-Max Scaling, Z-score Standardization, and Max Abs Scaling using a synthetic dataset.

- Min-Max Scaling: Rescales values to [0, 1]
- Z-Score Standardization: Centers around mean (0) and scales to unit variance
- Max-Abs Scaling: Scales by maximum absolute value (retains sparsity)

## Data Scaling Techniques in Python

This script demonstrates three popular data scaling techniques using a small sample dataset. Scaling is an important step in preprocessing data for machine learning algorithms, especially those that rely on distance metrics (like k-NN, SVM, etc.).

### Libraries Used

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
```

* `numpy` and `pandas` are used for handling numerical and tabular data.
* `MinMaxScaler`, `StandardScaler`, and `MaxAbsScaler` are scaling tools from `scikit-learn`.

### Sample Dataset

```python
data = {
    'Feature_1': [100, 200, 300, 400, 500],   # Large range
    'Feature_2': [1, 2, 3, 4, 5],             # Small range
    'Feature_3': [-50, -25, 0, 25, 50]        # Negative to positive
}

df = pd.DataFrame(data)
print("Original Data:\n", df)
```

This dataset includes three features with differing value ranges:

* `Feature_1`: Large values
* `Feature_2`: Small values
* `Feature_3`: Spans negative to positive

### 1. Min-Max Scaling

```python
minmax_scaler = MinMaxScaler()
df_minmax = pd.DataFrame(minmax_scaler.fit_transform(df), columns=df.columns)
print("\nMin-Max Scaled Data:\n", df_minmax)
```

* Scales features to a fixed range, usually \[0, 1].
* Formula: `(X - X.min) / (X.max - X.min)`

### 2. Z-Score Standardization (Standard Scaling)

```python
standard_scaler = StandardScaler()
df_standard = pd.DataFrame(standard_scaler.fit_transform(df), columns=df.columns)
print("\nZ-Score Standardized Data:\n", df_standard)
```

* Centers data to have a mean of 0 and standard deviation of 1.
* Formula: `(X - mean) / std`

### 3. Max-Abs Scaling

```python
maxabs_scaler = MaxAbsScaler()
df_maxabs = pd.DataFrame(maxabs_scaler.fit_transform(df), columns=df.columns)
print("\nMax-Abs Scaled Data:\n", df_maxabs)
```

* Scales each feature by its maximum absolute value.
* Maintains sparsity of data; useful for data with zero-centered distribution.

### Summary

| Feature    | Original Range | Min-Max Range | Z-Score Mean\~0 | Max-Abs Range |
| ---------- | -------------- | ------------- | --------------- | ------------- |
| Feature\_1 | 100 to 500     | 0 to 1        | -1.41 to 1.41   | 0.2 to 1.0    |
| Feature\_2 | 1 to 5         | 0 to 1        | -1.41 to 1.41   | 0.2 to 1.0    |
| Feature\_3 | -50 to 50      | 0 to 1        | -1.41 to 1.41   | -1.0 to 1.0   |

Each scaling method serves different purposes and choosing the right one depends on the algorithm and nature of the data.
