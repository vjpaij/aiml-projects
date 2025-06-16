### Description:

Anomaly detection identifies rare or unusual data points that differ significantly from the majority. Statistical methods like Z-score or IQR (Interquartile Range) are simple yet effective for detecting outliers in numerical data. In this project, we’ll use Z-score to find anomalies in a dataset representing daily transactions.

- Uses Z-score to measure how far each data point is from the mean
- Flags points that are unusually high or low as anomalies
- Visualizes anomalies clearly with a simple line plot

### Anomaly Detection Using Z-Score in Python

This script demonstrates how to detect anomalies in a dataset using the Z-score method. It simulates a simple use case of identifying unusual transaction amounts from a sequence of daily transactions.

#### Code Breakdown

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
```

* `numpy` is used for numerical operations and creating the data array.
* `matplotlib.pyplot` is used to visualize the data.
* `scipy.stats.zscore` calculates the Z-score for each data point, which measures how many standard deviations a value is from the mean.

```python
data = np.array([
    50, 52, 49, 51, 53, 48, 55, 54, 52, 49,
    51, 50, 500, 52, 47, 53, 49, 48, 700, 51
])
```

* A NumPy array named `data` represents daily transaction amounts in dollars.
* Most values are around 50, but there are two clear outliers: 500 and 700.

```python
z_scores = zscore(data)
```

* Computes the Z-score for each data point in the `data` array.

```python
threshold = 2.5
anomalies = np.where(np.abs(z_scores) > threshold)
```

* A threshold of 2.5 is used, meaning any data point with a Z-score greater than 2.5 or less than -2.5 is considered an anomaly.
* `np.where` identifies the indices where this condition is met.

```python
print("Anomaly Detection using Z-Score:\n")
for i in anomalies[0]:
    print(f"Index {i}: Value = {data[i]}, Z-score = {z_scores[i]:.2f}")
```

* Loops through the indices of anomalies and prints their index, value, and Z-score.

```python
plt.figure(figsize=(10, 5))
plt.plot(data, marker='o', label='Data Points')
plt.scatter(anomalies, data[anomalies], color='red', label='Anomalies')
plt.title("Transaction Anomaly Detection using Z-Score")
plt.xlabel("Day")
plt.ylabel("Transaction Amount ($)")
plt.legend()
plt.grid(True)
plt.show()
```

* Visualizes the data using a line plot.
* Anomalies are highlighted in red for easy identification.
* Provides axis labels, legend, grid, and a title for better understanding of the plot.

#### Summary

This example illustrates a simple but effective approach to anomaly detection using Z-scores. It is especially useful when working with normally distributed data and provides an easy method to flag outliers that deviate significantly from the mean.
