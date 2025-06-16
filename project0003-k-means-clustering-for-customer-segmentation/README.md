### Description:

K-Means Clustering is an unsupervised learning algorithm used to group similar data points into k clusters based on feature similarity. In this project, we’ll apply K-Means to simulate customer data (e.g., annual income and spending score) to identify customer segments like "high spenders" or "budget-conscious" shoppers.

This project segments customers based on their income and spending habits, helping businesses to target groups more effectively—like offering premium products to high-income, high-spending clusters.

## Customer Segmentation using K-Means Clustering

This Python script demonstrates how to use K-Means clustering for customer segmentation based on their annual income and spending score. This is a common unsupervised machine learning technique used in marketing and business analytics to group customers into different clusters for targeted strategies.

### Code Explanation

#### 1. **Import Required Libraries**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
```

* `numpy`: Used for numerical operations and handling arrays.
* `matplotlib.pyplot`: Used for visualizing the data and the resulting clusters.
* `sklearn.cluster.KMeans`: Provides the KMeans clustering algorithm.

#### 2. **Sample Dataset**

```python
X = np.array([
    [15, 39], [16, 81], [17, 6], [18, 77], [19, 40], [20, 76],
    [25, 50], [30, 60], [35, 80], [40, 20],
    [60, 85], [65, 70], [70, 60], [75, 50], [80, 30],
    [85, 90], [90, 70], [95, 40], [100, 20], [105, 10]
])
```

* The dataset represents 20 customers, each described by two features:

  * Annual income (in \$1000s)
  * Spending score (a numerical value indicating customer spending behavior)

#### 3. **Initialize KMeans Model**

```python
kmeans = KMeans(n_clusters=3, random_state=42)
```

* We initialize the KMeans algorithm to create 3 clusters (`n_clusters=3`).
* `random_state=42` ensures reproducibility.

#### 4. **Fit Model and Predict Clusters**

```python
y_kmeans = kmeans.fit_predict(X)
```

* The model is trained using the dataset `X`.
* `fit_predict()` assigns each data point to one of the 3 clusters.

#### 5. **Get Cluster Centers**

```python
centers = kmeans.cluster_centers_
```

* After training, the model identifies the coordinates of the cluster centers.

#### 6. **Plot the Results**

```python
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', label='Customers')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')
plt.xlabel("Annual Income ($1000s)")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means Clustering")
plt.legend()
plt.grid(True)
plt.show()
```

* Displays a 2D scatter plot of customers color-coded by cluster.
* Cluster centers are highlighted with red 'X' markers.

#### 7. **Print Cluster Centers**

```python
print("Cluster Centers (Annual Income, Spending Score):")
for i, center in enumerate(centers):
    print(f"Cluster {i + 1}: Income = {center[0]:.2f}, Spending Score = {center[1]:.2f}")
```

* Outputs the exact coordinates of each cluster center for further interpretation.

### Use Case

This clustering approach can help businesses:

* Identify high-income, high-spending customers.
* Tailor marketing strategies for each segment.
* Improve customer service and retention.

---

**Note:** K-Means assumes that clusters are spherical and equally sized, which may not always reflect real-world customer behavior. It's important to validate assumptions and explore other clustering methods if necessary.




