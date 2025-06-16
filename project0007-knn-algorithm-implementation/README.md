### Description:

The K-Nearest Neighbors (KNN) algorithm is a simple, instance-based learning method that classifies new data points based on the majority label among their k closest neighbors in the feature space. It’s intuitive, non-parametric, and works well for small to medium-sized datasets. In this project, we’ll use KNN to classify Iris flower species.

This KNN model classifies flowers based on their physical dimensions. You can:
- Change k to test model performance.
- Try distance weighting, or different distance metrics like Manhattan or Minkowski.
- Visualize decision boundaries for better understanding.

## K-Nearest Neighbors (KNN) Classifier on the Iris Dataset

This Python script demonstrates how to use the **K-Nearest Neighbors (KNN)** classification algorithm with the **Iris dataset**, a classic dataset in machine learning and statistics.

### Code Breakdown

#### 1. **Import Required Libraries**

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
```

These libraries provide:

* `load_iris`: Function to load the Iris dataset.
* `train_test_split`: Utility to split data into training and testing sets.
* `KNeighborsClassifier`: The KNN algorithm implementation.
* `classification_report`: Generates performance metrics for the classification model.

#### 2. **Load the Dataset**

```python
iris = load_iris()
X = iris.data        # Features: sepal length, sepal width, petal length, petal width
y = iris.target      # Labels: 0 = setosa, 1 = versicolor, 2 = virginica
```

Here we extract the feature matrix `X` and the target vector `y` from the Iris dataset.

#### 3. **Split the Data**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

Splits the dataset into 70% training and 30% testing subsets. The `random_state` ensures reproducibility.

#### 4. **Initialize and Train the KNN Model**

```python
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
```

Creates a KNN classifier with `k=3` (i.e., using the 3 nearest neighbors) and fits it to the training data.

#### 5. **Make Predictions**

```python
y_pred = knn.predict(X_test)
```

Predicts the labels for the test data.

#### 6. **Evaluate the Model**

```python
print("KNN Classification Report (k=3):\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

Outputs precision, recall, F1-score, and support for each class.

#### 7. **Predict a New Sample**

```python
sample = [[5.0, 3.5, 1.3, 0.2]]
prediction = knn.predict(sample)[0]
print(f"\nPrediction for sample {sample[0]}: {iris.target_names[prediction]}")
```

Predicts the class of a new flower sample and prints the predicted species.

### Output Example

```
KNN Classification Report (k=3):

              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        16
  versicolor       1.00      0.94      0.97        18
   virginica       0.94      1.00      0.97        11

    accuracy                           0.98        45
   macro avg       0.98      0.98      0.98        45
weighted avg       0.98      0.98      0.98        45

Prediction for sample [5.0, 3.5, 1.3, 0.2]: setosa
```

### Conclusion

This script provides a straightforward example of how to apply a basic machine learning algorithm using `scikit-learn` to classify flower species based on physical measurements.

