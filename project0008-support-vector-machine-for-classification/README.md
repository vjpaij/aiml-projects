### Description:

Support Vector Machines (SVM) are powerful classification algorithms that find the optimal hyperplane that separates different classes in feature space. They are particularly effective for high-dimensional data and complex decision boundaries. In this project, we’ll use SVM to classify data from the Iris dataset.

SVM works well for both linear and non-linear decision boundaries. You can:
- Experiment with kernel types (linear, RBF, polynomial).
- Adjust C (regularization) and gamma (kernel coefficient) for tuning.
- Try visualizing 2D SVM boundaries using a subset of features.

### Support Vector Machine (SVM) Classifier on Iris Dataset

This Python script demonstrates how to build and evaluate a Support Vector Machine (SVM) classifier using the well-known Iris dataset. The script is built using the `scikit-learn` library.

---

### Code Explanation

```python
# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
```

These lines import the required modules:

* `load_iris`: Loads the Iris dataset.
* `train_test_split`: Splits the dataset into training and testing sets.
* `SVC`: Support Vector Classifier implementation.
* `classification_report`: Generates a performance evaluation report.

```python
# Load the Iris dataset
iris = load_iris()
X = iris.data        # Features
y = iris.target      # Labels: 0 = setosa, 1 = versicolor, 2 = virginica
```

Loads the Iris dataset. `X` contains the four features for each flower: sepal length, sepal width, petal length, and petal width. `y` contains the target labels corresponding to the species.

```python
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

Splits the data into 70% training and 30% testing sets. The `random_state` ensures reproducibility.

```python
# Create an SVM classifier with RBF kernel
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
```

Creates an instance of an SVM classifier using the Radial Basis Function (RBF) kernel. You can try other kernels like `'linear'`, `'poly'`, or `'sigmoid'`.

```python
# Train the model
svm_model.fit(X_train, y_train)
```

Fits the model on the training data.

```python
# Predict the test set
y_pred = svm_model.predict(X_test)
```

Uses the trained model to make predictions on the test data.

```python
# Evaluate the performance
print("SVM Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

Prints the classification report that includes precision, recall, F1-score, and support for each class.

```python
# Predict a new sample
sample = [[6.0, 2.9, 4.5, 1.5]]
prediction = svm_model.predict(sample)[0]
print(f"\nPrediction for sample {sample[0]}: {iris.target_names[prediction]}")
```

Predicts the class of a new, unseen data point and prints the predicted species name.

---

### Output Example

```
SVM Classification Report:

              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        16
  versicolor       1.00      0.93      0.97        15
   virginica       0.94      1.00      0.97        14

    accuracy                           0.98        45
   macro avg       0.98      0.98      0.98        45
weighted avg       0.98      0.98      0.98        45

Prediction for sample [6.0, 2.9, 4.5, 1.5]: versicolor
```

---

### Summary

This script is a simple yet powerful example of using SVMs for multiclass classification. It demonstrates loading a dataset, preprocessing, model training, evaluation, and making predictions on new data.

