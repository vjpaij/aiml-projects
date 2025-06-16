### Description:

Gradient Boosting builds a strong learner by combining multiple weak learners (typically decision trees), each correcting the errors of the previous ones. It’s widely used in competitions and real-world applications due to its high accuracy and flexibility. In this project, we’ll use the Iris dataset and implement Gradient Boosting using scikit-learn.

Gradient Boosting is:
- Excellent for tabular data
- Less prone to overfitting (with tuning)
- Basis of powerful libraries like XGBoost, LightGBM, and CatBoost

## Gradient Boosting Classifier on Iris Dataset

This script demonstrates how to train and evaluate a **Gradient Boosting Classifier** using the well-known **Iris dataset**. It includes model training, evaluation, a sample prediction, and visualization of feature importances.

### Code Explanation

```python
# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
```

We import the essential libraries from `scikit-learn` for data handling, model training, evaluation, and `matplotlib` for plotting.

```python
# Load the Iris dataset
iris = load_iris()
X = iris.data        # Features
y = iris.target      # Labels: 0 = setosa, 1 = versicolor, 2 = virginica
```

We load the Iris dataset which contains 150 samples of iris flowers, each described by 4 features: sepal length, sepal width, petal length, and petal width. The labels represent 3 classes of iris species.

```python
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

We split the data into 70% training and 30% testing sets to evaluate the model's performance.

```python
# Initialize the Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
```

We initialize a `GradientBoostingClassifier` with:

* 100 estimators (trees)
* Learning rate of 0.1
* Maximum depth of 3 for each tree

```python
# Train the model
gb_model.fit(X_train, y_train)
```

We train the model using the training data.

```python
# Predict on the test data
y_pred = gb_model.predict(X_test)
```

We use the trained model to predict the classes of the test set.

```python
# Evaluate the model
print("Gradient Boosting Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

We evaluate the model performance using `classification_report`, which shows precision, recall, f1-score, and support for each class.

```python
# Predict a custom sample
sample = [[6.1, 2.8, 4.7, 1.2]]
prediction = gb_model.predict(sample)[0]
print(f"\nPrediction for sample {sample[0]}: {iris.target_names[prediction]}")
```

We predict the class of a new sample flower and print the predicted species.

```python
# Optional: Plot feature importance
feature_importances = gb_model.feature_importances_
features = iris.feature_names

plt.barh(features, feature_importances)
plt.xlabel("Feature Importance")
plt.title("Gradient Boosting Feature Importance (Iris Dataset)")
plt.grid(True)
plt.show()
```

We visualize the importance of each feature as determined by the trained model. This helps understand which features are most influential in the classification decisions.

---

### Output Example

```
Gradient Boosting Classification Report:

              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        16
  versicolor       1.00      0.94      0.97        16
   virginica       0.94      1.00      0.97        13

    accuracy                           0.98        45
   macro avg       0.98      0.98      0.98        45
weighted avg       0.98      0.98      0.98        45

Prediction for sample [6.1, 2.8, 4.7, 1.2]: versicolor
```

---

### Requirements

Make sure to install the required packages:

```bash
pip install scikit-learn matplotlib
```

---

### License

This example is provided under the MIT License for educational purposes.
