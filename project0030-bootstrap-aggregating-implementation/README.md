### Description:

Bagging (Bootstrap Aggregating) is an ensemble method that builds multiple versions of a predictor (e.g., decision trees) on different bootstrapped datasets and averages their outputs to reduce variance and prevent overfitting. In this project, we implement BaggingClassifier using scikit-learn on the Iris dataset.

- Applies bootstrapping to create multiple training subsets
- Trains base learners (Decision Trees) on these subsets
- Aggregates predictions (majority vote for classification)

## Bagging Classifier on the Iris Dataset

This code demonstrates how to implement a Bagging (Bootstrap Aggregation) Classifier using a Decision Tree as the base estimator on the classic Iris dataset. The Bagging technique is used to improve the stability and accuracy of machine learning algorithms by combining the results of multiple models trained on random subsets of the training data.

### Code Explanation

```python
# Import required libraries
from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
```

* **sklearn.datasets.load\_iris**: Loads the Iris flower dataset.
* **BaggingClassifier**: Implements bagging ensemble meta-estimator.
* **DecisionTreeClassifier**: Serves as the base model for each ensemble member.
* **train\_test\_split**: Splits dataset into training and testing subsets.
* **accuracy\_score**: Evaluates the prediction accuracy.
* **matplotlib.pyplot**: Imported but unused in this specific example.

```python
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
```

* Loads the Iris dataset into feature matrix `X` and target labels `y`.

```python
# Split into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

* Splits data into 70% training and 30% testing sets with a fixed random seed for reproducibility.

```python
# Create a base model: Decision Tree
base_model = DecisionTreeClassifier()
```

* A decision tree is chosen as the base model.

```python
# Create BaggingClassifier with multiple trees trained on bootstrapped samples
bagging_model = BaggingClassifier(
    base_estimator=base_model,
    n_estimators=50,           # Number of trees
    max_samples=0.8,           # Fraction of dataset for each tree
    bootstrap=True,            # Use bootstrapped sampling
    random_state=42
)
```

* A Bagging ensemble of 50 decision trees is created.
* Each tree is trained on 80% of the training data using bootstrap sampling.

```python
# Train the bagging model
bagging_model.fit(X_train, y_train)
```

* The ensemble model is trained on the training data.

```python
# Make predictions
y_pred = bagging_model.predict(X_test)
```

* Predicts the class labels for the test data.

```python
# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Bagging (Bootstrap Aggregation) Accuracy: {accuracy:.2f}")
```

* Computes and prints the accuracy of the Bagging model.

```python
# Optional: Visualize individual tree predictions (for the curious)
print("\nPredictions from first 5 individual estimators:")
for i, estimator in enumerate(bagging_model.estimators_[:5]):
    print(f"Tree {i+1} predictions: {estimator.predict(X_test)}")
```

* Displays the predictions from the first 5 decision trees in the ensemble to give insight into how individual models contribute to the final result.

---

### Summary

This implementation illustrates how Bagging can be used to improve the performance of a base classifier by reducing variance and overfitting, using multiple decision trees trained on different subsets of the data.
