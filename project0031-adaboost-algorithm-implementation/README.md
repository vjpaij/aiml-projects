### Description:

AdaBoost (Adaptive Boosting) is a boosting ensemble technique that combines multiple weak learners (typically decision stumps) to create a strong classifier. Each learner is trained focusing more on the samples that were misclassified by previous models. In this project, we implement AdaBoost using scikit-learn with a base Decision Tree on the Iris dataset.

- Builds multiple weak learners (decision stumps)
- Adjusts focus toward misclassified samples at each round
- Produces a strong classifier through adaptive boosting

## AdaBoost Classifier on Iris Dataset (Binary Classification)

This code demonstrates how to implement an AdaBoost classifier using the scikit-learn library on a subset of the Iris dataset. Specifically, it focuses on a binary classification task involving only the *Setosa* and *Versicolor* classes.

### Step-by-Step Explanation

#### 1. **Import Required Libraries**

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
```

These libraries are used for loading datasets, building classifiers, splitting data, and evaluating models.

#### 2. **Load the Iris Dataset**

```python
iris = load_iris()
X = iris.data
y = iris.target
```

`load_iris()` loads the famous Iris flower dataset. `X` contains feature data and `y` contains class labels.

#### 3. **Filter for Binary Classification**

```python
X_binary = X[y != 2]
y_binary = y[y != 2]
```

This step reduces the dataset to only two classes: *Setosa* (class 0) and *Versicolor* (class 1), excluding *Virginica* (class 2).

#### 4. **Split the Dataset**

```python
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.3, random_state=42)
```

The filtered data is split into training and testing sets. 30% is used for testing.

#### 5. **Create Base Estimator**

```python
base_estimator = DecisionTreeClassifier(max_depth=1)
```

A weak learner (decision stump) is defined. It is a decision tree with a maximum depth of 1.

#### 6. **Initialize AdaBoost Classifier**

```python
ada_model = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
```

An AdaBoost model is created using the base estimator. It will use 50 weak learners and a learning rate of 1.0.

#### 7. **Train the Model**

```python
ada_model.fit(X_train, y_train)
```

The AdaBoost model is trained on the training data.

#### 8. **Make Predictions**

```python
y_pred = ada_model.predict(X_test)
```

Predictions are made using the trained model on the test dataset.

#### 9. **Evaluate the Model**

```python
print(f"AdaBoost Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names[:2]))
```

The model's performance is evaluated using accuracy and a detailed classification report that includes precision, recall, and F1-score.

---

This code is a concise example of using boosting techniques for binary classification tasks using a well-known dataset.
