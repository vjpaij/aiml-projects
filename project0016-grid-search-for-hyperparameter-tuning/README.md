### Description:

Grid Search is a brute-force approach to hyperparameter tuning, where we evaluate model performance over a predefined grid of parameters. This helps in finding the best combination that yields the highest accuracy. In this project, we use GridSearchCV to tune a Support Vector Machine (SVM) on the Iris dataset.

- Searches all combinations of C, gamma, and kernel
- Uses 5-fold cross-validation to evaluate each combination
- Selects the best hyperparameters and evaluates the model on the test set

## SVM Classification on Iris Dataset with Grid Search

This code demonstrates how to perform Support Vector Machine (SVM) classification on the popular Iris dataset using grid search to optimize hyperparameters.

### 1. Importing Required Libraries

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
```

These libraries provide utilities for:

* Loading the Iris dataset.
* Splitting the data into training and test sets.
* Training an SVM model.
* Performing hyperparameter tuning using Grid Search with cross-validation.
* Evaluating the model performance.

### 2. Loading the Dataset

```python
iris = load_iris()
X = iris.data
y = iris.target
```

* `load_iris()` loads the Iris dataset.
* `X` contains the feature data (sepal and petal measurements).
* `y` contains the target labels (species).

### 3. Splitting Data into Train and Test Sets

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

* The dataset is split into 70% training and 30% testing.
* `random_state=42` ensures reproducibility.

### 4. Defining Hyperparameter Grid

```python
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['linear', 'rbf']
}
```

This grid defines the hyperparameters to tune:

* `C`: Regularization parameter.
* `gamma`: Kernel coefficient.
* `kernel`: Type of SVM kernel function (`linear` or `rbf`).

### 5. Model Creation and Grid Search

```python
model = SVC()
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
```

* `SVC()` creates the SVM classifier.
* `GridSearchCV` tests all parameter combinations using 5-fold cross-validation.
* The model is trained on the training data.

### 6. Outputting Best Parameters and Score

```python
print("Best Parameters Found:", grid_search.best_params_)
print(f"Best Cross-Validation Score: {grid_search.best_score_:.2f}")
```

Displays the best combination of hyperparameters and the corresponding accuracy score from cross-validation.

### 7. Testing on Unseen Data

```python
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
```

* Uses the best model found by Grid Search to make predictions on the test set.

### 8. Evaluation Metrics

```python
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

* Generates a detailed classification report showing precision, recall, and F1-score for each class (Iris species).

---

This pipeline is a standard approach in supervised machine learning: load data, split, tune with cross-validation, train, and evaluate.
