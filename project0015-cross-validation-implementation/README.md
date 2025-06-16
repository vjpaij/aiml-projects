### Description:

Cross-validation is a powerful model evaluation technique used to assess a model’s generalization capability. Instead of relying on a single train-test split, it partitions the data into multiple folds and evaluates performance across them. In this project, we demonstrate k-fold cross-validation using the Iris dataset and a logistic regression model.

- Loads the Iris dataset.
- Trains and evaluates a Logistic Regression model using 5-fold cross-validation.
- Computes and prints the accuracy for each fold, along with the mean and standard deviation.

## Logistic Regression with 5-Fold Cross-Validation on Iris Dataset

This script demonstrates how to apply **Logistic Regression** on the classic **Iris dataset** using **5-fold cross-validation** to evaluate model performance.

### Code Explanation

```python
# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
```

* **`load_iris`**: Loads the Iris dataset from scikit-learn's built-in datasets.
* **`LogisticRegression`**: Implements logistic regression classification.
* **`cross_val_score`**: Performs cross-validation and returns scores.
* **`KFold`**: Used to create k-fold cross-validation splits.
* **`numpy`**: Used for numerical operations like computing mean and standard deviation.

```python
# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
```

* Loads the features (`X`) and labels (`y`) from the Iris dataset.

```python
# Create a Logistic Regression model
model = LogisticRegression(max_iter=200)
```

* Initializes a logistic regression model with a maximum of 200 iterations (default is 100, which may not converge).

```python
# Define 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
```

* Splits the data into 5 folds.
* **`shuffle=True`** ensures that the data is randomly shuffled before splitting.
* **`random_state=42`** ensures reproducibility.

```python
# Perform cross-validation and get accuracy for each fold
cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
```

* Trains and evaluates the model on each of the 5 folds.
* Returns accuracy scores for each fold.

```python
# Output results
print("Cross-Validation Accuracies (5-Fold):", cv_scores)
print(f"Mean Accuracy: {np.mean(cv_scores):.2f}")
print(f"Standard Deviation: {np.std(cv_scores):.2f}")
```

* Prints individual fold accuracies, mean accuracy, and standard deviation to assess model stability and performance.

### Example Output

```
Cross-Validation Accuracies (5-Fold): [0.97 1.00 0.93 0.97 1.00]
Mean Accuracy: 0.97
Standard Deviation: 0.03
```

### Summary

This approach helps assess the model's generalization ability using cross-validation. The low standard deviation and high accuracy indicate that the logistic regression model performs well on the Iris dataset.
