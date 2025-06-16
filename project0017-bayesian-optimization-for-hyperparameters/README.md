### Description:

Bayesian Optimization is a smart hyperparameter tuning technique that builds a probabilistic model of the objective function and chooses the next set of hyperparameters to evaluate based on expected improvement. Unlike grid search, it’s much more efficient and sample-aware. In this project, we’ll use skopt (Scikit-Optimize) to tune an SVM classifier on the Iris dataset.

What Makes Bayesian Optimization Powerful:
- Selects next trial using probabilistic models (like Gaussian Processes)
- Finds optimal parameters using fewer evaluations
- Efficient for expensive models or large search spaces

## Bayesian Hyperparameter Optimization with scikit-optimize

This example demonstrates how to use **Bayesian optimization** to tune hyperparameters for an SVM (Support Vector Classifier) using the **scikit-optimize** library. We use the Iris dataset and perform 5-fold cross-validation to evaluate model performance.

### Installation

To use `scikit-optimize`, ensure it's installed:

```bash
pip install scikit-optimize
```

### Code Explanation

```python
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
import warnings

warnings.filterwarnings("ignore")  # Suppresses convergence warnings

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Define the hyperparameter search space
search_space = {
    'C': Real(0.01, 100.0, prior='log-uniform'),
    'gamma': Real(1e-6, 1.0, prior='log-uniform'),
    'kernel': Categorical(['linear', 'rbf'])
}

# Instantiate the SVM classifier
model = SVC()

# Set up the Bayesian optimizer with cross-validation
opt = BayesSearchCV(
    estimator=model,
    search_spaces=search_space,
    n_iter=30,             # Number of iterations (different parameter combinations to evaluate)
    cv=5,                  # 5-fold cross-validation
    scoring='accuracy',   # Metric to optimize
    random_state=42       # For reproducibility
)

# Run the optimization process
opt.fit(X, y)

# Display the best parameters and corresponding accuracy
print("Best Parameters (Bayesian Optimization):", opt.best_params_)
print(f"Best Cross-Validated Accuracy: {opt.best_score_:.4f}")
```

### Key Components

* **BayesSearchCV**: Optimizes hyperparameters using a probabilistic model (Bayesian optimization), more efficient than grid/random search.
* **search\_space**:

  * `C`: Regularization parameter for SVM.
  * `gamma`: Kernel coefficient for the 'rbf' kernel.
  * `kernel`: Type of SVM kernel to use ('linear' or 'rbf').
* **cv=5**: Ensures performance is averaged over 5 data splits.
* **n\_iter=30**: Limits the number of parameter evaluations for efficiency.

### Output

After execution, the script prints:

* The best hyperparameter combination found.
* The corresponding cross-validated accuracy score.

This method is particularly useful for efficiently tuning models with expensive evaluation functions or large search spaces.


