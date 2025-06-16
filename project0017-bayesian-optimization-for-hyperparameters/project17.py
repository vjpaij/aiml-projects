# Install scikit-optimize if not already installed:
# pip install scikit-optimize
 
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
import warnings
 
warnings.filterwarnings("ignore")  # To suppress convergence warnings for demo
 
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
 
# Define the search space for hyperparameters
search_space = {
    'C': Real(0.01, 100.0, prior='log-uniform'),
    'gamma': Real(1e-6, 1.0, prior='log-uniform'),
    'kernel': Categorical(['linear', 'rbf'])
}
 
# Create the SVM model
model = SVC()
 
# Set up Bayesian search
opt = BayesSearchCV(
    estimator=model,
    search_spaces=search_space,
    n_iter=30,             # Number of parameter settings to try
    cv=5,                  # 5-fold cross-validation
    scoring='accuracy',
    random_state=42
)
 
# Run the optimization
opt.fit(X, y)
 
# Output best parameters and score
print("Best Parameters (Bayesian Optimization):", opt.best_params_)
print(f"Best Cross-Validated Accuracy: {opt.best_score_:.4f}")