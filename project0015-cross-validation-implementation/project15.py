# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
 
# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
 
# Create a Logistic Regression model
model = LogisticRegression(max_iter=200)
 
# Define 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
 
# Perform cross-validation and get accuracy for each fold
cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
 
# Output results
print("Cross-Validation Accuracies (5-Fold):", cv_scores)
print(f"Mean Accuracy: {np.mean(cv_scores):.2f}")
print(f"Standard Deviation: {np.std(cv_scores):.2f}")
