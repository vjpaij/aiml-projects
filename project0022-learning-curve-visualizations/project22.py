# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
 
# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
 
# Create the model
model = LogisticRegression(max_iter=200)
 
# Generate learning curve data
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),  # 10 different sizes from 10% to 100%
    cv=5,                                   # 5-fold cross-validation
    scoring='accuracy',
    shuffle=True,
    random_state=42
)
 
# Calculate mean and std for plotting
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)
 
# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
plt.plot(train_sizes, test_mean, 'o-', color='green', label='Cross-validation score')
 
# Fill between for std shading
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='green')
 
plt.title("Learning Curve - Logistic Regression (Iris Dataset)")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("learning_curve_logistic_regression_iris.png")  # Save the plot as an image