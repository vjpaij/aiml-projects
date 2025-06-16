# Import required libraries
from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
 
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
 
# Split into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Create a base model: Decision Tree
base_model = DecisionTreeClassifier()
 
# Create BaggingClassifier with multiple trees trained on bootstrapped samples
bagging_model = BaggingClassifier(
    base_estimator=base_model,
    n_estimators=50,           # Number of trees
    max_samples=0.8,           # Fraction of dataset for each tree
    bootstrap=True,            # Use bootstrapped sampling
    random_state=42
)
 
# Train the bagging model
bagging_model.fit(X_train, y_train)
 
# Make predictions
y_pred = bagging_model.predict(X_test)
 
# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Bagging (Bootstrap Aggregation) Accuracy: {accuracy:.2f}")
 
# Optional: Visualize individual tree predictions (for the curious)
print("\nPredictions from first 5 individual estimators:")
for i, estimator in enumerate(bagging_model.estimators_[:5]):
    print(f"Tree {i+1} predictions: {estimator.predict(X_test)}")