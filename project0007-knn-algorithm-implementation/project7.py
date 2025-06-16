# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
 
# Load the Iris dataset
iris = load_iris()
X = iris.data        # Features: sepal length, sepal width, petal length, petal width
y = iris.target      # Labels: 0 = setosa, 1 = versicolor, 2 = virginica
 
# Split dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Create a KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)
 
# Train the model on training data
knn.fit(X_train, y_train)
 
# Predict the classes for the test set
y_pred = knn.predict(X_test)
 
# Evaluate the model
print("KNN Classification Report (k=3):\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
 
# Predict a new sample
sample = [[5.0, 3.5, 1.3, 0.2]]
prediction = knn.predict(sample)[0]
print(f"\nPrediction for sample {sample[0]}: {iris.target_names[prediction]}")