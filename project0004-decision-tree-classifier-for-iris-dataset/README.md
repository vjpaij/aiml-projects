### Description:

A Decision Tree Classifier uses a tree-like model to make decisions based on feature values. In this project, we’ll use the famous Iris dataset to classify flower species (Setosa, Versicolor, Virginica) based on features like petal and sepal length/width. This model is easy to interpret and visualize.

This decision tree classifier is perfect for understanding how machine learning makes decisions based on feature thresholds. It's also easily visualizable, making it ideal for teaching, model interpretability, or quick prototyping.

## Decision Tree Classifier on the Iris Dataset

This script demonstrates how to train and evaluate a Decision Tree classifier using the famous Iris dataset with the help of `scikit-learn`. The Iris dataset consists of 150 samples from three species of Iris flowers (`setosa`, `versicolor`, and `virginica`) with four features each: sepal length, sepal width, petal length, and petal width.

### Code Explanation

```python
# Import required libraries
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
```

* **load\_iris**: Loads the Iris dataset.
* **DecisionTreeClassifier**: A classifier based on decision trees.
* **plot\_tree**: A utility to visualize the trained decision tree.
* **train\_test\_split**: Splits data into training and testing sets.
* **classification\_report**: Evaluates the prediction results.
* **matplotlib.pyplot**: Used for plotting the tree.

```python
# Load the Iris dataset
iris = load_iris()
X = iris.data      # Features: sepal length, sepal width, petal length, petal width
y = iris.target    # Labels: 0 = setosa, 1 = versicolor, 2 = virginica
```

* The dataset is loaded and separated into features (`X`) and labels (`y`).

```python
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

* The dataset is split into 70% training and 30% testing data to evaluate the model's generalization performance.
* `random_state` ensures reproducibility.

```python
# Create and train the decision tree classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
```

* A `DecisionTreeClassifier` is instantiated and trained using the training data.

```python
# Predict on the test set
y_pred = model.predict(X_test)
```

* The model is used to predict the labels for the test set.

```python
# Evaluate the model
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

* Generates a classification report including precision, recall, f1-score, and support for each class.

```python
# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Trained on Iris Dataset")
plt.show()
```

* The trained decision tree is visualized with nodes colored according to class purity.
* `filled=True` helps distinguish different classes visually.

---

This script is a useful starting point for beginners exploring supervised learning using decision trees in `scikit-learn`.
