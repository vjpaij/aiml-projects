### Description:

The Precision-Recall (PR) curve is a useful evaluation metric, especially for imbalanced binary classification problems. It helps visualize the trade-off between precision (positive predictive value) and recall (true positive rate). In this project, we'll train a logistic regression model and plot its PR curve using the Iris dataset (binary version).

- Calculates precision-recall pairs for different thresholds
- Plots the PR curve and average precision (AP) score
- Shows how well a model identifies positives without too many false alarms

## Precision-Recall Curve Using Logistic Regression on Iris Dataset

This example demonstrates how to compute and plot a Precision-Recall curve for a binary classification problem using the Iris dataset and a logistic regression model.

### Code Breakdown

```python
# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
```

* **load\_iris**: Loads the Iris dataset.
* **train\_test\_split**: Splits the data into training and testing sets.
* **LogisticRegression**: Applies logistic regression for binary classification.
* **precision\_recall\_curve**, **average\_precision\_score**: Used for evaluating model performance.
* **matplotlib.pyplot**: For plotting the precision-recall curve.

```python
# Load Iris dataset (we'll simplify to binary classification: Setosa vs Versicolor)
iris = load_iris()
X = iris.data
y = iris.target

# Keep only Setosa and Versicolor classes (labels 0 and 1)
X_binary = X[y != 2]
y_binary = y[y != 2]
```

* We load the dataset and filter out the 'Virginica' class to simplify the problem to a binary classification: Setosa (0) vs Versicolor (1).

```python
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.3, random_state=42)
```

* Split the data into 70% training and 30% testing with a fixed random seed for reproducibility.

```python
# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
```

* Initialize and train the logistic regression model on the training data.

```python
# Get predicted probabilities
y_probs = model.predict_proba(X_test)[:, 1]  # Probability of class "1"
```

* Predict the probability that each instance in the test set belongs to class 1 (Versicolor).

```python
# Calculate precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
average_precision = average_precision_score(y_test, y_probs)
```

* Compute the precision-recall pairs for different probability thresholds.
* Calculate the average precision score.

```python
# Plot Precision-Recall curve
plt.figure(figsize=(8, 5))
plt.plot(recall, precision, color='purple', label=f'PR Curve (AP = {average_precision:.2f})')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

* Plot the precision-recall curve using matplotlib.
* Display the average precision score in the legend.
* Add appropriate labels and grid for better readability.

### Result

This visualization helps understand the trade-off between precision and recall for different classification thresholds, especially important for imbalanced datasets.
