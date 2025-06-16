### Description:

The Receiver Operating Characteristic (ROC) curve is a graphical plot that illustrates a binary classifier’s performance as its discrimination threshold varies. It plots True Positive Rate (TPR) against False Positive Rate (FPR) and helps assess model quality. In this project, we generate and visualize the ROC curve for a logistic regression model using the Iris dataset (binary case).

- Trains a logistic regression model
- Computes TPR, FPR, AUC score
- Plots a clean and informative ROC curve

## Logistic Regression ROC Curve on Iris Dataset (Binary Classification)

This script demonstrates how to perform binary classification using logistic regression on the Iris dataset and evaluate its performance using the ROC curve and AUC score.

### Code Explanation

#### 1. **Import Required Libraries**

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
```

* `sklearn.datasets.load_iris`: Loads the Iris dataset.
* `train_test_split`: Splits data into training and testing sets.
* `LogisticRegression`: Implements logistic regression.
* `roc_curve`, `roc_auc_score`: Used for calculating the ROC curve and AUC.
* `matplotlib.pyplot`: For plotting the ROC curve.

#### 2. **Load and Prepare the Dataset**

```python
iris = load_iris()
X = iris.data
y = iris.target
```

* Loads the Iris dataset and separates the features (`X`) and target labels (`y`).

#### 3. **Filter for Binary Classification**

```python
X_binary = X[y != 2]
y_binary = y[y != 2]
```

* The Iris dataset has 3 classes. This script filters it down to 2 classes (Setosa and Versicolor) to enable binary classification.

#### 4. **Split the Data**

```python
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.3, random_state=42)
```

* Splits the binary-class data into training (70%) and testing (30%) sets.

#### 5. **Train the Logistic Regression Model**

```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

* Creates and trains a logistic regression model on the training data.

#### 6. **Predict Probabilities**

```python
y_probs = model.predict_proba(X_test)[:, 1]
```

* Predicts class probabilities for the test set and selects the probabilities for class 1 (Versicolor).

#### 7. **Calculate ROC Curve and AUC Score**

```python
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)
```

* Computes the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds for the ROC curve.
* Calculates the Area Under the Curve (AUC) score, a performance metric for classification.

#### 8. **Plot the ROC Curve**

```python
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, color='blue', label=f'Logistic Regression (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

* Plots the ROC curve for the logistic regression model and a baseline "random classifier".
* Displays the AUC score in the legend.

### Summary

This code provides a clear and concise example of:

* Preparing a dataset for binary classification
* Training a logistic regression model
* Evaluating the model using ROC curve and AUC
* Visualizing the performance using matplotlib
