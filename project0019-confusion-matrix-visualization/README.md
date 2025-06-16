### Description:

A confusion matrix gives a detailed breakdown of classification results—how many predictions were true positives, false positives, true negatives, and false negatives. This project implements a simple classification model and visualizes its confusion matrix using Seaborn heatmaps for better interpretability.

What This Project Shows:
- How the model performed in terms of:
    - True Positives (TP)
    - False Positives (FP)
    - True Negatives (TN)
    - False Negatives (FN)

- Visualizes results using a heatmap for quick interpretation

## Logistic Regression on Iris Dataset (Binary Classification)

This example demonstrates how to perform a binary classification using the **Iris dataset** with **Logistic Regression**, and visualize the performance using a **confusion matrix heatmap**.

### 📦 Required Libraries

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
```

### 📊 Step 1: Load the Iris Dataset

```python
iris = load_iris()
X = iris.data
y = iris.target
labels = iris.target_names
```

* `X` contains feature vectors.
* `y` contains class labels: 0 (Setosa), 1 (Versicolor), 2 (Virginica).
* `labels` contains class names.

### 🎯 Step 2: Filter for Binary Classification

```python
X_binary = X[y != 2]
y_binary = y[y != 2]
```

We exclude class `2` (Virginica) for a binary classification task between:

* Setosa (0)
* Versicolor (1)

### 🧪 Step 3: Split Data into Training and Test Sets

```python
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.3, random_state=42)
```

* 70% of data is used for training, 30% for testing.

### 🤖 Step 4: Train the Logistic Regression Model

```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

* The logistic regression classifier is trained on the training data.

### 🔍 Step 5: Make Predictions

```python
y_pred = model.predict(X_test)
```

* Use the trained model to predict test set labels.

### 📉 Step 6: Generate and Display Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=["Actual Setosa", "Actual Versicolor"],
                         columns=["Predicted Setosa", "Predicted Versicolor"])
```

* The confusion matrix counts correct and incorrect predictions.
* It's converted to a DataFrame for easy labeling.

```python
print("Confusion Matrix:\n")
print(cm_df)
```

* This prints the matrix to the console.

### 📊 Step 7: Visualize with Seaborn Heatmap

```python
plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix Visualization")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()
```

* Uses seaborn to render the confusion matrix as a heatmap for clearer interpretation.

### ✅ Output

You will see a heatmap indicating:

* True positives (correctly predicted classes)
* False positives and false negatives (misclassifications)

This is a compact and effective way to visualize model performance on a binary classification problem.
