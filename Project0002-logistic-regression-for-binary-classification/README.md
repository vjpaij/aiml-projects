###Description:

Logistic Regression is a supervised learning algorithm used for binary classification problems—where the output is either 0 or 1 (e.g., spam vs. ham, disease vs. no disease). It models the probability that a given input belongs to a particular class using a sigmoid function. In this project, we’ll use it to classify whether a student will pass (1) or fail (0) based on study hours.

This basic logistic regression classifier shows how even a simple model can effectively separate binary outcomes. For real-world applications, you can use it for problems like disease prediction, customer churn, or email classification.

## 📘 Logistic Regression: Pass/Fail Prediction Based on Hours Studied

This example demonstrates how to use **logistic regression** to predict whether a student passes or fails an exam based on the number of hours studied.

### 📦 Requirements

```bash
pip install numpy matplotlib scikit-learn
```

---

### 🔍 Description

We simulate a dataset representing how many hours each student studied and whether they passed (1) or failed (0). Using `scikit-learn`, we build a **logistic regression model**, train it, and visualize the results with a sigmoid curve.

---

### 🧐 Code Explanation

```python
# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
```

* **NumPy**: for handling numerical arrays.
* **Matplotlib**: for plotting data and sigmoid curve.
* **LogisticRegression**: for binary classification.
* **classification\_report**: for model performance summary.

---

```python
# Simulated data: Hours studied (X) vs. pass/fail (y)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
```

* `X`: Number of hours studied (feature).
* `y`: Binary labels where `1 = Pass`, `0 = Fail`.

---

```python
# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X, y)
```

* Fit the logistic regression model to the data.

---

```python
# Predict probabilities and labels
y_prob = model.predict_proba(X)[:, 1]  # Probability of "Pass"
y_pred = model.predict(X)              # Predicted labels (0 or 1)
```

* `predict_proba`: gives class probabilities.
* `predict`: gives binary predictions.

---

```python
# Print model parameters and evaluation metrics
print("Model Coefficient (Slope):", model.coef_[0][0])
print("Model Intercept:", model.intercept_[0])
print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=["Fail", "Pass"]))
```

* Shows the model's slope and intercept.
* `classification_report`: shows precision, recall, f1-score, and accuracy.

---

```python
# Make prediction for a new student
test_hours = [[5.5]]
predicted = model.predict(test_hours)[0]
probability = model.predict_proba(test_hours)[0][1]

print(f"\nStudent who studied 5.5 hours is predicted to: {'Pass' if predicted else 'Fail'}")
print(f"Probability of passing: {probability:.2f}")
```

* Predicts if a student who studied 5.5 hours will pass.
* Displays the associated probability.

---

```python
# Plot sigmoid curve and data
hours_range = np.linspace(0, 11, 100).reshape(-1, 1)
probabilities = model.predict_proba(hours_range)[:, 1]

plt.plot(hours_range, probabilities, color='blue', label='Probability of Passing')
plt.scatter(X, y, color='red', label='Training Data')
plt.axhline(0.5, color='gray', linestyle='--', label='Decision Boundary (0.5)')
plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression - Pass/Fail Prediction")
plt.legend()
plt.grid(True)
plt.show()
```

* Plots:

  * **Sigmoid curve** showing predicted probability.
  * **Data points** from the dataset.
  * **Decision boundary** at 0.5 probability.

---

### 📊 Output Example

```
Model Coefficient (Slope): 1.078
Model Intercept: -5.093

Classification Report:
              precision    recall  f1-score   support
        Fail       1.00      1.00      1.00         4
        Pass       1.00      1.00      1.00         6

Student who studied 5.5 hours is predicted to: Pass
Probability of passing: 0.63
```

---

### 🎯 Summary

This simple project shows how logistic regression can be applied to real-world-like binary classification problems such as predicting pass/fail outcomes based on continuous input variables like hours studied. The sigmoid output curve helps interpret how confident the model is across different input values.
