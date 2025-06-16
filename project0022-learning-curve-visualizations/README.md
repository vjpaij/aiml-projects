### Description:

A learning curve shows how a model’s performance changes as the size of the training dataset increases. It helps diagnose whether a model suffers from high bias (underfitting) or high variance (overfitting). In this project, we generate learning curves for a Logistic Regression model using the Iris dataset.

- Visualizes model performance vs. training set size
- Helps identify:
    - Underfitting: both curves are low and close
    - Overfitting: big gap between training and validation scores
    - Good fit: both curves are high and close

## Learning Curve for Logistic Regression on the Iris Dataset

This script demonstrates how to generate and visualize a learning curve using logistic regression on the popular Iris dataset. A learning curve helps in understanding the model performance with increasing training set size, identifying potential underfitting or overfitting issues.

### Code Breakdown

#### 1. **Import Required Libraries**

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
```

* `load_iris`: Loads the Iris dataset.
* `LogisticRegression`: The classification model used.
* `learning_curve`: Generates scores for training and validation sets at varying sizes.
* `matplotlib.pyplot`: For plotting the learning curve.
* `numpy`: For numerical operations.

#### 2. **Load the Dataset**

```python
iris = load_iris()
X = iris.data
y = iris.target
```

* `X`: Features (sepal and petal measurements).
* `y`: Target class (three Iris species).

#### 3. **Initialize the Model**

```python
model = LogisticRegression(max_iter=200)
```

* A logistic regression model is created with a maximum of 200 iterations to ensure convergence.

#### 4. **Generate Learning Curve Data**

```python
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy',
    shuffle=True,
    random_state=42
)
```

* Trains the model with training set sizes ranging from 10% to 100%.
* 5-fold cross-validation is used.
* Scores are evaluated using accuracy.
* Data is shuffled for better randomness.

#### 5. **Compute Mean and Standard Deviation**

```python
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)
```

* Averages and standard deviations are calculated across cross-validation folds for plotting.

#### 6. **Plot the Learning Curve**

```python
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
plt.plot(train_sizes, test_mean, 'o-', color='green', label='Cross-validation score')

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='green')

plt.title("Learning Curve - Logistic Regression (Iris Dataset)")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()
```

* Training and cross-validation scores are plotted.
* Shaded regions represent ±1 standard deviation.
* The plot is customized with titles, labels, legends, and layout for better readability.

### Purpose

This visualization helps determine:

* Whether more training data would improve performance.
* If the model is overfitting (high training, low validation score).
* If the model is underfitting (both scores low).

---

This script is especially useful for machine learning practitioners who want to evaluate model learning behavior early in the development process.
