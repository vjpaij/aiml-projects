### Description:

Ensemble methods combine multiple machine learning models to improve predictive performance, reduce overfitting, and enhance generalization. In this project, we implement and compare three popular ensemble techniques: Bagging (Random Forest), Boosting (Gradient Boosting), and Voting Classifier using the Iris dataset.

- Random Forest: Aggregates many decision trees (bagging).
- Gradient Boosting: Builds trees sequentially to correct errors (boosting).
- Voting Classifier: Combines predictions from multiple different models.

## Ensemble Learning on Iris Dataset

This code demonstrates the use of **ensemble learning techniques** on the famous **Iris dataset** using three popular methods: **Bagging (Random Forest)**, **Boosting (Gradient Boosting)**, and **Voting Classifier**. The goal is to compare the performance of these approaches in classifying iris species.

### 1. Importing Required Libraries

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
```

* `sklearn.datasets.load_iris`: Loads the Iris dataset.
* `train_test_split`: Splits data into training and testing sets.
* `RandomForestClassifier`, `GradientBoostingClassifier`, `VotingClassifier`: Ensemble classifiers.
* `LogisticRegression`, `SVC`: Individual classifiers used in Voting.
* `accuracy_score`: Measures model performance.

### 2. Loading and Preparing Data

```python
iris = load_iris()
X = iris.data
y = iris.target
```

* Loads feature matrix `X` and target vector `y`.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

* Splits the dataset into 70% training and 30% testing data.

### 3. Bagging: Random Forest

```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_preds):.2f}")
```

* Uses Random Forest, which builds multiple decision trees and aggregates their results.
* Trained and tested for accuracy.

### 4. Boosting: Gradient Boosting

```python
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
gb_preds = gb.predict(X_test)
print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, gb_preds):.2f}")
```

* Applies Gradient Boosting to sequentially improve model performance.
* Each new tree corrects errors of the previous ones.

### 5. Voting Classifier (Hard Voting)

```python
log_clf = LogisticRegression(max_iter=200)
svc_clf = SVC(probability=True)
voting_clf = VotingClassifier(estimators=[
    ('lr', log_clf), ('svc', svc_clf), ('rf', rf)
], voting='hard')

voting_clf.fit(X_train, y_train)
voting_preds = voting_clf.predict(X_test)
print(f"Voting Classifier Accuracy: {accuracy_score(y_test, voting_preds):.2f}")
```

* Combines predictions from Logistic Regression, SVM, and Random Forest.
* **Hard Voting** means predicting the class with the majority vote from classifiers.

### Summary

This script shows how ensemble techniques can be used to potentially improve prediction accuracy over single models. Each method leverages multiple models in different ways:

* **Random Forest**: Parallel trees (Bagging).
* **Gradient Boosting**: Sequential improvement.
* **Voting Classifier**: Aggregates different algorithms.

Ideal for educational demonstrations or initial benchmarking in classification problems.
