### Description:

Feature selection helps identify the most relevant features for a model by eliminating irrelevant, redundant, or noisy data. This improves model performance, interpretability, and reduces overfitting. In this project, we demonstrate multiple common feature selection techniques on the Iris dataset.

Techniques Demonstrated:
- Univariate Selection (ANOVA F-test) – Select features individually based on statistical tests.
- Recursive Feature Elimination (RFE) – Recursively removes less important features.
- Tree-Based Feature Importance – Uses models like Random Forest to rank features.
- These techniques can be combined with cross-validation, or used with other datasets like customer churn or credit scoring.

## Feature Selection Techniques on Iris Dataset

This script demonstrates three different methods to perform **feature selection** on the Iris dataset using Python and scikit-learn. Feature selection helps identify the most important features that contribute to the prediction output.

### Prerequisites

```bash
pip install scikit-learn pandas
```

### Code Overview

```python
# Import required libraries
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load Iris dataset
iris = load_iris()
X = iris.data                    # Feature matrix
y = iris.target                  # Target vector
feature_names = iris.feature_names

# Convert to DataFrame for clarity
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y
```

### 1. Univariate Selection (ANOVA F-test)

```python
selector = SelectKBest(score_func=f_classif, k=2)
X_new = selector.fit_transform(X, y)
selected_features_univariate = [feature_names[i] for i in selector.get_support(indices=True)]

print("Top 2 Features (Univariate Selection - ANOVA F-test):")
print(selected_features_univariate)
```

This method uses statistical tests to select the top 2 features that are most strongly related to the target variable. ANOVA F-test is used here for classification tasks.

### 2. Recursive Feature Elimination (RFE)

```python
rfe_model = RFE(estimator=LogisticRegression(max_iter=200), n_features_to_select=2)
rfe_model.fit(X, y)
selected_features_rfe = [feature_names[i] for i in range(len(feature_names)) if rfe_model.support_[i]]

print("\nTop 2 Features (Recursive Feature Elimination):")
print(selected_features_rfe)
```

RFE recursively removes the least important features using a Logistic Regression model until only the desired number of features is left.

### 3. Feature Importance from Random Forest

```python
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)
feature_importances = rf_model.feature_importances_

# Rank and select top 2
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importances (Random Forest):")
print(importance_df)

# Display top 2
top_features_rf = importance_df["Feature"].values[:2]
print("\nTop 2 Features (Random Forest Importance):")
print(top_features_rf)
```

This method uses the built-in feature importance attribute of the Random Forest classifier to identify and rank features. The top 2 most important features are selected based on their contribution to the prediction accuracy.

---

### Summary

| Method                        | Top 2 Selected Features          |
| ----------------------------- | -------------------------------- |
| Univariate Selection          | Based on ANOVA F-test            |
| Recursive Feature Elimination | Based on Logistic Regression     |
| Random Forest Importance      | Based on ensemble decision trees |

These techniques can be used to improve model performance and reduce complexity by focusing on the most informative features.
