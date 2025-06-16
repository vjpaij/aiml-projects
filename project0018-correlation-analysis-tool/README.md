### Description:

Correlation analysis is used to measure and visualize the strength and direction of relationships between numerical variables. This project builds a simple tool to compute and visualize correlations using Pearson correlation coefficients and a heatmap. It helps in feature selection and understanding data relationships.

- Calculates pairwise correlations between features
- Prints correlation matrix
- Visualizes relationships using a color-coded heatmap

### Correlation Matrix and Heatmap Visualization

This Python script demonstrates how to compute and visualize the correlation matrix of a dataset using Pandas, Seaborn, and Matplotlib. The example uses a small in-memory dataset for simplicity, but it can be easily adapted to load data from a CSV or other sources.

#### Code Explanation

```python
# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

* `pandas`: Used for data manipulation and analysis.
* `seaborn`: A statistical data visualization library built on top of Matplotlib.
* `matplotlib.pyplot`: Used for plotting graphs and figures.

```python
# Sample dataset (can also be replaced with any CSV)
data = {
    "Age": [25, 32, 47, 51, 62, 23, 43, 36, 52, 48],
    "Income": [45000, 54000, 72000, 80000, 88000, 43000, 69000, 62000, 85000, 79000],
    "SpendingScore": [30, 40, 50, 60, 65, 25, 45, 42, 58, 62],
    "CreditScore": [650, 680, 720, 700, 710, 640, 690, 685, 705, 695]
}
```

* A dictionary containing sample numerical data for four attributes: `Age`, `Income`, `SpendingScore`, and `CreditScore`.

```python
# Convert to DataFrame
df = pd.DataFrame(data)
```

* Converts the dictionary into a Pandas DataFrame, which provides easy handling and analysis of tabular data.

```python
# Calculate the correlation matrix using Pearson method
correlation_matrix = df.corr()
```

* Calculates the pairwise Pearson correlation coefficients between all numerical features in the DataFrame.
* The `.corr()` function by default uses the Pearson method.

```python
# Print correlation matrix
print("Correlation Matrix:\n")
print(correlation_matrix)
```

* Displays the correlation matrix in the console.

```python
# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
```

* Creates a heatmap using Seaborn to visually represent the correlation matrix.
* `annot=True` shows the correlation values inside the heatmap cells.
* `cmap='coolwarm'` sets the color palette.
* `fmt=".2f"` formats the float values to two decimal places.
* `linewidths=0.5` separates the cells with lines.
* `tight_layout()` ensures proper spacing in the layout.

#### Output

* A printed correlation matrix.
* A heatmap visualizing the correlation between the dataset features.

This script is useful for identifying relationships between variables, which can inform feature selection or preprocessing in data analysis and machine learning workflows.
