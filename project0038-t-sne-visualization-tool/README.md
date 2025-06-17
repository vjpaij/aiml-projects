### Description:

t-SNE (t-distributed Stochastic Neighbor Embedding) is a powerful dimensionality reduction technique particularly suited for visualizing high-dimensional data in 2D or 3D. It preserves local structure and clusters similar samples together. In this project, we create a t-SNE visualization tool using the Iris dataset, and display its 2D embedding by class.

- Transforms 4D data into 2D using t-SNE
- Reveals natural clusters and class separations
- Provides interpretable visual feedback on data structure

## t-SNE Visualization of the Iris Dataset

This script demonstrates how to perform dimensionality reduction on the Iris dataset using t-SNE (t-distributed Stochastic Neighbor Embedding) and visualize the results in 2D using Matplotlib.

### Code Explanation

```python
# Import required libraries
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
import pandas as pd
```

* `matplotlib.pyplot`: Used for plotting the results.
* `TSNE` from `sklearn.manifold`: Used to reduce the dataset's dimensionality to two dimensions.
* `load_iris` from `sklearn.datasets`: Loads the Iris dataset.
* `pandas`: Used for easier data manipulation and plotting.

```python
# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
```

* Loads the Iris dataset into memory.
* `X` contains the feature data (4D).
* `y` contains the class labels (0, 1, 2 for the three Iris species).
* `target_names` holds the actual names of the species.

```python
# Apply t-SNE for dimensionality reduction to 2D
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X)
```

* Initializes a t-SNE model with:

  * `n_components=2`: Target dimension is 2.
  * `perplexity=30`: Balances attention between local and global aspects of data.
  * `learning_rate=200`: Affects how fast the algorithm converges.
  * `n_iter=1000`: Number of iterations for optimization.
  * `random_state=42`: Ensures reproducibility.
* `fit_transform(X)` applies the dimensionality reduction.

```python
# Create a DataFrame for easy plotting
df_tsne = pd.DataFrame(X_tsne, columns=["Component 1", "Component 2"])
df_tsne['Target'] = y
```

* Wraps the t-SNE results into a pandas DataFrame for easier plotting.
* Adds the original class labels for coloring the points.

```python
# Plot t-SNE result
plt.figure(figsize=(8, 6))
for i, label in enumerate(target_names):
    subset = df_tsne[df_tsne['Target'] == i]
    plt.scatter(subset["Component 1"], subset["Component 2"], label=label, s=60)

plt.title("t-SNE Visualization of Iris Dataset")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

* Creates a scatter plot of the 2D t-SNE projection.
* Each class (Iris-setosa, Iris-versicolor, Iris-virginica) is plotted in a different color.
* Adds axis labels, a title, legend, and grid for clarity.

### Output

This visualization allows us to see how the three Iris species are separated in 2D space based on their features, after applying t-SNE. It provides an intuitive view of the clustering patterns in the data.
