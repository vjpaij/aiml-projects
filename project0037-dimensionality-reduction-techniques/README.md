### Description:

Dimensionality reduction is the process of reducing the number of input variables (features) in a dataset while preserving essential information. It is useful for visualization, noise reduction, and improving model performance. This project demonstrates and compares three popular techniques: PCA, t-SNE, and UMAP using the Iris dataset.

- PCA reduces dimensionality linearly, preserving global structure.
- t-SNE is great for visualizing local clusters.
- UMAP combines advantages of both and is great for general-purpose embeddings.

## Dimensionality Reduction Visualization on Iris Dataset

This script demonstrates how to apply and visualize three popular dimensionality reduction techniques on the classic Iris dataset:

* **PCA (Principal Component Analysis)** – a linear method
* **t-SNE (t-distributed Stochastic Neighbor Embedding)** – a nonlinear embedding
* **UMAP (Uniform Manifold Approximation and Projection)** – a manifold learning approach

### Libraries Used

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
```

### Step 1: Load the Iris Dataset

```python
iris = load_iris()
X = iris.data          # Feature data (shape: 150 samples x 4 features)
y = iris.target        # Class labels (0, 1, 2)
target_names = iris.target_names  # Class names (['setosa', 'versicolor', 'virginica'])
```

### Step 2: Apply Dimensionality Reduction Techniques

#### PCA (Linear Projection)

```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
```

Performs linear dimensionality reduction to map the original 4D data into 2D.

#### t-SNE (Nonlinear Embedding)

```python
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)
```

t-SNE maps high-dimensional data to 2D by minimizing divergence between probability distributions. `perplexity` controls the balance between local and global structure.

#### UMAP (Manifold Learning)

```python
umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X)
```

UMAP builds a high-dimensional graph and optimizes a 2D representation that preserves both local and global structure.

### Step 3: Visualization Function

```python
def plot_embedding(X_embed, title):
    plt.figure(figsize=(6, 4))
    for i, label in enumerate(target_names):
        plt.scatter(X_embed[y == i, 0], X_embed[y == i, 1], label=label)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
```

This helper function takes in 2D projected data and plots it, color-coded by class.

### Step 4: Generate Plots

```python
plot_embedding(X_pca, "PCA - Linear Projection")
plot_embedding(X_tsne, "t-SNE - Nonlinear Embedding")
plot_embedding(X_umap, "UMAP - Uniform Manifold Approximation")
```

Each call generates a scatter plot for a specific dimensionality reduction technique.

---

### Summary

This script helps compare how different dimensionality reduction techniques perform when projecting high-dimensional data into 2D. PCA captures linear structure, t-SNE captures local non-linear relationships, and UMAP captures both local and global structure efficiently.
