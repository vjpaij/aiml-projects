import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
 
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
 
# ---- PCA (Linear) ----
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
 
# ---- t-SNE (Non-linear) ----
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)
 
# ---- UMAP (Manifold Learning) ----
umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X)
 
# ---- Plot Function ----
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
    plt.savefig(f"{title}.png")  # Save the plot as a PNG file
 
# Plot all techniques
plot_embedding(X_pca, "PCA - Linear Projection")
plot_embedding(X_tsne, "t-SNE - Nonlinear Embedding")
plot_embedding(X_umap, "UMAP - Uniform Manifold Approximation")