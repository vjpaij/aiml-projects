# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
 
# Load the Iris dataset
iris = load_iris()
X = iris.data        # 4 features
y = iris.target      # 3 classes: setosa, versicolor, virginica
labels = iris.target_names
 
# Apply PCA to reduce to 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
 
# Print explained variance
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance retained:", sum(pca.explained_variance_ratio_))
 
# Plot the 2D PCA result
plt.figure(figsize=(8, 6))
for i in range(len(labels)):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=labels[i])
    
plt.title("PCA on Iris Dataset (2D Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("pca_iris.png")  # Save the plot as an image file