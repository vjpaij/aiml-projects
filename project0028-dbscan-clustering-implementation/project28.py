# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
 
# Generate a synthetic dataset with non-linear (moon-shaped) clusters
X, _ = make_moons(n_samples=300, noise=0.08, random_state=42)
 
# Visualize raw data
plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], s=30, color='gray')
plt.title("Raw Moon-Shaped Data")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("raw_moon_data.png")
 
# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.2, min_samples=5)  # eps = neighborhood radius, min_samples = density threshold
labels = dbscan.fit_predict(X)
 
# Plot DBSCAN results
plt.figure(figsize=(6, 4))
unique_labels = set(labels)
colors = [plt.cm.Set1(i / float(len(unique_labels))) for i in unique_labels]
 
for label, color in zip(unique_labels, colors):
    class_member_mask = (labels == label)
    plt.scatter(X[class_member_mask, 0], X[class_member_mask, 1], s=30, label=f'Cluster {label}' if label != -1 else 'Noise', color=color)
 
plt.title("DBSCAN Clustering Results")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("dbscan_clustering_results.png")