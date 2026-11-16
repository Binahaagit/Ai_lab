# EXPERIMENT 23: K-MEANS CLUSTERING
# Perform K-Means clustering on a dataset and visualize the clusters for K = 2, 3 and 4.

#CODE:
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('csv1.csv')
X = df.iloc[:, :2].values

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

axes[0].scatter(X[:, 0], X[:, 1], c='gray', s=50)
axes[0].set_title("Before Clustering", fontsize=14)

k_values = [2, 3, 4]

for idx, k in enumerate(k_values, start=1):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    axes[idx].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
    axes[idx].scatter(
        centroids[:, 0], centroids[:, 1],
        c='red', marker='X', s=150
    )
    axes[idx].set_title(f"K-Means (k={k})", fontsize=14)

plt.tight_layout()
plt.show()






## EXPLANATION ##
# 1. Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 2. Load the dataset
df = pd.read_csv('csv1.csv')

# Select the first two columns as input features.
# iloc[:, :2] selects all rows and the first two columns.
# .values converts the DataFrame into a NumPy array.
X = df.iloc[:, :2].values

# 3. Create a 2x2 grid to display four plots.
# One plot for original data and three for different K values.
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Convert the 2D axes array into a 1D array.
# This makes subplot indexing easier: axes[0], axes[1], etc.
axes = axes.flatten()

# 4. Plot original data before clustering.
# X[:, 0] = first feature (X-axis)
# X[:, 1] = second feature (Y-axis)
# c = colour, s = marker size
axes[0].scatter(X[:, 0], X[:, 1], c='gray', s=50)
axes[0].set_title("Before Clustering", fontsize=14)

# 5. Perform K-Means for different numbers of clusters.
k_values = [2, 3, 4]

# enumerate() gives both the subplot index and K value.
# start=1 because axes[0] is already used for original data.
for idx, k in enumerate(k_values, start=1):

    # Create the K-Means model.
    # n_clusters = number of clusters.
    # random_state = ensures reproducible initialization.
    # n_init = run with 10 initializations and keep the best result.
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    # fit_predict() trains the model and returns a cluster label
    # for every data point.
    labels = kmeans.fit_predict(X)

    # Get the final centroid coordinates of each cluster.
    centroids = kmeans.cluster_centers_

    # 6. Plot data points with colours based on cluster labels.
    axes[idx].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')

    # Plot centroids as red X markers.
    axes[idx].scatter(
        centroids[:, 0], centroids[:, 1],
        c='red', marker='X', s=150
    )

    # Set title for each clustering result.
    axes[idx].set_title(f"K-Means (k={k})", fontsize=14)

# 7. Automatically adjust spacing between subplots.
plt.tight_layout()

# Display all four plots.
plt.show()


# IMPORTANT CONCEPTS TO REMEMBER:

# 1. K-Means is an UNSUPERVISED learning algorithm.
# 2. It is used for CLUSTERING.
# 3. K represents the number of clusters.
# 4. A centroid is the mean position of points in a cluster.
# 5. Euclidean distance is commonly used to assign points to their nearest centroid.
# 6. K-Means repeats assignment and centroid calculation until convergence.
# 7. WCSS / Inertia measures the sum of squared distances between points and their assigned centroids.
# 8. K-Means tries to minimize WCSS.
# 9. fit_predict() fits the model and returns cluster labels.
# 10. cluster_centers_ returns centroid coordinates.
# 11. Increasing K generally decreases or maintains WCSS.(Within-Cluster Sum of Squares)
# 12. Elbow method helps select a suitable K.

