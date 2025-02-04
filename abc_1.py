from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
import random
import math

class KMeans:
    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit_predict(self, X):
        # Select random initial centroids
        random_indices = random.sample(range(len(X)), self.n_clusters)
        self.centroids = [X[i][:] for i in random_indices]  # Deep copy to avoid reference issues

        for _ in range(self.max_iter):
            # Assign clusters
            cluster_group = self.assign_clusters(X)
            old_centroids = [centroid[:] for centroid in self.centroids]  # Copy for convergence check

            # Move centroids
            self.centroids = self.move_centroids(X, cluster_group)

            # Check convergence
            if self.has_converged(old_centroids, self.centroids):
                break

        return cluster_group

    def assign_clusters(self, X):
        cluster_group = []
        for i in range(len(X)):
            distances = []
            for j in range(len(self.centroids)):
                distances.append(self.euclidean_distance(X[i], self.centroids[j]))  # Compute distances manually
            
            min_index = 0
            min_distance = distances[0]
            for k in range(1, len(distances)):  # Find index of minimum distance manually
                if distances[k] < min_distance:
                    min_distance = distances[k]
                    min_index = k
            cluster_group.append(min_index)  # Assign to nearest centroid
        return cluster_group

    def move_centroids(self, X, cluster_group):
        # Initialize new centroids as zero vectors
        new_centroids = [[0] * len(X[0]) for _ in range(self.n_clusters)]
        counts = [0] * self.n_clusters  # Track number of points in each cluster

        # Sum up points in each cluster
        for i in range(len(X)):
            cluster = cluster_group[i]
            for j in range(len(X[i])):
                new_centroids[cluster][j] += X[i][j]
            counts[cluster] += 1

        # Compute mean for each cluster
        for i in range(self.n_clusters):
            if counts[i] > 0:  # Avoid division by zero
                for j in range(len(new_centroids[i])):
                    new_centroids[i][j] /= counts[i]
            else:
                new_centroids[i] = self.centroids[i]  # Keep old centroid if no points assigned

        return new_centroids

    def euclidean_distance(self, point1, point2):
        total = 0
        for i in range(len(point1)):
            total += (point1[i] - point2[i]) ** 2  # Squared difference
        return math.sqrt(total)

    def has_converged(self, old_centroids, new_centroids, tolerance=1e-6):
        for i in range(len(old_centroids)):
            if self.euclidean_distance(old_centroids[i], new_centroids[i]) > tolerance:
                return False
        return True

# Load dataset
df = pd.read_csv('student_clustering.csv')
X = df.iloc[:, :].values  # Convert to list of lists

# Convert DataFrame to pure Python list of lists (since NumPy is removed)
X = [list(row) for row in X]

# Elbow Method to Find Optimal k
inertia_values = []
k_values = range(1, 11)  # Testing k from 1 to 10

for k in k_values:
    km = KMeans(n_clusters=k, max_iter=500)
    y_means = km.fit_predict(X)
    
    # Corrected inertia calculation (sum of squared distances)
    inertia = sum(
        sum((X[i][d] - km.centroids[y_means[i]][d]) ** 2 for d in range(len(X[0])))
        for i in range(len(X))
    )
    inertia_values.append(inertia)

# Plot elbow method graph
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia_values, marker='o', linestyle='-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method to Find Optimal k')
plt.grid()
plt.show()

# Clustering Visualization
km = KMeans(n_clusters=4, max_iter=500)
y_means = km.fit_predict(X)

# Extract clusters for plotting (Manual list filtering)
cluster_0 = [X[i] for i in range(len(X)) if y_means[i] == 0]
cluster_1 = [X[i] for i in range(len(X)) if y_means[i] == 1]
cluster_2 = [X[i] for i in range(len(X)) if y_means[i] == 2]
cluster_3 = [X[i] for i in range(len(X)) if y_means[i] == 3]

# Scatter plot
plt.scatter([point[0] for point in cluster_0], [point[1] for point in cluster_0], color='red', label="Cluster 0")
plt.scatter([point[0] for point in cluster_1], [point[1] for point in cluster_1], color='blue', label="Cluster 1")
plt.scatter([point[0] for point in cluster_2], [point[1] for point in cluster_2], color='green', label="Cluster 2")
plt.scatter([point[0] for point in cluster_3], [point[1] for point in cluster_3], color='yellow', label="Cluster 3")

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering (Without NumPy)')
plt.legend()
plt.show()
