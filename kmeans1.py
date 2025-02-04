import random
import numpy as np
#(1) decide #clusters
#(2) select random centrids
#(3) Assign clusters
#(4) move centroids
#(5) check finish

class KMeans:
    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit_predict(self, X):
        random_index = random.sample(range(0, X.shape[0]), self.n_clusters)
        self.centroids = X[random_index]

        for i in range(self.max_iter):
            # Assign clusters
            cluster_group = self.assign_clusters(X)
            old_centroids = self.centroids.copy()

            # Move centroids
            self.centroids = self.move_centroids(X, cluster_group)

            # Check convergence
            if np.allclose(old_centroids, self.centroids):
                break

        return cluster_group

    def assign_clusters(self, X):
        cluster_group = []

        for row in X:
            distances = [np.linalg.norm(row - centroid) for centroid in self.centroids]
            cluster_group.append(np.argmin(distances))

        return np.array(cluster_group)

    def move_centroids(self, X, cluster_group):
        new_centroids = []
        for cluster in range(self.n_clusters):
            cluster_points = X[np.where(cluster_group == cluster)]
            if len(cluster_points) > 0:
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                new_centroids.append(self.centroids[cluster])  # Keep the old centroid if empty

        return np.array(new_centroids)
