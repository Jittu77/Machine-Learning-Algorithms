import matplotlib.pyplot as plt
import pandas as pd
import random
import math

class K_Means_Clustering:
    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def initialize_centroids(self, X):
        """Improved centroid initialization using k-means++"""
        self.centroids = [random.choice(X)]
        for _ in range(1, self.n_clusters):
            distances = [min([self.calculate_distance(x, centroid) for centroid in self.centroids]) for x in X]
            probabilities = [d**2 for d in distances]
            total = sum(probabilities)
            probabilities = [p / total for p in probabilities]
            self.centroids.append(random.choices(X, probabilities)[0])

    def cluster_data(self, X):
        self.initialize_centroids(X)  

        for _ in range(self.max_iter):
            cluster_labels = self.assign_labels(X)
            previous_centroids = [centroid[:] for centroid in self.centroids]  
            self.centroids = self.update_centroids(X, cluster_labels)
            
            if self.is_converged(previous_centroids, self.centroids):
                break

        return cluster_labels

    def assign_labels(self, X):
        cluster_labels = []
        for i in range(len(X)):
            distances = [self.calculate_distance(X[i], centroid) for centroid in self.centroids]
            closest_cluster = distances.index(min(distances))
            cluster_labels.append(closest_cluster)
        return cluster_labels

    def update_centroids(self, X, cluster_labels):
        new_centroids = [[0] * len(X[0]) for _ in range(self.n_clusters)]
        cluster_sizes = [0] * self.n_clusters
        
        for i in range(len(X)):
            cluster = cluster_labels[i]
            for j in range(len(X[i])):
                new_centroids[cluster][j] += X[i][j]
            cluster_sizes[cluster] += 1
        
        for i in range(self.n_clusters):
            if cluster_sizes[i] > 0:
                for j in range(len(new_centroids[i])):
                    new_centroids[i][j] /= cluster_sizes[i]
            else:
                new_centroids[i] = self.centroids[i]  
        
        return new_centroids

    def calculate_distance(self, point1, point2):
        return math.sqrt(sum((point1[i] - point2[i]) ** 2 for i in range(len(point1))))

    def is_converged(self, old_centroids, new_centroids, tolerance=1e-6):
        return all(self.calculate_distance(old_centroids[i], new_centroids[i]) <= tolerance for i in range(len(old_centroids)))


# Load dataset
df = pd.read_csv('Machine-Learning-Algorithms\student_clustering.csv')
X = [list(row) for row in df.iloc[:, :].values]

# Elbow Method
inertia_values = []
k_values = range(1, 11)

for k in k_values:
    km = K_Means_Clustering(n_clusters=k, max_iter=500)
    labels = km.cluster_data(X)
    inertia = sum(sum((X[i][d] - km.centroids[labels[i]][d]) ** 2 for d in range(len(X[0]))) for i in range(len(X)))
    inertia_values.append(inertia)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia_values, marker='o', linestyle='-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method to Find Optimal k')
plt.grid()
plt.show()

# Clustering Visualization
km = K_Means_Clustering(n_clusters=4, max_iter=500)
labels = km.cluster_data(X)

clusters = [[X[i] for i in range(len(X)) if labels[i] == c] for c in range(4)]
colors = ['red', 'blue', 'green', 'yellow']

for i, cluster in enumerate(clusters):
    plt.scatter([point[0] for point in cluster], [point[1] for point in cluster], color=colors[i], label=f"Cluster {i}")

plt.xlabel('CGPA')
plt.ylabel('IQ')
plt.title('K-Means Clustering with k-means++ Initialization')
plt.legend()
plt.show()


# Custom K-Means Inertia Calculate
custom_inertia = sum(
    sum((X[i][d] - km.centroids[labels[i]][d]) ** 2 for d in range(len(X[0])))
    for i in range(len(X))
)
print(f"Custom K-Means Inertia: {custom_inertia}")
