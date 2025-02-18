import pandas as pd
import KMeanClustering as kmc
class CustomSilhouetteScore:
    @staticmethod
    def compute_silhouette_score(X, labels, centroids):
        """Calculate the silhouette score manually."""
        n = len(X)
        silhouette_scores = []

        for i in range(n):
            own_cluster = labels[i]
            own_cluster_points = [X[j] for j in range(n) if labels[j] == own_cluster and j != i]
            
            # Compute a(i) - Mean intra-cluster distance
            if own_cluster_points:
                a_i = sum(kmc.K_Means_Clustering().calculate_distance(X[i], p) for p in own_cluster_points) / len(own_cluster_points)
            else:
                a_i = 0  # If only one point in the cluster

            # Compute b(i) - Minimum mean inter-cluster distance
            other_clusters = set(labels) - {own_cluster}
            b_i_values = []

            for cluster in other_clusters:
                cluster_points = [X[j] for j in range(n) if labels[j] == cluster]
                if cluster_points:
                    b_i = sum(kmc.K_Means_Clustering().calculate_distance(X[i], p) for p in cluster_points) / len(cluster_points)
                    b_i_values.append(b_i)

            b_i = min(b_i_values) if b_i_values else 0  # If there are no other clusters

            # Compute silhouette score for point i
            silhouette_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
            silhouette_scores.append(silhouette_i)

        # Compute the overall silhouette score
        return sum(silhouette_scores) / n if n > 0 else 0


# Load dataset
df = pd.read_csv('Machine-Learning-Algorithms/student_clustering.csv')
X = [list(row) for row in df.iloc[:, :].values]

# Run clustering
optimal_k = 4  # Assume we got this from the Elbow Method
kmeans = kmc.K_Means_Clustering(n_clusters=optimal_k, max_iter=500)
labels = kmeans.cluster_data(X)

# Compute silhouette score
silhouette_score = CustomSilhouetteScore.compute_silhouette_score(X, labels, kmeans.centroids)
print(f"Custom Silhouette Score: {silhouette_score}")
