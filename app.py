from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from kmeans import KMeans
import pandas as pd
import numpy as np
#centroids = [(-5,-5),(5,5),(-2.5,2.5),(2.5,-2.5)]
#cluster_std = [1,1,1,1]

#X,y = make_blobs(n_samples=100,cluster_std=cluster_std,centers=centroids,n_features=2,random_state=2)
  
#plt.scatter(X[:,0],X[:,1])
# Load dataset
df = pd.read_csv('student_clustering.csv')
X = df.iloc[:,:].values

# Elbow Method to Find Optimal k
inertia_values = []
k_values = range(1, 11)  # Testing k from 1 to 10

for k in k_values:
    km = KMeans(n_clusters=k, max_iter=500)
    y_means = km.fit_predict(X)
    
    # Corrected inertia calculation: sum of squared distances from each point to its assigned centroid
    inertia = sum(np.linalg.norm(X[i] - km.centroids[y_means[i]])**2 for i in range(len(X)))
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
km = KMeans(n_clusters=4,max_iter=500)
y_means = km.fit_predict(X)

# Plot clusters
plt.scatter(X[y_means == 0,0],X[y_means == 0,1],color='red')
print("printed")
plt.scatter(X[y_means == 1,0],X[y_means == 1,1],color='blue')
plt.scatter(X[y_means == 2,0],X[y_means == 2,1],color='green')
plt.scatter(X[y_means == 3,0],X[y_means == 3,1],color='yellow')
plt.show()