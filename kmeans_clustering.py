# kmeans_clustering.py
from sklearn.cluster import KMeans
import numpy as np

def perform_kmeans_clustering(data_points, n_clusters):
    
    X = np.array(data_points)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    
    # Get the cluster centers and labels
    cluster_centers = kmeans.cluster_centers_.tolist()
    labels = kmeans.labels_.tolist()

    return {'cluster_centers': cluster_centers, 'labels': labels}
