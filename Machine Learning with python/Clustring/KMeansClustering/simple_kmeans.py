import numpy as np

# Define the points and their initial cluster assignments
points = {
    'A1': np.array([1, 2]),
    'A2': np.array([6, 3]),
    'A3': np.array([8, 4]),
    'A4': np.array([2, 5]),
    'A5': np.array([7, 5]),
    'A6': np.array([4 ,6]),
    'A7': np.array([5 ,7]),
    'A8': np.array([2, 8])
}

# Initial clusters
clusters = {
    3: ['A1', 'A6'],
    2: ['A3', 'A4', 'A8'],
    1: ['A2', 'A5', 'A7']
}

def calculate_centroids(clusters, points):
    centroids = {}
    for key, members in clusters.items():
        centroids[key] = np.mean([points[member] for member in members], axis=0)
    return centroids

def assign_points_to_clusters(centroids, points):
    new_clusters = {key: [] for key in centroids}
    for point_key, coordinates in points.items():
        closest_cluster = min(centroids, key=lambda c: np.linalg.norm(coordinates - centroids[c]))
        new_clusters[closest_cluster].append(point_key)
    return new_clusters

centroids = calculate_centroids(clusters, points)

print("Initial centroids:")
for key, value in centroids.items():
    print(f"Cluster {key}: {value}")

for i in range(3): 
    clusters = assign_points_to_clusters(centroids, points)
    centroids = calculate_centroids(clusters, points)
    print(f"\nAfter iteration {i + 1}:")

    print("Centroids:")
    for key, value in centroids.items():
        print(f"Cluster {key}: {value}")
    print("Cluster assignments:")
    for key, value in clusters.items():
        print(f"Cluster {key}: {value}")

