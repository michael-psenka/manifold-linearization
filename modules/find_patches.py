import torch

def knn(X):
	n, d = X.shape

	centroids = list()
	points_in_each_cluster = list()
	clusters_for_each_point = list(list() for _ in range(n))

	# stupid way: points are in cluster with their two nearest neighbors
	# obviously not good, but there for testing if code works with this method
	pairwise_distances = [(idx1, idx2, torch.linalg.norm(X[idx1] - X[idx2])) for idx1 in range(n) for idx2 in range(n) if idx2 != idx1]
	key_fn = lambda a: (a[0], a[2])
	sorted_pairwise_distances = sorted(pairwise_distances, key=key_fn)
	for i in range(n):
		neighbor_1 = sorted_pairwise_distances[(n - 1) * i]
		neighbor_2 = sorted_pairwise_distances[(n - 1) * i + 1]
		cluster = [X[i], X[neighbor_1[1]], X[neighbor_2[1]]]
		centroids.append(sum(cluster) / 3.0)
		points_in_each_cluster.append(cluster)
		clusters_for_each_point[i].append(i)
		clusters_for_each_point[neighbor_1[1]].append(i)
		clusters_for_each_point[neighbor_2[1]].append(i)

	return centroids, points_in_each_cluster, clusters_for_each_point