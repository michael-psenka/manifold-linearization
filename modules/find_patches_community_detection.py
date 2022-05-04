import copy

import networkx as nx
import networkx.algorithms.community as nx_comm
import torch

from sklearn.neighbors import kneighbors_graph


class CommunityDetection:
    def __init__(self, X: torch.Tensor, k: int = -1, eps: float = 1e-8):
        """
        Builds KNN graph, where k = O(sqrt(n)). Vertices are n data points xi, edges are 1/d(xi, xj)
        if xj is one of the k nearest neighbors of xi.

        :param X: data matrix, (d, n)
        :param eps: perturbation to avoid divide-by-zero error, default=1e-8 should be good

        properties:
        knn_graph: networkx graph, where nodes are data points, and edges are 1/d(xi, xj) for k nearest neighbors
        """
        d, n = X.shape

        # if unspecified, default to sqrt(n)
        if k == -1:
            k = min(int(n ** 0.5) + 1, n)
        # EXPLICIT COMPUTATION
        # pairwise_distances = torch.cdist(X, X)
        # sorted_pairwise_distances, sorted_indices = torch.sort(pairwise_distances, dim=1)
        # sorted_pairwise_distances = sorted_pairwise_distances.tolist()
        # sorted_indices = sorted_indices.tolist()

        # self.knn_graph = nx.Graph()
        # for i in range(n):
        #     for j in range(1, k):
        #         self.knn_graph.add_edge(i, sorted_indices[i][j], weight=1/(sorted_pairwise_distances[i][j] + eps))

        # VECTORIZED COMPUTATION
        # output is a scipy sparse matrix
        knn_graph = kneighbors_graph(X.T, k, mode='distance', include_self=False)
        # multiplicative inverse of all nonzero entries
        knn_data = knn_graph.data
        knn_graph.data = 1 / (knn_data + eps)
        # convert to networkx
        knn_graph = nx.from_scipy_sparse_array(knn_graph)

        self.knn_graph = knn_graph

    def find_communities(self):
        """
        Finds the communities in the KNN graph via Louvain community detection algorithm.
        How this works: lots of points which are very close to each other are considered a "community", and
        regions which are far away are considered different communities. Finds best number of communities
        and partitions the vertices into these communities.

        NOTE: this is strict partition, no overlap.

        :return: a list of index sets where each index set is a community
        """
        return nx_comm.louvain_communities(self.knn_graph)

    def find_merge_path(self, communities):
        """
        Finds the best order to merge communities in, by checking which merge order maximizes the modularity.
        This is done naively, so it's cubic in the number of communities, but this is usually <= 100 so this isn't
        any kind of issue.

        :param communities: list of index sets
        :return: merge path
        """
        merge_path = []
        current_neighborhood_indices = [{i} for i in range(len(communities))]
        current_communities = copy.deepcopy(communities)
        merge_path.append(copy.deepcopy(current_neighborhood_indices))

        while len(current_neighborhood_indices) > 1:
            max_modularity = -float('inf')
            max_modularity_ij = (-1, -1)
            for i in range(len(current_neighborhood_indices)):
                for j in range(i):
                    # test merging i and j
                    communities_ij = copy.deepcopy(current_communities)
                    communities_ij.append(current_communities[i] | current_communities[j])
                    communities_ij.pop(i)
                    communities_ij.pop(j)
                    modularity_ij = nx_comm.modularity(self.knn_graph, communities_ij)
                    if max_modularity < modularity_ij:
                        max_modularity = modularity_ij
                        max_modularity_ij = (i, j)
            best_i, best_j = max_modularity_ij
            current_communities.append(current_communities[best_i] | current_communities[best_j])
            current_communities.pop(best_i)
            current_communities.pop(best_j)
            current_neighborhood_indices.append(current_neighborhood_indices[best_i] | current_neighborhood_indices[best_j])
            current_neighborhood_indices.pop(best_i)
            current_neighborhood_indices.pop(best_j)

            merge_path.append(copy.deepcopy(current_neighborhood_indices))

        return merge_path


def find_patches_and_merge_path(X: torch.Tensor, k: int = -1):
    """
    Finds the neighborhoods in X. Neighborhoods are disjoint for now, but there is a principled merging scheme
    (run a few more iterations of Clauset-Newman-Moore and see which communities it chooses to merge).

    :param X: data matrix, (d, n)
    :param k: number of neighbors to consider
    :return: a list of index sets I_k where each I_k is a community and U I_k = {1, ..., n}
    """
    cd = CommunityDetection(X, k = k)
    communities = cd.find_communities()
    merge_path = cd.find_merge_path(communities)
    return communities, merge_path


X = torch.randn(10, 10000)
find_patches_and_merge_path(X)
