import networkx as nx
import networkx.algorithms.community as nx_comm
import torch


class CommunityDetection:
    def __init__(self, X: torch.Tensor, eps: float = 1e-8):
        """
        Builds KNN graph, where k = O(sqrt(n)). Vertices are n data points xi, edges are 1/d(xi, xj)
        if xj is one of the k nearest neighbors of xi.

        :param X: data matrix, (n, d)
        :param eps: perturbation to avoid divide-by-zero error, default=1e-8 should be good
        """
        n, d = X.shape
        k = min(int(n ** 0.5) + 1, X.shape[0])
        pairwise_distances = torch.cdist(X, X)
        sorted_pairwise_distances, sorted_indices = torch.sort(pairwise_distances, dim=1)
        sorted_pairwise_distances = sorted_pairwise_distances.tolist()
        sorted_indices = sorted_indices.tolist()

        self.knn_graph = nx.Graph()
        for i in range(n):
            for j in range(1, k):
                self.knn_graph.add_edge(i, sorted_indices[i][j], weight=1/(sorted_pairwise_distances[i][j] + eps))

    def find_communities(self):
        """
        Finds the communities in the KNN graph via Clauset-Newman-Moore greedy modularity maximization.
        How this works: lots of points which are very close to each other are considered a "community", and
        regions which are far away are considered different communities. Finds best number of communities
        and partitions the vertices into these communities.

        NOTE: this is strict partition, no overlap.

        :return: a list of index sets where each index set is a community
        """
        return nx_comm.greedy_modularity_communities(self.knn_graph)


def find_patches(X: torch.Tensor):
    """
    Finds the neighborhoods in X. Neighborhoods are disjoint for now, but there is a principled merging scheme
    (run a few more iterations of Clauset-Newman-Moore and see which communities it chooses to merge).

    :param X: data matrix, (n, d)
    :return: a list of index sets I_k where each I_k is a community and U I_k = {1, ..., n}
    """
    return CommunityDetection(X).find_communities()
