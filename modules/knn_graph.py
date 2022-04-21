import torch
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt

n = 1000
d = 2

class KNNGraph:
    def __init__(self, X: torch.Tensor):
        k = min(int(X.shape[0] ** 0.5) + 1, X.shape[0])
        pairwise_distances = torch.cdist(X, X)
        sorted_pairwise_distances, sorted_indices = torch.sort(pairwise_distances, dim=1)
        sorted_pairwise_distances = sorted_pairwise_distances.tolist()
        sorted_indices = sorted_indices.tolist()

        self.knn_graph = nx.Graph()
        for i in range(n):
            for j in range(1, k+1):
                self.knn_graph.add_edge(i, sorted_indices[i][j], weight=1/(sorted_pairwise_distances[i][j] + 1e-8))
        self.communities = nx_comm.greedy_modularity_communities(self.knn_graph)



# w = torch.randn(n)
# fw = torch.sin(w)
# X = torch.stack((w, fw)).T
X = torch.randn((n, d))
G = KNNGraph(X)

for neighborhood in G.communities:
    plt.scatter(X[list(neighborhood), 0], X[list(neighborhood), 1])
plt.show()