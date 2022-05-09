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
        knn_graph = kneighbors_graph(
            X.T, k, mode='distance', include_self=False)
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
                    communities_ij.append(
                        current_communities[i] | current_communities[j])
                    communities_ij.pop(i)
                    communities_ij.pop(j)
                    modularity_ij = nx_comm.modularity(
                        self.knn_graph, communities_ij)
                    if max_modularity < modularity_ij:
                        max_modularity = modularity_ij
                        max_modularity_ij = (i, j)
            best_i, best_j = max_modularity_ij
            current_communities.append(
                current_communities[best_i] | current_communities[best_j])
            current_communities.pop(best_i)
            current_communities.pop(best_j)
            current_neighborhood_indices.append(
                current_neighborhood_indices[best_i] | current_neighborhood_indices[best_j])
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
    cd = CommunityDetection(X, k=k)
    communities = cd.find_communities()
    merge_path = cd.find_merge_path(communities)
    return communities, merge_path


# Main method that cc.py calls
# given data and some hyperparameters, returns neighborhood structure
# as needed for the construction of the cc_network

# Input: X, k same as above (find_patches_and_merge_path)
# eps: estimated magnitude of noise we want neighborhoods to account for
# ------ note this is NOT necessarily meant to just be machine precision
# eps_N: amount to widen neighborhoods by for neighboring detection

    # return ind_X, merge_abbrv, A_N, mu_N, G_N
# Returns:
    # ind_X: list of index sets, where each index set is a neighborhoods
    # merges: list of merges from neighborhoods to full set (note return denoted
    # --- merge_abbrv since we shorten it from the community detection output)
    # --- does not include the first set (all singletons, nothing merged), and does not
    #     include the last set (one set, everything merged)
    # A_N: list of linear operators representing shape and size of each neighborhood
    # mu_N: list of vectors representing centers of each neighborhood
    # G_N: list of adjacency matrices for neighborhoods and merged neighborhoods

def find_neighborhood_structure(X: torch.Tensor, k: int = -1, _eps: float = 1e-8, _eps_N: float = 0):
    # extract ambient dimension
    D = X.shape[0]
    
    # determine if using CPU or GPU
    on_GPU = X.is_cuda

    if on_GPU:
        # need X on cpu for community detections
        X = X.cpu()
        # set default to cuda
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    # find neighborhoods
    ind_X, merge_path = find_patches_and_merge_path(X, k)
    # extract number of neighborhoods
    p = len(merge_path)
    ############# 1. We first abbreviate the redundant merges as to make the resulting
    # network as shallow as possible

    # for disjoint merges, we can do them simultaneously.

    # shortened list of merges, starting with unmerged neighborhoods
    merge_abbrv = []
    # tracks which neighborhoods we've merged
    curr_merges = []

    for i in range(1, p):
        # 1. find new merge
        for ind_set in merge_path[i]:
            if ind_set not in merge_path[i-1]:
                merge_new = ind_set
        # 2. if we just merged something we merged before, keep previous merge
        for ind in merge_new:
            if ind in curr_merges:
                merge_abbrv.append(merge_path[i-1])
                curr_merges = []
                break

        # 3. update what we've merged
        for ind in merge_new:
            curr_merges.append(ind)


    ############# 2. Find neighborhood structure of each neighborhood
    # convert index sets to pytorch tensors
    if on_GPU:
        ind_X = [torch.tensor(list(ind_set)).cuda() for ind_set in ind_X]
        # if started on GPU, can put back on now
        X = X.cuda()
    else:
        ind_X = [torch.tensor(list(ind_set)) for ind_set in ind_X]
    # create neighborhood sets
    X_N = []
    for i in range(p):
        X_N.append(X[:,torch.tensor(list(ind_X[i]))])

    # ####### Find neighboring structure of neighborhoods

    A_N = []
    mu_N = []

    # create neighborhood sets
    for i in range(p):
        X_i_mu = X_N[i].mean(dim=1, keepdim=True)
        # RHS indexing just to get pytorch to work
        mu_N.append(X_i_mu)

        X_c = X_N[i] - X_i_mu
        U, S, V = torch.svd(X_c)
        # account for potential noise in future
        S += _eps
        Sinv = torch.diag(1/S)
        A_N.append(Sinv@U.T)

    ############# 3. Find adjacency structure for neighborhoods
    # neighboring graph

    # note we want a list of neighborhood graphs, one for each level of merged neighborhoods
    # note it is undirected, but stored in upper triangular form

    G_N = []
    G_N0 = torch.zeros((p,p))

    for i in range(p):
        for j in range(i+1,p):
            # calc dist of all points in j neighborhood to i
            n_mult = 1/(1+_eps_N)
            # note we want A_N[:,:,i] to be of shape (D,D) and mu_N[:,i] to be of shape (D,1)
            norms_i_j = (n_mult*A_N[i]@(X_N[j] - mu_N[i])).norm(dim=0)
            num_intersect = (norms_i_j <= 1).sum()
            # if any intersect, neighborhoods are connected
            if num_intersect > 0:
                G_N0[i,j] = 1

            # do same flipped
            norms_j_i = (n_mult*A_N[j]@(X_N[i] - mu_N[j])).norm(dim=0)
            num_intersect = (norms_j_i <= 1).sum()
            # if any intersect, neighborhoods are connected
            if num_intersect > 0:
                G_N0[i,j] = 1

    # construct adjacency matrices for downstream merges
    # we want to store a boolean tensor for indexing, original adjacency matrix
    # is efficiently recoverable if needed
    G_N.append(G_N0 > 0)
    for merge_num in range(0,len(merge_abbrv)):
        p_curr = len(merge_abbrv[merge_num])
        G_curr = torch.zeros((p_curr, p_curr))
        for i in range(p_curr):
            for j in range(i+1,p_curr):
                # see if any points in cluster i match with any points in cluster j
                for ind_i in merge_abbrv[merge_num][i]:
                    for ind_j in merge_abbrv[merge_num][j]:
                        if G_N0[ind_i,ind_j] == 1 or G_N0[ind_j,ind_i] == 1:
                            G_curr[i,j] = 1

        G_N.append(G_curr > 0)


    # convert merge_abbrv to pytorch format
    for merge in merge_abbrv:
        # each merge[i] is an index set within the selected merge
        for i in range(len(merge)):
            merge[i] = torch.tensor(list(merge[i]))

    # return all computed neighborhood info
    return ind_X, merge_abbrv, A_N, mu_N, G_N
