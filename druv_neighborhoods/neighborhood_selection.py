import torch

"""
Implements neighborhood selection algorithm presented in "Adaptive Manifold Learning" by Zhenyue Zhang et al.
"""

def neighborhood_selection(X: torch.Tensor, d: int, k_min: int, k_max: int, eta: float) -> list[torch.Tensor]:
    assert k_min <= k_max
    n, m = X.shape

    pairwise_distances = torch.cdist(X, X)
    _, indices = torch.sort(input=pairwise_distances)  # indices[i][1:k+1] is indices of k closest neighbors to x_i
    neighborhoods = []
    for i in range(n):
        ## Algorithm NC: Neighborhood Contraction
        # find k_max nearest neighbors of each point

        X_i_k_max = X[indices[i][:k_max+1]]  # Step 1

        k = k_max
        X_i_k = X_i_k_max[:]

        min_r_i = float('inf')
        min_k = -1

        while True:
            xbar_i_k = torch.mean(X_i_k, dim=0)
            sigma_i_k = torch.linalg.svdvals(X_i_k - xbar_i_k)
            r_i_k = (torch.sum(sigma_i_k[d:] ** 2) / torch.sum(sigma_i_k[:d] ** 2)) ** 0.5  # Step 2

            if r_i_k < min_r_i:
                min_r_i = r_i_k
                min_k = k

            if r_i_k < eta:  # Step 3
                X_i = X_i_k
                break
            if k > k_min:  # Step 4
                X_i_k = X_i_k[:k-1]
                k = k-1
            else:  # Step 5
                X_i = X_i_k_max[:min_k]
                break

        ## Algorithm NE: Neighborhood Expansion
        xbar_i = torch.mean(X_i, dim=0)
        U, S, Vh = torch.linalg.svd(X_i - xbar_i)
        QiT = Vh[:d]  # step 1

        added_points = []
        ki = X_i.shape[0]
        for j in range(ki, k_max):
            theta_i_j = QiT @ (X_i_k_max[j] - xbar_i)  # step 2
            if torch.linalg.norm(X_i_k_max[j] - xbar_i - QiT.T @ theta_i_j) <= eta * torch.linalg.norm(theta_i_j): # Step 3
                added_points.append(X_i_k_max[j])
        if len(added_points) > 0:
            added_points = torch.stack(tensors=added_points, dim=0)
            X_i = torch.cat(tensors=(X_i, added_points))
        neighborhoods.append(X_i)
    return neighborhoods

__all__ = ["neighborhood_selection"]

