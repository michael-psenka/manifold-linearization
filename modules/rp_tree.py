import typing
import torch
import matplotlib.pyplot as plt

n = 1000
d = 2


class RPTree:
    def __init__(self, X: torch.Tensor, min_size: int, c: float):
        self.is_leaf = False
        if X.shape[0] < min_size:
            self.X = X
            self.is_leaf = True
        else:
            self.rule = self.choose_rule_pca(X, c)
            left_data = X[self.rule(X)]
            right_data = X[torch.logical_not(self.rule(X))]
            self.left_subtree = RPTree(left_data, min_size, c)
            self.right_subtree = RPTree(right_data, min_size, c)

    def choose_rule_pca(self, X: torch.Tensor, c: int):
        Sigma = torch.cov(X.T)
        Lambda, Q = torch.linalg.eigh(Sigma)
        u = Q[:, 0]
        proj = X @ u
        median_proj = torch.median(proj)
        return lambda x: x @ u <= median_proj

    def choose_rule_max(self, X: torch.Tensor, c: int):
        v = torch.randn((X.shape[1], ))
        x_idx = torch.randint(low=0, high=X.shape[0], size=())
        x = X[x_idx]
        y_idx = torch.argmax(torch.norm(X - x, dim=1))
        y = X[y_idx]
        delta = (2*torch.rand(size=()) - 1) * 6 * torch.linalg.norm(x - y) / (X.shape[1] ** 0.5)
        proj = X @ v
        median_proj = torch.median(proj)
        return lambda x: x @ v <= median_proj + delta

    def choose_rule_mean(self, X: torch.Tensor, c: float):
        pairwise_distances = torch.cdist(X, X)
        Delta_sq = torch.max(pairwise_distances) ** 2
        Delta_A_sq = torch.sum(pairwise_distances ** 2) / (X.shape[0] ** 2)
        if Delta_sq <= c * Delta_A_sq:
            v = torch.randn((X.shape[1], ))
            proj = X @ v
            median_proj = torch.median(proj)
            return lambda x: x @ v <= median_proj
        else:
            mean = torch.mean(X, dim=0)
            median_distance_from_mean = torch.median(torch.linalg.norm(X - mean, dim=1))
            return lambda x: torch.linalg.norm(x - mean, dim=-1) <= median_distance_from_mean

    def segment(self, X: torch.Tensor):
        if self.is_leaf:
            return X
        else:
            left_segments = self.left_subtree.segment(X[self.rule(X)])
            right_segments = self.right_subtree.segment(X[torch.logical_not(self.rule(X))])
            return left_segments, right_segments




w = torch.randn((n,))
fw = torch.sin(w)
S = torch.stack([w, fw]).T
tree = RPTree(S, min_size=250, c=100.0)
segments = tree.segment(S)


# do bfs for scatterplot
def plot_segments(segment):
    if isinstance(segment, tuple):
        plot_segments(segment[0])
        plot_segments(segment[1])
    else:
        plt.scatter(segment[:, 0], segment[:, 1])


plot_segments(segments)
plt.show()
