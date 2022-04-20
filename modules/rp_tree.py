import typing
import torch
import matplotlib.pyplot as plt

n = 1000
d = 2


class RPTree:
    def __init__(self, depth: int, c:float):
        self.depth = depth
        self.c = c
        if self.depth == 0:
            self.S = None
        else:
            self.left_subtree = RPTree(depth - 1, c)
            self.right_subtree = RPTree(depth - 1, c)
            self.rule = None

    def fit(self, S: torch.Tensor):  # (n, d)
        if self.depth == 0:
            self.S = S
        else:
            pairwise_distances = torch.cdist(S, S)  # (n, n)
            pairwise_sq_distances = torch.square(pairwise_distances)  # (n, n)
            Delta_sq = torch.max(pairwise_sq_distances)  # ()
            Delta_A_sq = torch.sum(pairwise_sq_distances) / (S.shape[0] ** 2)
            if Delta_sq <= self.c * Delta_A_sq:
                v = torch.randn(S.shape[1])  # (d, )
                projections = S @ v  # (n, )
                sorted_projections = torch.sort(projections)[0]  # (n, )

                c_min = float('inf')
                i_min = -1

                mu_1 = 0.0
                mu_2 = torch.mean(sorted_projections)
                for i in range(S.shape[0] - 1):
                    mu_1 = ((i * mu_1) + sorted_projections[i]) / (i + 1)
                    mu_2 = (((S.shape[0] - i) * mu_2) - sorted_projections[i]) / (S.shape[0] - i - 1)
                    c_i = torch.sum(torch.square(sorted_projections[:i] - mu_1)) + \
                          torch.sum(torch.square(sorted_projections[i:] - mu_2))
                    if c_i < c_min:
                        c_min = c_i
                        i_min = i
                theta = (sorted_projections[i_min] + sorted_projections[i_min + 1]) / 2
                self.rule = lambda x: x @ v <= theta
            else:
                mean = torch.mean(S, dim=0)  # (d, )
                distances = torch.linalg.norm(S - mean, dim=1)
                median_dist = torch.median(distances)
                self.rule = lambda x: torch.linalg.norm(x - mean, dim=-1) <= median_dist
            self.left_subtree.fit(S[self.rule(S)])
            self.right_subtree.fit(S[torch.logical_not(self.rule(S))])

    def segment(self, S: torch.Tensor):
        if self.depth == 0:
            return S
        else:
            assert self.left_subtree is not None
            left_segments = self.left_subtree.segment(S[self.rule(S)])
            right_segments = self.right_subtree.segment(S[torch.logical_not(self.rule(S))])
            return left_segments, right_segments


S = torch.cat(tensors=(torch.randn(n // 2, d) + 0.1*torch.ones(d), torch.randn(n // 2, d) - 0.1*torch.ones(d)))
tree = RPTree(depth=2, c=2.0)
tree.fit(S)
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
