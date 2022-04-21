import typing

import torch
import matplotlib.pyplot as plt

n = 10000
d = 2

class KNN:
    def __init__(self, kappa: float):
        self.kappa = kappa

    def extract_neighborhoods(self, X: torch.Tensor):
        k = min(int(X.shape[0] * self.kappa) + 1, X.shape[0])
        pairwise_distances = torch.cdist(X, X)
        sorted_pairwise_distances, sorted_indices = torch.sort(pairwise_distances, dim=1)
        neighborhoods = [set(sorted_indices[i][:k].tolist()) for i in range(X.shape[0])]
        return neighborhoods

class TSLN:
    def __init__(self, alpha: float):
        assert alpha > 0
        self.alpha = alpha

    def find_sequence(self, X: typing.Set[int], neighborhoods: typing.List[typing.Set[int]]):
        max_neighborhood_size = max(len(neighborhood) for neighborhood in neighborhoods)
        maximizing_neighborhoods = [neighborhood for neighborhood in neighborhoods
                                    if len(neighborhood) == max_neighborhood_size]
        C1 = maximizing_neighborhoods[0]
        S = [C1]
        current_C = C1
        neighborhoods.remove(C1)

        union_in_S = set().union(*S)
        while not X.issubset(union_in_S):
            cond_S = []
            for C in neighborhoods:
                if len(C.intersection(current_C)) > 0:
                    if len(C - union_in_S) < (1 - self.alpha) * len(C):
                        cond_S.append(C)
            max_neighborhood_size = max(len(neighborhood) for neighborhood in cond_S)
            maximizing_neighborhoods = [neighborhood for neighborhood in cond_S
                                        if len(neighborhood) == max_neighborhood_size]
            C = maximizing_neighborhoods[0]
            S.append(C)
            union_in_S.update(C)
            current_C = C
            neighborhoods.remove(C)
        return S


w = torch.randn((n, ))
fw = torch.sin(w)
X = torch.stack(tensors=(w, fw)).T
knn = KNN(0.2)
neighborhoods = knn.extract_neighborhoods(X)
print("KNN finished")
tsln = TSLN(0.4)
neighborhoods = tsln.find_sequence(set(range(X.shape[0])), neighborhoods)
print(len(neighborhoods))
print("TSLN finished")

if d == 2:
    for i in range(len(neighborhoods)):
        idx = list(neighborhoods[i])
        else_idx = list(set(range(X.shape[0])) - neighborhoods[i])
        plt.scatter(X[idx, 0], X[idx, 1])
        plt.scatter(X[else_idx, 0], X[else_idx, 1])
        plt.show()


