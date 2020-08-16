from numpy import ndarray
import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_accuracy(
    samples_i: ndarray, samples_j: ndarray, n_neighbours: int = 5
) -> float:
    nn_estimator = NearestNeighbors(n_neighbors=n_neighbours)
    nn_estimator = nn_estimator.fit(samples_i)
    among_knn = 0
    for j in range(len(samples_j)):
        sample_j = samples_j[j]
        knn_idc = nn_estimator.kneighbors(
            sample_j.reshape(1, -1), return_distance=False
        )
        if j in knn_idc:
            among_knn += 1
    knn_acc = among_knn / len(samples_j)
    return knn_acc
