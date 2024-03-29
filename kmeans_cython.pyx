# cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def euclidean_dist_square(np.ndarray[DTYPE_t, ndim=2] data_point, np.ndarray[DTYPE_t, ndim=2] centroid):
    cdef int dim = data_point.shape[0]
    cdef double dist = 0
    for i in range(dim):
        dist += (data_point[i] - centroid[i]) ** 2
    return dist

def assign_points_to_clusters_cython(np.ndarray[DTYPE_t, ndim=2] data, np.ndarray[DTYPE_t, ndim=2] centroids):
    cdef int n_points = data.shape[0]
    cdef int n_clusters = centroids.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] assignments = np.zeros(n_points, dtype=DTYPE)

    cdef int i, j
    cdef double min_dist, dist
    for i in range(n_points):
        min_dist = euclidean_dist_square(data[i, :], centroids[0, :])
        assignments[i] = 0
        for j in range(1, n_clusters):
            dist = euclidean_dist_square(data[i, :], centroids[j, :])
            if dist < min_dist:
                min_dist = dist
                assignments[i] = j
    return assignments

def update_centroids_cython(np.ndarray[DTYPE_t, ndim=2] data, np.ndarray[DTYPE_t, ndim=1] assignments, int n_clusters):
    cdef np.ndarray[DTYPE_t, ndim=2] new_centroids = np.zeros((n_clusters, data.shape[1]), dtype=DTYPE)
    cdef np.ndarray[int, ndim=1] counts = np.zeros(n_clusters, dtype=np.int32)

    cdef int n_points = data.shape[0]
    cdef int i, cluster

    for i in range(n_points):
        cluster = int(assignments[i])
        new_centroids[cluster, :] += data[i, :]
        counts[cluster] += 1

    for i in range(n_clusters):
        if counts[i] > 0:
            new_centroids[i, :] /= counts[i]
    return new_centroids
