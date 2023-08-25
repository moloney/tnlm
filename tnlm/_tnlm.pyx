"""Fast Cython implementation of noise estimation and removal methods"""
cimport cython
from cython.view cimport array as cvarray
from libc.math cimport sqrt, fabs
import numpy as np


ctypedef fused in_type:
    cython.ushort
    cython.short
    cython.uint
    cython.int
    cython.float
    cython.double


ctypedef fused accum_type:
    cython.float
    cython.double


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef estimate_neigh_bias_and_noise(
    float[:, :, :, :] image,
    char[:, :, :] mask, 
    int radius=5,
    int n_best=3,
):
    """Estimate amount of bias / noise between each voxel and its neighbors
    
    For each voxel in the mask we compute the residuals between that voxel and its 
    neighbors (that are also in the mask) within `radius` distance. We compute the mean 
    (bias) and stdev (noise) of the residuals, keeping track of the `n_best` lowest 
    results so we can charterize the distribution in results for the most similar 
    voxels. If there are less neighbors than `n_best` a -1 result will be generated and 
    should be ignored. We divide the returned stdev by `sqrt(2)` to account for the 
    increase in noise when subtracting two noisy signals.

    Generally this should just be used on voxels with minimal tissue heterogeneity.
    """
    cdef int[:] dims = cvarray((4,), itemsize=sizeof(int), format="i")
    dims[0] = image.shape[0]
    dims[1] = image.shape[1]
    dims[2] = image.shape[2]
    dims[3] = image.shape[3]
    cdef int i, j, k, ni, nj, nk, t_idx, n_neigh, buf_idx
    cdef float[:] center_ts, neigh_ts
    cdef double mean, m2, var, diff, delta, delta2, max_bias, max_noise
    cdef float[:, :, :, :] best_bias = np.zeros(tuple(dims[:3]) + (n_best,), dtype=np.float32)
    cdef float[:, :, :, :] best_noise = np.zeros(tuple(dims[:3]) + (n_best,), dtype=np.float32)
    cdef float[:] bias_buf = np.zeros(n_best, dtype=np.float32)
    cdef float[:] noise_buf = np.zeros(n_best, dtype=np.float32)
    cdef float inv_sqrt_2 = 1.0 / sqrt(2)
    with nogil:
        for k in range(0, dims[2]):
            for i in range(0, dims[1]):
                for j in range(0, dims[0]):
                    if mask[j, i, k] == 0:
                        continue
                    center_ts = image[j, i, k]
                    n_neigh = 0
                    for nk in range(k - radius, k + radius + 1):
                        for ni in range(i - radius, i + radius + 1):
                            for nj in range(j - radius, j + radius + 1):
                                if ni == i and nj == j and nk == k:
                                    continue
                                if ni < 0 or nj < 0 or nk < 0 or nj >= dims[0] or ni >= dims[1] or nk >= dims[2]:
                                    continue
                                if mask[nj, ni, nk] == 0:
                                    continue
                                neigh_ts = image[nj, ni, nk]
                                mean = m2 = 0.0
                                for t_idx in range(dims[3]):
                                    diff = center_ts[t_idx] - neigh_ts[t_idx]
                                    delta = diff - mean
                                    mean += delta / (t_idx + 1)
                                    delta2 = diff - mean
                                    m2 += delta * delta2
                                var = (m2 / dims[3]) * inv_sqrt_2
                                bias = fabs(mean)
                                if n_neigh < n_best:
                                    bias_buf[n_neigh] = bias
                                    noise_buf[n_neigh] = var
                                else:
                                    for buf_idx in range(n_best):
                                        if bias < bias_buf[buf_idx]:
                                            bias_buf[buf_idx] = bias
                                            break
                                    for buf_idx in range(n_best):
                                        if var < noise_buf[buf_idx]:
                                            noise_buf[buf_idx] = var
                                            break
                                n_neigh += 1
                    # Handle case where we have less neighbors than n_best
                    if n_neigh < n_best:
                        max_bias = max_noise = -1
                        for buf_idx in range(n_neigh, n_best):
                            if buf_idx < n_neigh:
                                if bias_buf[buf_idx] > max_bias:
                                    max_bias = bias_buf[buf_idx]
                                if noise_buf[buf_idx] > max_noise:
                                    max_noise = noise_buf[buf_idx]
                            else:
                                bias_buf[buf_idx] = max_bias
                                noise_buf[buf_idx] = max_noise
                    best_bias[j, i, k, :] = bias_buf
                    best_noise[j, i, k, :] = noise_buf
    return best_bias, best_noise


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef temporal_nl_means(
    in_type[:, :, :, :] image, 
    float[:, :, :] bias_thresh,
    float[:, :, :] noise_thresh,
    char[:, :, :] mask,
    in_type[:, :, :, :] out, 
    accum_type[::1] accum,
    int radius=5,
):
    """Perform non-local means of 4D (3D + t) image based off similarity in time signals

    Look nearby signals that could plausibly be the same signal modulo noise and average
    with them.
    """
    cdef int[:] dims = cvarray((4,), itemsize=sizeof(int), format="i")
    dims[0] = image.shape[0]
    dims[1] = image.shape[1]
    dims[2] = image.shape[2]
    dims[3] = image.shape[3]
    cdef int i, j, k, ni, nj, nk, t_idx, n_avg
    #cdef double[:] res = np.empty(dims[3], dtype=np.float64)
    cdef in_type[:] center_ts, neigh_ts
    cdef double mean, m2, var, diff, delta, delta2
    cdef double inv_sqrt_2 = 1.0 / sqrt(2)
    with nogil:
        for k in range(0, dims[2]):
            for i in range(0, dims[1]):
                for j in range(0, dims[0]):
                    if mask[j, i, k] == 0:
                        out[j, i, k] = image[j, i, k]
                        continue
                    center_ts = image[j, i, k]
                    for t_idx in range(dims[3]):
                        accum[t_idx] = center_ts[t_idx]
                    n_avg = 1
                    for nk in range(k - radius, k + radius + 1):
                        for ni in range(i - radius, i + radius + 1):
                            for nj in range(j - radius, j + radius + 1):
                                if ni == i and nj == j and nk == k:
                                    continue
                                if ni < 0 or nj < 0 or nk < 0 or nj >= dims[0] or ni >= dims[1] or nk >= dims[2]:
                                    continue
                                if mask[nj, ni, nk] == 0:
                                    continue
                                neigh_ts = image[nj, ni, nk]
                                mean = m2 = 0.0
                                for t_idx in range(dims[3]):
                                    diff = center_ts[t_idx] - neigh_ts[t_idx]
                                    delta = diff - mean
                                    mean += delta / (t_idx + 1)
                                    delta2 = diff - mean
                                    m2 += delta * delta2
                                var = m2 / dims[3]
                                if fabs(mean) > bias_thresh[j, i, k] or (var * inv_sqrt_2) > noise_thresh[j, i, k]:
                                    continue
                                for t_idx in range(dims[3]):
                                    accum[t_idx] += neigh_ts[t_idx]
                                n_avg +=1
                    for t_idx in range(dims[3]):
                        out[j, i, k, t_idx] = <in_type> (accum[t_idx] / n_avg)
    return out
