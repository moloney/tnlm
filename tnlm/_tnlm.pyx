cimport cython
from cython.view cimport array as cvarray
from libc.math cimport sqrt, fabs
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef temporal_nl_means(
    double[:, :, :, :] image, 
    double sigma,
    char[:, :, :] mask, 
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
    #Taking residuals of (gaussian) noisy signals boosts noise ~41%
    cdef double res_sigma = sigma * 1.41
    # We tighten threshold on noise stats in proportion to sqrt of # of samples
    cdef double noise_thresh = (5 * res_sigma) / sqrt(dims[3])
    print("Using noise thresh = %f", noise_thresh)
    cdef int i, j, k, ni, nj, nk, t_idx, n_avg
    cdef double[:] res = np.empty(dims[3])
    cdef double[:] center_ts, neigh_ts
    cdef double mean, m2, var, diff, delta, delta2
    cdef double[:, :, :, :] out = np.zeros_like(image)

    with nogil:
        for k in range(0, dims[2]):
            for i in range(0, dims[1]):
                for j in range(0, dims[0]):
                    if mask[j, i, k] == 0:
                        continue
                    center_ts = image[j, i, k]
                    for t_idx in range(dims[3]):
                        res[t_idx] = center_ts[t_idx]
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
                                if fabs(mean) > noise_thresh or (var - res_sigma) > noise_thresh:
                                    continue
                                for t_idx in range(dims[3]):
                                    res[t_idx] += neigh_ts[t_idx]
                                n_avg +=1
                    for t_idx in range(dims[3]):
                        out[j, i, k, t_idx] = res[t_idx] / n_avg
    return out

