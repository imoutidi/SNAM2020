import numpy as np
import random


def kmedoids(distances, clust_num, max_itter=200):
    # determine dimensions of distance matrix
    m, n = distances.shape

    if clust_num > n:
        raise Exception("Too many medoids")

    valid_inds = set(range(n))
    invalid_inds = set([])
    rows, cols = np.where(distances == 0)
    # the rows, cols must be shuffled because we will keep the first duplicate below
    index_shuf = list(range(len(rows)))
    np.random.shuffle(index_shuf)
    rs = rows[index_shuf]
    cs = cols[index_shuf]
    for r,c in zip(rs,cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_inds:
            invalid_inds.add(c)
    valid_inds = list(valid_inds - invalid_inds)

    if clust_num > len(valid_inds):
        raise Exception("Too many medoids (after removing {} duplicate points)".format(
            len(invalid_inds)))

    # initializa array with random values
    M = np.array(valid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:clust_num])

    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(max_itter):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(distances[:, M], axis=1)
        for kappa in range(clust_num):
            C[kappa] = np.where(J == kappa)[0]
        # update cluster medoids
        for kappa in range(clust_num):
            J = np.mean(distances[np.ix_(C[kappa],C[kappa])], axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(distances[:, M], axis=1)
        for kappa in range(clust_num):
            C[kappa] = np.where(J == kappa)[0]

    # return results
    return M, C