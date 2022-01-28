from typing import List, Tuple
import numpy as np
from numpy.random import randint, shuffle, power, choice, dirichlet, normal, permutation
from sklearn.decomposition import PCA
from scipy.stats.mstats import mquantiles

def quantity_skew(X: np.ndarray,
                  y: np.ndarray,
                  n: int,
                  min_quantity: int=2,
                  alpha: float=4.) -> List[np.ndarray]:
    """
    Distribute the examples across the users according to the following probability density function:
    $P(x; a) = a x^{a-1}$
    where x is the id of a client (x in [0, n-1]), and a = `alpha` > 0 with
    - alpha = 1  => examples are equidistributed across clients;
    - alpha = 2  => the examples are "linearly" distributed across users;
    - alpha >= 3 => the examples are power law distributed;
    - alpha -> \infty => all users but one have `min_quantity` examples, and the remaining user all the rest.
    Each client is guaranteed to have at least `min_quantity` examples.

    Parameters
    ----------
    X: np.ndarray
        The examples.
    y: np.ndarray
        The labels.
    n: int
        The number of clients upon which the examples are distributed.
    min_quantity: int, default 2
        The minimum quantity of examples to assign to each user.
    alpha: float=4.
        Hyper-parameter of the power law density function  $P(x; a) = a x^{a-1}$.

    Returns
    -------
    n-dimensional list of arrays. The examples' ids assignment.
    """
    assert min_quantity*n <= X.shape[0], "# of instances must be > than min_quantity*n"
    assert min_quantity > 0, "min_quantity must be >= 1"
    s = np.array(power(alpha, X.shape[0] - min_quantity*n) * n, dtype=int)
    m = np.array([[i] * min_quantity for i in range(n)]).flatten()
    assignment = np.concatenate([s, m])
    shuffle(assignment)
    return [np.where(assignment == i)[0] for i in range(n)]


def class_wise_quantity_skew(X: np.ndarray,
                             y: np.ndarray,
                             n: int,
                             min_quantity: int=2,
                             alpha: float=4.) -> List[np.ndarray]:
    assert min_quantity*n <= X.shape[0], "# of instances must be > than min_quantity*n"
    assert min_quantity > 0, "min_quantity must be >= 1"
    labels = list(range(len(set(y))))
    lens = [np.where(y == l)[0].shape[0] for l in labels]
    min_lbl = min(lens)
    assert min_lbl >= n, "Under represented class!"

    s = [np.array(power(alpha, lens[c] - n) * n, dtype=int) for c in labels]
    assignment = []
    for c in labels:
        ass = np.concatenate([s[c], list(range(n))])
        shuffle(ass)
        assignment.append(ass)

    res = [[] for _ in range(n)]
    for c in labels:
        idc = np.where(y == c)[0]
        for i in range(n):
            res[i] += list(idc[np.where(assignment[c] == i)[0]])

    return [np.array(r, dtype=int) for r in res]


def quantity_skew_lbl(X: np.ndarray,
                      y: np.ndarray,
                      n: int,
                      class_per_client: int=2) -> List[np.ndarray]:
    """
    Suppose each party only has data samples of `class_per_client` (i.e., k) different labels.
    We first randomly assign k different label IDs to each party. Then, for the samples of each
    label, we randomly and equally divide them into the parties which own the label.
    In this way, the number of labels in each party is fixed, and there is no overlap between
    the samples of different parties.
    See: https://arxiv.org/pdf/2102.02079.pdf

    Parameters
    ----------
    X: np.ndarray
        The examples.
    y: np.ndarray
        The lables.
    n: int
        The number of clients upon which the examples are distributed.
    class_per_client: int, default 2
        The number of different labels in each client.

    Returns
    -------
    n-dimensional list of arrays. The examples' ids assignment.
    """
    labels = set(y)
    assert 0 < class_per_client <= len(labels), "class_per_client must be > 0 and <= #classes"
    assert class_per_client * n >= len(labels), "class_per_client * n must be >= #classes"
    nlbl = [choice(len(labels), class_per_client, replace=False)  for u in range(n)]
    check = set().union(*[set(a) for a in nlbl])
    while len(check) < len(labels):
        missing = labels - check
        for m in missing:
            nlbl[randint(0, n)][randint(0, class_per_client)] = m
        check = set().union(*[set(a) for a in nlbl])
    class_map = {c:[u for u, lbl in enumerate(nlbl) if c in lbl] for c in labels}
    assignment = np.zeros(y.shape[0])
    for lbl, users in class_map.items():
        ids = np.where(y == lbl)[0]
        assignment[ids] = choice(users, len(ids))
    return [np.where(assignment == i)[0] for i in range(n)]


def noise_feat_dist_skew(X: np.ndarray,
                         y: np.ndarray,
                         n: int,
                         sigma: float=0.001) -> Tuple[np.ndarray, np.ndarray]:
    """
    The function divide the whole dataset into multiple parties randomly and equally.
    For each party, we add different levels of Gaussian noise to its local dataset to
    achieve different feature distributions. Specifically, given user-defined noise level sigma,
    we add noises ~ Gau(sigma * i/N) for Party i, where Gau(sigma * i/N) is a Gaussian distribution
    with mean 0 and variance sigma * i/N.
    See: https://arxiv.org/pdf/2102.02079.pdf

    Parameters
    ----------
    X: np.ndarray
        The examples.
    y: np.ndarray
        The lables.
    n: int
        The number of clients upon which the examples are distributed.
    sigma: float, default 0.001
        The noise level to apply to the features.

    Returns
    -------
    Tuple with two lists
      1. list of arrays representing the examples of each user;
      2. list of arrays representing the labels of each user.
    """
    assignment = choice(n, X.shape[0], replace=True)
    X_i, y_i = [], []
    for i in range(n):
        ids = np.where(assignment == i)[0]
        X_i.append(X[ids] + normal(0, sigma * (i+1)/n, (len(ids), X.shape[1])))
        y_i.append(y[ids])
    return X_i, y_i


def dist_skew_lbl(X: np.ndarray,
                  y: np.ndarray,
                  n: int,
                  beta: float=.5) -> List[np.ndarray]:
    """
    The function samples p_k ~ Dir_n (beta) and allocate a p_{k,j} proportion of the instances of
    class k to party j. Here Dir(_) denotes the Dirichlet distribution and beta is a
    concentration parameter (beta > 0).
    See: https://arxiv.org/pdf/2102.02079.pdf

    Parameters
    ----------
    X: np.ndarray
        The examples.
    y: np.ndarray
        The lables.
    n: int
        The number of clients upon which the examples are distributed.
    beta: float, default .5
        The beta parameter of the Dirichlet distribution, i.e., Dir(beta).

    Returns
    -------
    n-dimensional list of arrays. The examples' ids assignment.
    """
    assert beta > 0, "beta must be > 0"
    labels = set(y)
    pk = {c: dirichlet([beta]*n, size=1)[0] for c in labels}
    assignment = np.zeros(y.shape[0])
    for c in labels:
        ids = np.where(y == c)[0]
        shuffle(ids)
        shuffle(pk[c])
        assignment[ids[n:]] = choice(n, size=len(ids)-n, p=pk[c])
        assignment[ids[:n]] = list(range(n))

    return [np.where(assignment == i)[0] for i in range(n)]


def pathological_dist_skew_lbl(X: np.ndarray,
                               y: np.ndarray,
                               n: int,
                               shards_per_client: int=2):
    """
    The function first sort the data by label, divide it into `n * shards_per_client` shards, and
    assign each of n clients `shards_per_client` shards. This is a pathological non-IID partition
    of the data, as most clients will only have examples of a limited number of classes.
    See: http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf

    Parameters
    ----------
    X: np.ndarray
        The examples.
    y: np.ndarray
        The lables.
    n: int
        The number of clients upon which the examples are distributed.
    shards_per_client: int, default 2
        Number of shards per client.

    Returns
    -------
    n-dimensional list of arrays. The examples' ids assignment.
    """
    sorted_ids = np.argsort(y)
    n_shards = int(shards_per_client * n)
    shard_size = int(np.ceil(len(y) / n_shards))
    assignments = np.zeros(y.shape[0])
    perm = permutation(n_shards)
    j = 0
    for i in range(n):
        for _ in range(shards_per_client):
            left = perm[j] * shard_size
            right = min((perm[j]+1) * shard_size, len(y))
            assignments[sorted_ids[left:right]] = i
            j += 1
    return [np.where(assignments == i)[0] for i in range(n)]


def covariate_shift(X: np.ndarray,
                    y: np.ndarray,
                    n: int,
                    modes: int=2):
    """
    The function first extracts the first principal component (through PCA) and then divides it in
    `modes` percentiles. To each user, only examples from a single mode are selected (uniformly).
    
    Parameters
    ----------
    X: np.ndarray
        The examples.
    y: np.ndarray
        The lables.
    n: int
        The number of clients upon which the examples are distributed.
    modes: int, default 2
        Number of different modes to consider in the input data first principal component.
    
    Returns
    -------
    n-dimensional list of arrays. The examples' ids assignment.
    """
    assert 2 <= modes <= n, "modes must be >= 2 and <= n"

    ids_mode = [[] for _ in range(modes)]
    for lbl in set(y):
        ids = np.where(y == lbl)[0]
        X_pca = PCA(n_components=2).fit_transform(X[ids])
        quantiles = mquantiles(X_pca[:, 0], prob=np.linspace(0, 1, num=modes+1)[1:-1])

        y_ = np.zeros(y[ids].shape)
        for i, q in enumerate(quantiles):
            if i == 0: continue
            id_pos = np.where((quantiles[i-1] < X_pca[:, 0]) & (X_pca[:, 0] <= quantiles[i]))[0]
            y_[id_pos] = i
        y_[np.where(X_pca[:, 0] > quantiles[-1])[0]] = modes-1

        for m in range(modes):
            ids_mode[m].extend(ids[np.where(y_ == m)[0]])

    ass_mode = (list(range(modes)) * int(np.ceil(n/modes)))[:n]
    shuffle(ass_mode)
    mode_map = {m:[u for u, mu in enumerate(ass_mode) if mu == m] for m in range(modes)}
    assignment = np.zeros(y.shape[0])
    for mode, users in mode_map.items():
        ids = ids_mode[mode]
        assignment[ids] = choice(users, len(ids))
    return [np.where(assignment == i)[0] for i in range(n)]
