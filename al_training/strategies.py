"""Acquisition strategies: which unlabelled points to send for labelling next.

Every strategy is a function  pick(state, n) -> array of pool indices
where state is a dict with

    X         pool features, shape (N, d)
    labeled   boolean mask of the points that already have a label
    probs     model probabilities on the pool, shape (N, 2), column 1 = target class
    rng       numpy random generator

The logic follows the ALFM code that produced the simulation figures in the
EarthQuery report (entropy, TypiClust, max_positive, exploit_explore) and
the thirds rule of the EarthQuery tool (utils/active_learning.py::top_stratified).
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

TARGET = 1   # the class we are looking for (solar farm)


def entropy(probs):
    """Shannon entropy of each row of class probabilities (0 = certain)."""
    p = np.clip(probs, 1e-10, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def _unlabeled(state):
    return np.flatnonzero(~state["labeled"])


# ---------------------------------------------------------------- basics --
def random_pick(state, n):
    """Baseline: n unlabelled points chosen uniformly at random."""
    u = _unlabeled(state)
    return state["rng"].choice(u, size=min(n, len(u)), replace=False)


def entropy_pick(state, n):
    """Uncertainty sampling: the n points whose prediction is least certain."""
    u = _unlabeled(state)
    h = entropy(state["probs"][u])
    return u[np.argsort(-h)[:n]]


def exploit_pick(state, n):
    """Exploit only: the n points most likely to be the target class."""
    u = _unlabeled(state)
    p = state["probs"][u, TARGET]
    return u[np.argsort(-p)[:n]]


def exploit_entropy_pick(state, n):
    """Half the batch by highest target probability, half by highest entropy."""
    u = _unlabeled(state)
    p = state["probs"][u, TARGET]
    h = entropy(state["probs"][u])
    n_exploit = n // 2
    chosen = list(np.argsort(-p)[:n_exploit])
    for i in np.argsort(-h):
        if len(chosen) >= n:
            break
        if i not in chosen:
            chosen.append(i)
    return u[np.array(chosen)]


# ------------------------------------------------------------- TypiClust --
class TypiClust:
    """TypiClust (Hacohen et al. 2022): cluster the pool, then label the most
    typical point of the clusters that have no labels yet.

    The number of clusters grows with the number of labels
    (k = labels + batch, capped at max_clusters). A typical point is one
    whose mean distance to its knn nearest neighbours is small, that is a
    point in a dense part of its cluster. Clusters smaller than min_size are
    ignored. The clustering runs on unit-length features and is cached per k.
    """

    def __init__(self, X, max_clusters=100, knn=20, min_size=5, seed=0):
        self.X = np.asarray(X, dtype=np.float32)
        self.U = self.X / (np.linalg.norm(self.X, axis=1, keepdims=True) + 1e-9)
        self.max_clusters, self.knn, self.min_size, self.seed = max_clusters, knn, min_size, seed
        self._cache = {}

    def clusters(self, k):
        if k not in self._cache:
            km = KMeans(n_clusters=k, n_init=1, random_state=self.seed).fit(self.U)
            self._cache = {k: km.labels_}        # keep only the latest clustering
        return self._cache[k]

    def typical(self, members):
        """Index (into members) of the most typical point."""
        knn = min(self.knn, len(members) // 2)
        D = pairwise_distances(self.U[members])
        D.sort(axis=1)
        return int(D[:, 1:knn + 1].mean(axis=1).argmin())

    def __call__(self, state, n):
        labeled = state["labeled"]
        k = min(int(labeled.sum()) + n, self.max_clusters)
        lab = self.clusters(k).copy()
        ids, sizes = np.unique(lab, return_counts=True)
        have = np.bincount(lab[labeled], minlength=k)[ids]
        order = sorted(range(len(ids)), key=lambda i: (have[i], -sizes[i]))
        order = [i for i in order if sizes[i] > self.min_size]
        lab[labeled] = -1                        # labelled points are used up
        picks = []
        for i in range(n):
            cid = ids[order[i % len(order)]]
            members = np.flatnonzero(lab == cid)
            if len(members) > self.min_size:
                j = members[self.typical(members)]
                picks.append(j)
                lab[j] = -1
        return np.array(picks, dtype=int)


# ------------------------------------------------- the EarthQuery rule ----
class EarthQueryRule:
    """The thirds rule of the EarthQuery tool (utils/active_learning.py::top_stratified).

    The candidates are the `pool` highest-scoring unlabelled points (the tool
    takes the hot blobs of the score map, at most 1,500). A batch of n is
    filled in three parts of n // 3, the remainder going to exploit, then
    diversity:

      exploit    the highest scores: mostly farms, plus the negatives the model
                 scores highest.
      diversity  k-means over the candidates' embeddings with k = quota, then
                 the best-scoring member of every cluster. One per cluster,
                 so the batch is not twenty examples of the same farm type.
      novelty    score times distance to the nearest labelled point.
                 High score, far from every labelled point.

    Every part skips points already taken. Any shortfall is topped up by
    score and tagged "topup". With `coords` and `min_distance` the picks also
    keep a minimum distance from each other, later parts keeping their
    distance from earlier picks, as the tool does on the map.

    metric = "cosine" (unit embeddings; distance = 1 - cosine similarity) or
    "euclidean" (plain distance scaled by its largest value among the
    candidates; used for the two-dimensional toy data).

    After a call, `last` holds everything about the batch for plotting:
    cand (indices, best first), score, cluster, distance, novelty, picked
    (positions in cand) and source (the part that took each pick).
    """

    def __init__(self, X, pool=300, seed=0, metric="cosine", coords=None, min_distance=None):
        X = np.asarray(X, dtype=np.float32)
        if metric not in ("cosine", "euclidean"):
            raise ValueError("metric must be 'cosine' or 'euclidean'")
        self.metric = metric
        self.F = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9) if metric == "cosine" else X
        self.pool, self.seed = pool, seed
        self.coords = None if coords is None else np.asarray(coords, dtype=float)
        self.min_distance = min_distance
        self.last, self.last_source = None, []

    @staticmethod
    def quotas(n):
        q, extra = divmod(int(n), 3)
        return {"exploit": q + (extra > 0), "diversity": q + (extra > 1), "novelty": q}

    def distance_to_labeled(self, cand, labeled):
        """Distance from each candidate to its nearest labelled point, in [0, 1]."""
        ref = self.F[labeled]
        if len(ref) == 0:
            return np.ones(len(cand))
        if self.metric == "cosine":
            return 1.0 - (self.F[cand] @ ref.T).max(axis=1)
        d = pairwise_distances(self.F[cand], ref).min(axis=1)
        return d / (d.max() + 1e-9)

    def __call__(self, state, n):
        u = _unlabeled(state)
        score_u = state["probs"][u, TARGET]
        cand = u[np.argsort(-score_u)[:self.pool]]           # sorted by score, best first
        s = state["probs"][cand, TARGET]
        quota = self.quotas(n)
        picked, source = [], []

        def far_enough(i):
            if self.coords is None or self.min_distance is None or not picked:
                return True
            d = np.linalg.norm(self.coords[cand[picked]] - self.coords[cand[i]], axis=1)
            return bool((d >= self.min_distance).all())

        def take(order, k, tag):
            got = 0
            for i in order:
                if got >= k:
                    break
                if i in picked or not far_enough(i):
                    continue
                picked.append(int(i)); source.append(tag); got += 1

        # 1) exploit: the candidates are already sorted by score
        take(range(len(cand)), quota["exploit"], "exploit")
        # 2) diversity: one pick per cluster, clusters visited by their best member
        cluster = np.full(len(cand), -1)
        if quota["diversity"] > 0 and len(cand) > 1:
            k = min(quota["diversity"], len(cand))
            cluster = KMeans(n_clusters=k, n_init=4, random_state=self.seed).fit(self.F[cand]).labels_
            groups = [np.flatnonzero(cluster == c) for c in range(k)]
            for g in sorted(groups, key=lambda g: g.min()):
                take(sorted(g), 1, "diversity")
        # 3) novelty: confident but far from everything labelled
        dist = self.distance_to_labeled(cand, state["labeled"])
        novelty = s * dist
        if quota["novelty"] > 0:
            take(np.argsort(-novelty), quota["novelty"], "novelty")
        # 4) top up any shortfall by score
        take(range(len(cand)), int(n) - len(picked), "topup")
        idx = np.array(picked, dtype=int)
        self.last = dict(cand=cand, score=s, cluster=cluster, distance=dist, novelty=novelty,
                         picked=idx, source=list(source))
        self.last_source = list(source)
        return cand[idx]


# --------------------------------------------------------------- registry --
NAMES = ["random", "entropy", "typiclust", "exploit", "exploit + entropy", "earthquery"]


def get_strategy(name, X, seed=0, **kw):
    """Return the pick function for a strategy name (a fresh object per run)."""
    if name == "random":
        return random_pick
    if name == "entropy":
        return entropy_pick
    if name == "exploit":
        return exploit_pick
    if name == "exploit + entropy":
        return exploit_entropy_pick
    if name == "typiclust":
        return TypiClust(X, seed=seed, **kw)
    if name == "earthquery":
        return EarthQueryRule(X, seed=seed, **kw)
    raise ValueError(f"unknown strategy {name!r}; choose from {NAMES}")
