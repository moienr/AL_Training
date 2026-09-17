"""The active learning loop, run against a pool whose labels we already know.

We pretend the labels are unknown, let the strategy choose points, and reveal
the true label of each chosen point (the "oracle"). Scores are measured on a
separate test set after every round.
"""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .models import make_model
from .strategies import get_strategy, TARGET, NAMES


def fit_predict(model_name, X_lab, y_lab, X_query, seed=0):
    """Train on the labelled points and return P(target) for X_query.

    If only one class has been labelled so far there is nothing to learn;
    the model then predicts that class everywhere.
    """
    classes = np.unique(y_lab)
    if len(classes) < 2:
        return np.full(len(X_query), float(classes[0] == TARGET))
    if callable(model_name):                      # a factory such as toy.make_toy_model
        model = model_name(seed=seed)
    else:
        model = make_model(model_name, seed=seed, n_train=len(y_lab))
    return model.fit(X_lab, y_lab).predict_proba(X_query)[:, 1]


def initial_labels(y, rng, init="balanced", size=2):
    """Choose the starting labels: one (or more) of each class, or random."""
    if init == "balanced":
        per_class = max(size // 2, 1)
        return np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False)
                               for c in (0, 1)])
    return rng.choice(len(y), size, replace=False)


def run_active_learning(X, y, X_test, y_test, strategy="entropy", n_iter=50, batch=6,
                        model="probe", init="balanced", init_size=2, seed=0,
                        record_probs=False, strategy_kw=None):
    """Run one active learning experiment and return its history.

    Returns a dict with one entry per round (round 0 = the initial labels):
      n_labels, n_found, bal_acc, recall_target, recall_other, auc   lists of scores
      (n_found = how many target-class points have been labelled so far)
      picked   list of index arrays, the points labelled in each round
      probs    (optional) P(target) on the whole pool after each round
    `model` is 'probe', 'rf' or a function seed -> estimator.
    """
    X, y = np.asarray(X), np.asarray(y)
    rng = np.random.default_rng(seed)
    labeled = np.zeros(len(X), dtype=bool)
    first = initial_labels(y, rng, init, init_size)
    labeled[first] = True
    pick = get_strategy(strategy, X, seed=seed, **(strategy_kw or {}))
    hist = {k: [] for k in ("n_labels", "n_found", "bal_acc", "recall_target", "recall_other", "auc")}
    hist["picked"], hist["probs"], hist["strategy"] = [first], [], strategy
    hist["source"] = [["init"] * len(first)]     # which part of the strategy chose each point

    for it in range(n_iter + 1):
        p_test = fit_predict(model, X[labeled], y[labeled], X_test, seed=seed)
        pred = (p_test >= 0.5).astype(int)
        hist["n_labels"].append(int(labeled.sum()))
        hist["n_found"].append(int((y[labeled] == TARGET).sum()))      # target-class points labelled so far
        hist["bal_acc"].append(balanced_accuracy_score(y_test, pred))
        hist["recall_target"].append(float((pred[y_test == TARGET] == TARGET).mean()))
        hist["recall_other"].append(float((pred[y_test != TARGET] != TARGET).mean()))
        hist["auc"].append(roc_auc_score(y_test, p_test) if len(np.unique(p_test)) > 1 else 0.5)
        if it == n_iter:
            break
        p_pool = fit_predict(model, X[labeled], y[labeled], X, seed=seed)
        if record_probs:
            hist["probs"].append(p_pool.astype(np.float32))
        state = dict(X=X, labeled=labeled, probs=np.c_[1 - p_pool, p_pool], rng=rng)
        new = np.asarray(pick(state, batch), dtype=int)
        src = list(getattr(pick, "last_source", [])) or [strategy] * len(new)
        keep = ~labeled[new]
        new, src = new[keep], [t for t, k in zip(src, keep) if k]
        labeled[new] = True
        hist["picked"].append(new)
        hist["source"].append(src)
    if record_probs:
        hist["probs"].append(fit_predict(model, X[labeled], y[labeled], X, seed=seed).astype(np.float32))
    return hist


def run_many(X, y, X_test, y_test, strategies, seeds=(0, 1, 2), strategy_kw=None, **kw):
    """Run several strategies over several seeds: {strategy: [history per seed]}.

    strategy_kw is passed to every strategy, or, if its keys are strategy
    names, each strategy gets its own dict (for example
    {"earthquery": {"metric": "euclidean"}}).
    """
    def kw_for(name):
        if strategy_kw and set(strategy_kw) <= set(NAMES):
            return strategy_kw.get(name, {})
        return strategy_kw or {}
    return {s: [run_active_learning(X, y, X_test, y_test, strategy=s, seed=sd, strategy_kw=kw_for(s), **kw)
                for sd in seeds] for s in strategies}


def picks_by_source(hist, y):
    """Per part of the strategy: how many points it picked and how many were the target class."""
    import pandas as pd
    rows = {}
    for idx, src in zip(hist["picked"][1:], hist["source"][1:]):
        for i, t in zip(idx, src):
            r = rows.setdefault(t, [0, 0])
            r[0] += 1
            r[1] += int(y[i] == TARGET)
    df = pd.DataFrame([(t, n, f) for t, (n, f) in rows.items()],
                      columns=["criterion", "picks", "positives"]).set_index("criterion")
    df["hit rate"] = df["positives"] / df["picks"]
    return df


def split_pool(X, y, test_fraction=0.1, seed=0):
    """Stratified split into a pool (for labelling) and a fixed test set."""
    rng = np.random.default_rng(seed)
    test = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        test[rng.choice(idx, int(round(test_fraction * len(idx))), replace=False)] = True
    return ~test, test


def run_budgets(X, y, X_test, y_test, strategies, budgets=(2, 6, 12, 25, 50), total=300, seeds=(0, 1),
                model="probe", strategy_kw=None):
    """run_many once per budget (labels per round), each run to about `total` labels: {budget: results}."""
    return {b: run_many(X, y, X_test, y_test, strategies, seeds=seeds, strategy_kw=strategy_kw,
                        n_iter=max(1, total // b), batch=b, model=model) for b in budgets}
