"""A small two-dimensional dataset and the plots for notebook 01.

Two dimensions so that every point, every label and the model's boundary can
be drawn. The 'clusters' dataset has two classes spread over five clusters;
the 'rare' dataset has a target class that is only 3% of the points, which is
the situation of solar farms in a landscape.
"""
import hashlib
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

BLUE, ORANGE, GREY, TRUTH = "#3b6ea5", "#e08a1e", "#c8c8c8", "#2f9e44"

# the clusters of each dataset: (centre x, centre y, spread, size, class)
SPEC = {"clusters": [(-3.0, 2.0, 0.7, 160, 0), (0.5, 2.6, 0.6, 120, 1), (-1.0, -1.6, 0.8, 200, 0),
                     (3.0, 0.2, 0.7, 140, 1), (2.6, -2.8, 0.6, 80, 0)],
        "rare": [(-2.5, 1.5, 1.0, 330, 0), (1.5, 2.0, 0.9, 300, 0), (0.0, -2.0, 1.1, 350, 0),
                 (3.2, -1.2, 0.25, 18, 1), (-1.2, 3.6, 0.25, 12, 1)]}


def make_toy(kind="clusters", seed=0):
    """Return X (n, 2) and y (n,) with y = 1 for the target class."""
    rng = np.random.default_rng(seed)
    if kind not in SPEC:
        raise ValueError(kind)
    X, y = [], []
    for cx, cy, s, n, c in SPEC[kind]:
        X.append(rng.normal([cx, cy], s, size=(n, 2)))
        y.append(np.full(n, c))
    X, y = np.vstack(X), np.concatenate(y)
    order = rng.permutation(len(y))
    return X[order], y[order]


_FEATURES = OrderedDict()      # random Fourier features of the last large arrays transformed


class CachedRBFSampler(RBFSampler):
    """RBFSampler that keeps the features of the last large arrays it transformed.

    The pool, the test set and the plotting grid are the same in every round
    and every frame, and the weights depend only on the seed, so their
    features need computing once.
    """

    def transform(self, X):
        X = np.asarray(X)
        if len(X) < 1000:
            return super().transform(X)
        key = (hashlib.md5(np.ascontiguousarray(X).tobytes()).hexdigest(), X.shape,
               self.gamma, self.n_components, self.random_state)
        if key in _FEATURES:
            _FEATURES.move_to_end(key)
        else:
            if len(_FEATURES) >= 6:
                _FEATURES.popitem(last=False)
            F = super().transform(X)
            F.flags.writeable = False
            _FEATURES[key] = F
        return _FEATURES[key]


def make_toy_model(seed=0, gamma=0.2):
    """A classifier that can draw a curved boundary in two dimensions.

    Random Fourier features turn the plane into a richer space in which a
    logistic regression is enough; the boundary it draws in the plane is smooth.
    """
    return make_pipeline(CachedRBFSampler(gamma=gamma, n_components=200, random_state=seed),
                         LogisticRegression(C=10.0, max_iter=1000))


def true_probability(G, kind):
    """P(class 1 | x) of the clusters that generated the dataset: each cluster is a
    round Gaussian with a known centre, spread and size, so the true probability
    at any point follows from Bayes' rule."""
    num, den = np.zeros(len(G)), np.zeros(len(G))
    for cx, cy, s, n, c in SPEC[kind]:
        d = n / (s * s) * np.exp(-((G - [cx, cy]) ** 2).sum(axis=1) / (2 * s * s))
        den += d
        if c == 1:
            num += d
    return num / den


def truth_handle():
    return Line2D([], [], color=TRUTH, lw=1.3, ls="--", label="true boundary (P = 0.5)")


def plot_truth(ax, X, kind):
    """The true boundary, where the true P(class 1) of the generating clusters is 0.5, as a dashed green line."""
    gx, gy, G = grid(X)
    ax.contour(gx, gy, true_probability(G, kind).reshape(gx.shape), levels=[0.5], colors=[TRUTH],
               linestyles=["--"], linewidths=[1.3], zorder=3)
    return truth_handle()


def grid(X, step=0.08):
    x0, x1 = X[:, 0].min() - 0.6, X[:, 0].max() + 0.6
    y0, y1 = X[:, 1].min() - 0.6, X[:, 1].max() + 0.6
    gx, gy = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    return gx, gy, np.c_[gx.ravel(), gy.ravel()]


def dot(color, label, size=6, edge=None, hollow=False):
    """A legend handle: a filled dot, or a ring when hollow."""
    return Line2D([], [], marker="o", ls="", markersize=size, label=label,
                  markerfacecolor="none" if hollow else color,
                  markeredgecolor=edge or (color if hollow else "none"),
                  markeredgewidth=1.3 if hollow else 0.6)


def add_legend(ax, handles, loc="below", ncol=None):
    """The legend under the axes (the default), where it covers no point, or at a matplotlib location."""
    if not handles:
        return
    if loc == "below":
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=ncol or 2,
                  fontsize=7.5, frameon=False, handletextpad=0.5, columnspacing=1.4)
    else:
        ax.legend(handles=handles, loc=loc, fontsize=7.5, frameon=True, framealpha=0.85, edgecolor="none")


def figure_legend(fig, handles, ncol=None):
    """One legend under the whole figure, for panels that share the same entries; replaces an earlier one."""
    for old in list(fig.legends):
        old.remove()
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.06), ncol=ncol or min(len(handles), 4),
               fontsize=8, frameon=False, handletextpad=0.5, columnspacing=1.6)


def plot_points(ax, X, y=None, labeled=None, picks=None, title=None, legend=True, picks_label="picked this round",
                truth=None):
    """Grey pool; labelled points in class colours; new picks ringed in black. With a legend under
    the axes, or one under the whole figure when legend="figure" (panels with the same entries).
    truth = 'clusters' or 'rare' adds the true boundary of that dataset."""
    handles = []
    s_bg = 9 if len(X) <= 2000 else 2.5             # smaller dots for the 10,000-point pool
    ax.scatter(X[:, 0], X[:, 1], s=s_bg, color=GREY, lw=0, zorder=1)
    if y is not None and labeled is None:            # show every label (the truth)
        for c, col, sz in ((0, BLUE, s_bg), (1, ORANGE, 9)):
            ax.scatter(X[y == c, 0], X[y == c, 1], s=sz, color=col, lw=0, zorder=2)
        handles += [dot(BLUE, "class 0", 4), dot(ORANGE, "class 1, the class we look for", 4)]
    else:
        handles.append(dot(GREY, "unlabelled", 4))
    if labeled is not None:
        for c, col in ((0, BLUE), (1, ORANGE)):
            m = labeled & (y == c)
            ax.scatter(X[m, 0], X[m, 1], s=34, color=col, edgecolor="white", lw=0.6, zorder=4)
        handles += [dot(BLUE, "labelled: class 0", edge="white"), dot(ORANGE, "labelled: class 1", edge="white")]
    if picks is not None and len(picks):
        ax.scatter(X[picks, 0], X[picks, 1], s=110, facecolor="none", edgecolor="black",
                   lw=1.3, zorder=5)
        handles.append(dot("black", picks_label, 9, hollow=True))
    if getattr(ax, "_toy_boundary", False):
        handles.append(Line2D([], [], color="black", lw=1.2, label="model boundary (P = 0.5)"))
    if truth:
        handles.append(plot_truth(ax, X, truth))
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#bbbbbb")
    if title:
        ax.set_title(title, fontsize=10)
    if legend == "figure":
        figure_legend(ax.figure, handles)
    elif legend:
        add_legend(ax, handles)
    return handles


def plot_probability(ax, model, X, kind="prob", alpha=0.55, colorbar=True):
    """Shade the plane by P(target) ('prob') or by entropy ('entropy'), with a colour bar."""
    gx, gy, G = grid(X)
    p = model.predict_proba(G)
    if kind == "prob":
        z, cmap, vmin, vmax, label = p[:, 1], "RdYlBu_r", 0, 1, "P(class 1)"
    else:
        q = np.clip(p, 1e-10, 1)
        z, cmap, vmin, vmax, label = -(q * np.log(q)).sum(axis=1), "magma", 0, np.log(2), "entropy (0 = certain, 0.69 = unsure)"
    cf = ax.contourf(gx, gy, z.reshape(gx.shape), levels=np.linspace(vmin, vmax, 21), cmap=cmap, vmin=vmin, vmax=vmax,
                     alpha=alpha, extend="both")
    if kind == "prob":
        ax.contour(gx, gy, p[:, 1].reshape(gx.shape), levels=[0.5], colors="black", linewidths=1.2)
        ax._toy_boundary = True
    if colorbar:
        cb = ax.figure.colorbar(cf, ax=ax, shrink=0.8, pad=0.02, ticks=np.linspace(vmin, vmax, 5))
        cb.set_label(label, fontsize=8); cb.ax.tick_params(labelsize=7)
        cb.solids.set_alpha(1)
    return cf


def labeled_after(hist, i, n):
    """Boolean mask of the points labelled up to and including round i."""
    m = np.zeros(n, dtype=bool)
    m[np.concatenate(hist["picked"][:i + 1])] = True
    return m


def show_round(hist, X, y, i, model_factory=None, figsize=(5.2, 4.6), truth=None):
    """Refit the toy model on the labels of round i and draw the boundary."""
    from IPython.display import display
    m = labeled_after(hist, i, len(X))
    model = (model_factory or make_toy_model)().fit(X[m], y[m])
    fig, ax = plt.subplots(figsize=figsize)
    plot_probability(ax, model, X)
    plot_points(ax, X, y, labeled=m, picks=hist["picked"][i],
                title=f"round {i}: {int(m.sum())} labels, balanced accuracy {hist['bal_acc'][i]:.2f}", truth=truth)
    plt.close(fig)
    display(fig)


def animate_rounds(hist, X, y, model_factory=None, figsize=(5.5, 5.5), interval=700, found=False, truth=None):
    """One run as an animation, one frame per round: the model refit on the labels
    of that round, its boundary, the labels so far and the picks of the round.
    With found=True the title also counts the class 1 points labelled so far.
    When the strategy reports which criterion chose each pick (the EarthQuery
    rule), the rings take the colour of the criterion.
    Show it with HTML(anim.to_jshtml()), a player with a slider that needs no widgets."""
    from matplotlib import animation
    from .plotting import STRATA, STRATA_LABELS
    fig, ax = plt.subplots(figsize=figsize)
    state = {"colorbar": False}
    n_all = int((y == 1).sum())
    tags = [t for t in ("exploit", "diversity", "novelty")
            if any(t in r for r in hist.get("source", []))]
    fig.subplots_adjust(bottom=0.22)              # room for the legend under the plot

    def draw(i):
        ax.clear()
        m = labeled_after(hist, i, len(X))
        model = (model_factory or make_toy_model)().fit(X[m], y[m])
        plot_probability(ax, model, X, colorbar=not state["colorbar"])   # the colorbar is drawn once
        state["colorbar"] = True
        title = f"round {i}: {int(m.sum())} labels, balanced accuracy {hist['bal_acc'][i]:.2f}"
        if found:
            title = (f"round {i}: {int(m.sum())} labels, {hist['n_found'][i]} of {n_all} class 1 points found\n"
                     f"balanced accuracy {hist['bal_acc'][i]:.2f}")
        picks = np.asarray(hist["picked"][i], dtype=int)
        if not tags:
            plot_points(ax, X, y, labeled=m, picks=picks, title=title, truth=truth)
            return (ax,)
        handles = plot_points(ax, X, y, labeled=m, title=title, legend=False, truth=truth)
        src = hist["source"][i]
        for tag in ["init"] + tags:
            idx = [j for j, t in zip(picks, src) if t == tag]
            if idx:
                ax.scatter(X[idx, 0], X[idx, 1], s=110, facecolor="none", edgecolor=STRATA[tag], lw=1.6, zorder=5)
        handles += [dot(STRATA[t], "picked by " + STRATA_LABELS[t].split(":")[0], 9, hollow=True) for t in tags]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=7.5, frameon=False)
        return (ax,)

    anim = animation.FuncAnimation(fig, draw, frames=len(hist["picked"]), interval=interval, blit=False)
    plt.close(fig)
    return anim


def plot_criteria(X, y, labeled, model, n=9, pool=60, seed=0, figsize=(19, 5), truth=None):
    """The EarthQuery rule on the toy data, one panel per criterion.

    From the `pool` highest-scoring unlabelled points (the candidates), the
    rule takes n // 3 by exploit, n // 3 by diversity and n // 3 by novelty.
    Panel 1: the candidates shaded by score, the exploit picks ringed.
    Panel 2: the candidates coloured by k-means cluster, the best of each ringed.
    Panel 3: the candidates shaded by novelty (score times distance to the
    nearest label), the novelty picks ringed and joined to their nearest label.
    Panel 4: the batch, coloured by the criterion that chose each point.
    """
    from .strategies import EarthQueryRule
    from .plotting import STRATA, STRATA_LABELS
    p = model.predict_proba(X)[:, 1]
    rule = EarthQueryRule(X, pool=pool, metric="euclidean", seed=seed)
    picks = rule(dict(X=X, labeled=labeled, probs=np.c_[1 - p, p], rng=np.random.default_rng(seed)), n)
    L = rule.last
    cand, src, pk = L["cand"], np.array(L["source"]), L["picked"]
    by = {tag: cand[pk[src == tag]] for tag in ("exploit", "diversity", "novelty", "topup")}
    q = rule.quotas(n)

    fig, axes = plt.subplots(1, 4, figsize=figsize, constrained_layout=True)
    for ax in axes:
        plot_probability(ax, model, X, alpha=0.3, colorbar=False)
        ax.scatter(X[:, 0], X[:, 1], s=7 if len(X) <= 2000 else 2, color=GREY, lw=0, zorder=1)
    labels_h = [dot(BLUE, "labelled: class 0", edge="white"), dot(ORANGE, "labelled: class 1", edge="white"),
                Line2D([], [], color="black", lw=1.2, label="model boundary (P = 0.5); red = high P, blue = low")]
    if truth:
        labels_h.append(truth_handle())
        for ax in axes:
            plot_truth(ax, X, truth)
    # 1: exploit
    ax = axes[0]
    ax.scatter(X[cand, 0], X[cand, 1], s=26, c=L["score"], cmap="Greys", vmin=0, vmax=1, edgecolor="#555555", lw=0.4, zorder=3)
    plot_points(ax, X, y, labeled=labeled, picks=by["exploit"], legend=False,
                title=f"exploit\nthe {q['exploit']} highest scores of the {len(cand)} candidates")
    add_legend(ax, [dot("#9a9a9a", "candidate, darker = higher score", edge="#555555"),
                    dot("black", "exploit pick", 9, hollow=True)] + labels_h)
    # 2: diversity
    ax = axes[1]
    ax.scatter(X[cand, 0], X[cand, 1], s=26, c=L["cluster"], cmap="tab10", vmin=0, vmax=9, edgecolor="#555555", lw=0.4, zorder=3)
    plot_points(ax, X, y, labeled=labeled, picks=by["diversity"], legend=False,
                title=f"diversity\nk-means into {q['diversity']} clusters, the best of each")
    add_legend(ax, [dot("#1f77b4", "candidate, one colour per cluster", edge="#555555"),
                    dot("black", "diversity pick: the best of its cluster", 9, hollow=True)] + labels_h)
    # 3: novelty
    ax = axes[2]
    ax.scatter(X[cand, 0], X[cand, 1], s=26, c=L["novelty"], cmap="Reds", vmin=0, edgecolor="#555555", lw=0.4, zorder=3)
    lab_idx = np.flatnonzero(labeled)
    for i in by["novelty"]:
        j = lab_idx[np.argmin(((X[lab_idx] - X[i]) ** 2).sum(axis=1))]
        ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], color=STRATA["novelty"], lw=1, ls="--", zorder=3)
    plot_points(ax, X, y, labeled=labeled, picks=by["novelty"], legend=False,
                title=f"novelty\nscore x distance to the nearest label, the top {q['novelty']}")
    add_legend(ax, [dot("#e34a33", "candidate, redder = more novel", edge="#555555"),
                    dot("black", "novelty pick", 9, hollow=True),
                    Line2D([], [], color=STRATA["novelty"], lw=1, ls="--", label="to its nearest label")] + labels_h)
    # 4: the batch
    ax = axes[3]
    plot_points(ax, X, y, labeled=labeled, legend=False,
                title=f"the batch\n{n} points: {q['exploit']} + {q['diversity']} + {q['novelty']}")
    batch_h = []
    for tag, idx in by.items():
        if len(idx):
            ax.scatter(X[idx, 0], X[idx, 1], s=120, color=STRATA[tag], edgecolor="black", lw=0.8, zorder=6)
            batch_h.append(dot(STRATA[tag], STRATA_LABELS[tag], 8, edge="black"))
    add_legend(ax, batch_h + labels_h)
    return fig, rule
