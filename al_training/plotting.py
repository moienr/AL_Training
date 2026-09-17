"""Learning curves, the Austria probability map, its frames and animation helpers."""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# strategy colours (the palette of the report's Fig. 13)
COLORS = {"random": "#d62728", "entropy": "#ff7f0e", "typiclust": "#7f7f7f",
          "exploit": "#1f77b4", "exploit + entropy": "#2ca02c", "earthquery": "#9467bd"}
LABELS = {"random": "Random", "entropy": "Entropy", "typiclust": "TypiClust",
          "exploit": "Exploit only", "exploit + entropy": "Exploit + entropy",
          "earthquery": "EarthQuery rule (exploit + diversity + novelty)"}
METRIC_NAMES = {"bal_acc": "balanced accuracy", "n_found": "target-class points labelled so far",
                "recall_target": "recall of the target class",
                "recall_other": "recall of the other class", "auc": "ROC AUC"}

# the three parts of the EarthQuery rule (the colours of the three-criteria figure of the deck)
STRATA = {"exploit": "#d9a21b", "diversity": "#0e7c6b", "novelty": "#d95d39", "topup": "#9a9a9a", "init": "#555555"}
STRATA_LABELS = {"exploit": "exploit: highest score", "diversity": "diversity: one per cluster",
                 "novelty": "novelty: confident and unlike any label", "topup": "top-up by score",
                 "init": "starting labels"}
FARM, NEG = "#2ecc40", "#e41a1c"          # labelled solar farm (green) / labelled negative (red) on the probability map


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    return ax


def plot_learning_curves(results, metric="bal_acc", ax=None, title=None, chance=None, legend=True):
    """results: {strategy: history or [history per seed]}. Mean line, one-standard-deviation band."""
    ax = ax or plt.gca()
    for name, runs in results.items():
        runs = runs if isinstance(runs, list) else [runs]
        n = min(len(r["n_labels"]) for r in runs)
        x = np.array(runs[0]["n_labels"][:n])
        Y = np.array([r[metric][:n] for r in runs])
        col = COLORS.get(name, None)
        ax.plot(x, Y.mean(axis=0), color=col, lw=2, label=LABELS.get(name, name))
        if len(runs) > 1:                    # band = one standard deviation over the seeds
            sd = Y.std(axis=0)
            ax.fill_between(x, Y.mean(axis=0) - sd, Y.mean(axis=0) + sd, color=col, alpha=0.15, lw=0)
    if chance is not None:
        ax.axhline(chance, color="0.4", lw=0.8, ls="--", label="chance")
    ax.set_xlabel("number of labels"); ax.set_ylabel(METRIC_NAMES.get(metric, metric))
    tidy(ax)
    if legend:
        ax.legend(fontsize=8, frameon=False, loc="upper left" if metric == "n_found" else "center right")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def plot_budget_grid(by_budget, n_all, metrics=("n_found", "bal_acc", "recall_other"), figsize=None):
    """One row per budget (labels per round), one column per metric, legend in the first panel."""
    titles = {"n_found": f"solar farms labelled (of {n_all})", "bal_acc": "balanced accuracy",
              "recall_other": "recall of the background"}
    rows = len(by_budget)
    fig, axes = plt.subplots(rows, len(metrics), figsize=figsize or (17, 3.4 * rows), squeeze=False)
    for r, (b, results) in enumerate(by_budget.items()):
        for c, m in enumerate(metrics):
            ax = axes[r, c]
            plot_learning_curves(results, m, ax=ax, chance=0.5 if m == "bal_acc" else None,
                                 title=titles[m] if r == 0 else None, legend=(r == 0 and c == 0))
            if c == 0:
                ax.set_ylabel(f"{b} labels per round\n" + ax.get_ylabel())
            if r < rows - 1:
                ax.set_xlabel("")
    fig.tight_layout()
    return fig


def plot_strata_bars(ax, table, title=None):
    """Picks per part of the rule, with the positives filled and the hit rate written above."""
    order = [t for t in ("exploit", "diversity", "novelty", "topup") if t in table.index]
    x = np.arange(len(order))
    picks = table.loc[order, "picks"].to_numpy()
    pos = table.loc[order, "positives"].to_numpy()
    ax.bar(x, picks, color=[STRATA[t] for t in order], alpha=0.3, width=0.62)
    ax.bar(x, pos, color=[STRATA[t] for t in order], width=0.62)
    for i, (n, f) in enumerate(zip(picks, pos)):
        ax.text(i, n + max(picks) * 0.02, f"{f} of {n}\n({f / n:.0%})", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylim(0, max(picks) * 1.25); ax.set_ylabel("points picked (filled = solar farms)")
    tidy(ax)
    if title:
        ax.set_title(title, fontsize=10)
    return ax


# ------------------------------------------------------------- Austria ----
def austria_outline():
    with open(os.path.join(DATA, "austria_outline.geojson")) as f:
        g = json.load(f)["geometry"]
    return np.asarray(g["coordinates"][0], dtype=float)


K_LON = np.cos(np.radians(47.6))               # shrink longitudes so the map is not stretched


def draw_austria(ax, lw=0.8, face="#f3f2ee", return_patch=False):
    """The outline of Austria. Returns k, the factor that shrinks longitudes so the
    map is not stretched; with return_patch=True it returns (k, patch) as well."""
    ring = austria_outline()
    patch = Polygon(np.c_[ring[:, 0] * K_LON, ring[:, 1]], closed=True, fc=face, ec="#6f6f6f", lw=lw, zorder=1)
    ax.add_patch(patch)
    ax.set_xlim(ring[:, 0].min() * K_LON - 0.1, ring[:, 0].max() * K_LON + 0.1)
    ax.set_ylim(ring[:, 1].min() - 0.1, ring[:, 1].max() + 0.1)
    ax.set_aspect("equal"); ax.axis("off")
    return (K_LON, patch) if return_patch else K_LON


class ProbabilityGrid:
    """The model's P(solar farm) at the pool locations, put on a map grid.

    The pool is a sample of locations about 2 km apart, not every pixel. The
    grid has cells of `cell_km`. A cell takes the value of the samples that
    fall in it (the highest in mode "max", the average in mode "mean"), an
    empty cell takes the value of the nearest sample, and in mode "max" a
    maximum filter then gives every cell the highest probability within
    `radius_km`, which is what the coarse search of the tool looks at (the
    peak of every cell). A blur of `sigma_km` smooths the cell edges. Cells
    farther than `mask_km` from any sample stay transparent. This is the
    simulation's version of the probability map that Earth Engine draws in
    part 4.
    """

    def __init__(self, lon, lat, cell_km=1.0, radius_km=2.5, sigma_km=1.0, mode="max", mask_km=6.0):
        from scipy import ndimage
        self.nd, self.mode, self.radius_km = ndimage, mode, radius_km
        ring = austria_outline()
        self.x0, self.x1 = ring[:, 0].min() - 0.05, ring[:, 0].max() + 0.05
        self.y0, self.y1 = ring[:, 1].min() - 0.05, ring[:, 1].max() + 0.05
        dlat = cell_km / 111.0
        dlon = dlat / K_LON
        self.nx, self.ny = int(np.ceil((self.x1 - self.x0) / dlon)), int(np.ceil((self.y1 - self.y0) / dlat))
        self.ix = np.clip(((lon - self.x0) / dlon).astype(int), 0, self.nx - 1)
        self.iy = np.clip(((lat - self.y0) / dlat).astype(int), 0, self.ny - 1)
        self.count = np.zeros((self.ny, self.nx)); np.add.at(self.count, (self.iy, self.ix), 1)
        # every cell points to the nearest cell that holds a sample (a filled cell points to itself)
        dist, (iy_near, ix_near) = ndimage.distance_transform_edt(self.count == 0, return_indices=True)
        self.nearest = (iy_near, ix_near)
        self.empty = dist > mask_km / cell_km
        r = radius_km / cell_km
        n = int(np.ceil(r))
        yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
        self.footprint = (xx ** 2 + yy ** 2) <= r ** 2 + 1e-9
        self.sigma = sigma_km / cell_km
        self.extent = (self.x0 * K_LON, self.x1 * K_LON, self.y0, self.y1)

    def surface(self, p):
        p = np.asarray(p, dtype=float)
        if self.mode == "mean":
            total = np.zeros((self.ny, self.nx)); np.add.at(total, (self.iy, self.ix), p)
            z = (total / np.maximum(self.count, 1))[self.nearest]
            z = self.nd.uniform_filter(z, size=self.footprint.shape[0])
        else:
            z = np.zeros((self.ny, self.nx)); np.maximum.at(z, (self.iy, self.ix), p)
            z = z[self.nearest]                                  # empty cells: the nearest sample
            z = self.nd.maximum_filter(z, footprint=self.footprint)
        z = self.nd.gaussian_filter(z, self.sigma)
        return np.ma.array(z, mask=self.empty)


def draw_probability(ax, grid, p, patch, vmax=1.0, cmap="viridis"):
    """Draw the probability surface, clipped to the outline of Austria."""
    im = ax.imshow(grid.surface(p), extent=grid.extent, origin="lower", cmap=cmap, vmin=0, vmax=vmax,
                   interpolation="bilinear", zorder=2)
    im.set_clip_path(patch)
    return im


def draw_labels(ax, lon, lat, y, idx, new=None, size=1.0):
    """Labelled points on the map: farms orange, negatives white, the newest ringed."""
    idx = np.asarray(idx, dtype=int)
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    arts = [ax.scatter(lon[neg] * K_LON, lat[neg], s=11 * size, color=NEG, edgecolor="#333333", lw=0.4, zorder=4),
            ax.scatter(lon[pos] * K_LON, lat[pos], s=30 * size, color=FARM, edgecolor="black", lw=0.5, zorder=5)]
    if new is not None and len(new):
        arts.append(ax.scatter(lon[new] * K_LON, lat[new], s=95 * size, facecolor="none", edgecolor="black", lw=2.2, zorder=6))
        arts.append(ax.scatter(lon[new] * K_LON, lat[new], s=95 * size, facecolor="none", edgecolor="white", lw=1.0, zorder=7))
    return arts


def map_legend(ax, loc="lower left", ring=True):
    items = [Line2D([], [], marker="o", ls="", color=FARM, markeredgecolor="black", markersize=6, label="labelled: solar farm"),
             Line2D([], [], marker="o", ls="", color=NEG, markeredgecolor="#333333", markersize=4.5, label="labelled: not a farm")]
    if ring:
        items.append(Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="black", markersize=9, label="newest batch"))
    ax.legend(handles=items, loc=loc, fontsize=8, frameon=False)


class AustriaFrames:
    """Round i of a run: the probability map of Austria beside the search curve.

    Left: the model's P(solar farm) over the country (dark = low, yellow =
    high, the colours of the Earth Engine layer in part 4), the labelled
    points as they are chosen (orange = solar farm, white = not), the newest
    batch ringed; the title counts the farms found by each criterion and the
    negatives labelled. Right: how many of the pool's solar farms have been
    labelled so far, in total and by criterion, against what random sampling
    would find.
    """

    def __init__(self, hist, lon, lat, y, metric="n_found", mode="max", vmax=1.0, figsize=(11.5, 4.5)):
        self.h, self.lon, self.lat, self.y, self.metric = hist, lon, lat, y, metric
        self.fig, (self.ax, self.axc) = plt.subplots(1, 2, figsize=figsize, width_ratios=[1.75, 1],
                                                     constrained_layout=True)
        self.k, patch = draw_austria(self.ax, return_patch=True)
        self.grid = ProbabilityGrid(lon, lat, mode=mode)
        self.im = draw_probability(self.ax, self.grid, np.zeros(len(lon)), patch, vmax=vmax)
        cb = self.fig.colorbar(self.im, ax=self.ax, shrink=0.5, pad=0.01)
        cb.set_label(f"P(solar farm), peak within {self.grid.radius_km:g} km", fontsize=8); cb.ax.tick_params(labelsize=8)
        self.fig.get_layout_engine().set(wspace=0.08)
        self.lab_neg = self.ax.scatter([], [], s=11, color=NEG, edgecolor="#333333", lw=0.4, zorder=4)
        self.lab_pos = self.ax.scatter([], [], s=30, color=FARM, edgecolor="black", lw=0.5, zorder=5)
        self.ring_b = self.ax.scatter([], [], s=95, facecolor="none", edgecolor="black", lw=2.2, zorder=6)
        self.ring_w = self.ax.scatter([], [], s=95, facecolor="none", edgecolor="white", lw=1.0, zorder=7)
        self.title = self.ax.set_title("", fontsize=10, loc="left")
        map_legend(self.ax)
        # the curve
        x = np.array(hist["n_labels"]); n_all = int(y.sum()); col = COLORS.get(hist["strategy"], "k")
        if metric == "n_found":
            self.axc.plot(x, x * y.mean(), color="0.45", lw=1, ls="--", label="random sampling would find")
            self.axc.set_ylim(0, n_all * 1.06)
            self.axc.set_ylabel(f"solar farms labelled (of {n_all} in the pool)")
        else:
            self.axc.set_ylim(-0.02, 1.02); self.axc.set_ylabel(METRIC_NAMES.get(metric, metric))
        self.axc.plot(x, hist[metric], color="0.85", lw=1)
        self.line, = self.axc.plot([], [], color=col, lw=2.4, label=LABELS.get(hist["strategy"], hist["strategy"]) + ", all farms")
        self.dot, = self.axc.plot([], [], "o", color=col, markeredgecolor="black", ms=6)
        # farms found by each part of the strategy, cumulative over the rounds, and the negatives labelled
        picks = [np.asarray(p, dtype=int) for p in hist["picked"]]
        self.parts = [s for s in ("exploit", "diversity", "novelty") if any(s in r for r in hist["source"])]
        self.farms_by = {s: np.cumsum([int(sum(1 for j, t in zip(p, r) if t == s and y[j] == 1))
                                       for p, r in zip(picks, hist["source"])]) for s in self.parts}
        self.n_neg = np.cumsum([int((y[p] == 0).sum()) for p in picks])
        self.part_lines = {s: self.axc.plot([], [], color=STRATA[s], lw=1.5, label=f"farms found by {s}")[0] for s in self.parts}
        self.axc.set_xlabel("number of labels"); tidy(self.axc)
        self.axc.legend(fontsize=8, frameon=False, loc="upper left")

    def surface_at(self, i):
        """The probability surface after round i; flat before the first pick (2 labels are not a model)."""
        h = self.h
        if not h["probs"] or i < 1:
            return self.grid.surface(np.zeros(len(self.lon)))
        return self.grid.surface(h["probs"][min(i, len(h["probs"]) - 1)])

    def draw(self, i):
        h = self.h
        self.im.set_data(self.surface_at(i))
        idx = np.concatenate(h["picked"][:i + 1])
        pos, neg = idx[self.y[idx] == 1], idx[self.y[idx] == 0]
        self.lab_neg.set_offsets(np.c_[self.lon[neg] * self.k, self.lat[neg]])
        self.lab_pos.set_offsets(np.c_[self.lon[pos] * self.k, self.lat[pos]])
        new = h["picked"][i]
        off = np.c_[self.lon[new] * self.k, self.lat[new]] if len(new) else np.empty((0, 2))
        self.ring_b.set_offsets(off); self.ring_w.set_offsets(off)
        n_found, n_all = int(self.y[idx].sum()), int(self.y.sum())
        parts = ", ".join(f"{s} {int(self.farms_by[s][i])}" for s in self.parts)
        second = (f"farms found by criterion: {parts}; " if self.parts else "") + f"negatives labelled: {int(self.n_neg[i])}"
        self.title.set_text(f"round {i}: {len(idx)} labels, {n_found} of {n_all} solar farms found\n{second}")
        x = np.array(h["n_labels"][:i + 1])
        self.line.set_data(x, h[self.metric][:i + 1])
        self.dot.set_data([x[-1]], [h[self.metric][i]])
        for s, ln in self.part_lines.items():
            ln.set_data(x, self.farms_by[s][:i + 1])
        return (self.im, self.lab_neg, self.lab_pos, self.ring_b, self.ring_w, self.title, self.line, self.dot,
                *self.part_lines.values())

    def show_round(self, i):
        """Draw round i and display the figure (for an ipywidgets slider)."""
        from IPython.display import display
        self.draw(i)
        display(self.fig)

    def animation(self, interval=180):
        n = len(self.h["picked"])
        anim = animation.FuncAnimation(self.fig, self.draw, frames=n, interval=interval, blit=False)
        plt.close(self.fig)
        return anim


def probability_snapshots(hist, lon, lat, y, rounds=(0, 5, 20, 50), mode="max", vmax=1.0, figsize=(16, 3.6)):
    """The probability map at a few rounds of one run, side by side (a figure for slides)."""
    fig, axes = plt.subplots(1, len(rounds), figsize=figsize, constrained_layout=True)
    grid = ProbabilityGrid(lon, lat, mode=mode)
    for ax, r in zip(axes, rounds):
        k, patch = draw_austria(ax, return_patch=True)
        p = hist["probs"][min(r, len(hist["probs"]) - 1)] if r >= 1 else np.zeros(len(lon))
        im = draw_probability(ax, grid, p, patch, vmax=vmax)
        idx = np.concatenate(hist["picked"][:r + 1])
        draw_labels(ax, lon, lat, y, idx, size=0.7)
        ax.set_title(f"round {r}: {len(idx)} labels, {int(y[idx].sum())} farms found", fontsize=10)
    map_legend(axes[0], loc="lower left", ring=False)
    cb = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.01); cb.set_label(f"model P(solar farm), peak within {grid.radius_km:g} km", fontsize=9)
    return fig


def plot_round_picks(ax, hist, lon, lat, y, i, grid=None, vmax=1.0, title=None):
    """The batch of round i, coloured by the part of the rule that chose each point,
    on the probability map the rule was looking at (the map after round i - 1)."""
    k, patch = draw_austria(ax, return_patch=True)
    grid = grid or ProbabilityGrid(lon, lat)
    if hist["probs"]:
        draw_probability(ax, grid, hist["probs"][max(i - 1, 0)], patch, vmax=vmax)
    before = np.concatenate(hist["picked"][:i])
    draw_labels(ax, lon, lat, y, before, size=0.7)
    new, src = np.asarray(hist["picked"][i]), np.asarray(hist["source"][i])
    handles = []
    for tag in ("exploit", "diversity", "novelty", "topup"):
        sel = new[src == tag]
        if not len(sel):
            continue
        farm = y[sel] == 1
        ax.scatter(lon[sel[farm]] * k, lat[sel[farm]], s=150, facecolor=STRATA[tag], edgecolor="black", lw=0.8, zorder=8)
        ax.scatter(lon[sel[~farm]] * k, lat[sel[~farm]], s=150, facecolor="white", edgecolor=STRATA[tag], lw=2.5, zorder=8)
        handles.append(Line2D([], [], marker="o", ls="", color=STRATA[tag], markeredgecolor="black", markersize=10,
                              label=f"{STRATA_LABELS[tag]}: {int(farm.sum())} of {len(sel)} were farms"))
    handles += [Line2D([], [], marker="o", ls="", color="#bbbbbb", markeredgecolor="black", markersize=10, label="filled = a solar farm"),
                Line2D([], [], marker="o", ls="", color="white", markeredgecolor="#bbbbbb", markeredgewidth=2.5, markersize=10, label="hollow = not a farm")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.01), fontsize=8, frameon=False, ncol=2)
    ax.set_title(title or f"round {i}: the {len(new)} picks, coloured by the criterion that chose each one", fontsize=10, loc="left")
    return ax


def save_gif(anim, path, fps=6, dpi=90):
    anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    return path


# ---------------------------------------------------- what the model sees --
def load_shap():
    """Mean |SHAP| per embedding band of the Austrian solar farm model (data/austria_shap.csv)."""
    import pandas as pd
    return pd.read_csv(os.path.join(DATA, "austria_shap.csv"))["mean_abs_shap"].to_numpy()


def plot_embedding_bars(X, y, shap=None, farm=None, other=None, n_top=3, figsize=(11, 6.2)):
    """The 64 numbers of a typical solar farm and of a typical ordinary location
    (each the member of its class closest to the class average, unless farm and
    other are given), and the average difference between farms and the rest,
    with every bar coloured by how much the Austrian solar farm model relies on
    that dimension (mean |SHAP|). The most important dimensions are named."""
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    shap = load_shap() if shap is None else np.asarray(shap)

    def typical(cls):                      # the member of a class closest to the class average
        idx = np.flatnonzero(y == cls)
        return int(idx[np.argmin(((X[idx] - X[idx].mean(axis=0)) ** 2).sum(axis=1))])
    farm = typical(1) if farm is None else farm
    other = typical(0) if other is None else other
    cmap, norm = plt.get_cmap("YlOrRd"), Normalize(0, shap.max())
    colors = cmap(norm(shap))
    top = np.argsort(-shap)[:n_top]
    diff = X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)
    rows = [(X[farm], "a typical solar farm"), (X[other], "a typical ordinary location"),
            (diff, f"average farm minus\naverage other location\n({int((y == 1).sum())} farms, {int((y == 0).sum()):,} others)")]
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, constrained_layout=True)
    for ax, (v, name) in zip(axes, rows):
        ax.bar(range(64), v, color=colors, width=0.8, edgecolor="#555555", linewidth=0.3)
        ax.axhline(0, color="#888888", lw=0.6)
        lim = max(0.4, float(np.abs(v).max()) * 1.25)
        ax.set_ylim(-lim, lim); ax.set_ylabel(name, fontsize=9)
        for b in top:
            ax.annotate(f"A{b:02d}", (b, v[b]), xytext=(0, 6 if v[b] >= 0 else -12), textcoords="offset points",
                        ha="center", fontsize=8, fontweight="bold")
        tidy(ax)
    axes[-1].set_xlabel("embedding dimension (A00 to A63)")
    axes[-1].set_xticks(range(0, 64, 4)); axes[-1].set_xticklabels([f"A{i:02d}" for i in range(0, 64, 4)], fontsize=8)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.6, pad=0.01)
    cb.set_label("how much the Austrian solar farm model relies on the dimension\n(mean |SHAP| of the random forest)", fontsize=8)
    axes[0].set_title(f"the {n_top} most important dimensions are named: " + ", ".join(f"A{b:02d}" for b in top), fontsize=10, loc="left")
    return fig
