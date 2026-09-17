"""The global campaign map for notebook 03: data loading and animation frames."""
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import PolyCollection

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TEAL, GOLD, LAND, EDGE = "#0e7c6b", "#c9a24d", "#efeee9", "#bdbbb2"


def load_points():
    """Campaign positives (with a time order) and the Microsoft sites."""
    P = pd.read_csv(os.path.join(DATA, "campaign_positives.csv"))
    P["region_finished"] = pd.to_datetime(P["region_finished"], utc=True, errors="coerce")
    # order in which the points appeared: the hand-labelled pilot regions first,
    # then the waves in the order the regions finished, then the extra passes (wave 7)
    key = np.where(P["hand_labelled"] == 1, 0, np.where(P["wave"] == 7, 2, 1))
    P["step"] = pd.Series(key).astype(int)
    P = P.sort_values(["step", "region_finished", "wave", "region", "pass_no"], kind="stable").reset_index(drop=True)
    P["order"] = np.arange(len(P))
    S = pd.read_csv(os.path.join(DATA, "microsoft_sites.csv"))
    return P, S


def world_polys():
    feats = json.load(open(os.path.join(DATA, "world_outline.geojson")))["features"]
    return [np.asarray(r[0], float) for f in feats for r in f["geometry"]["coordinates"]]


def draw_world(ax, lat_min=-56, lat_max=75):
    ax.add_collection(PolyCollection(world_polys(), facecolors=LAND, edgecolors=EDGE, linewidths=0.3))
    ax.set_xlim(-180, 180); ax.set_ylim(lat_min, lat_max)
    ax.set_aspect(1.0); ax.axis("off")


def frame_stages(P, n_frames=40):
    """Cut the ordered points into n_frames cumulative stages (point counts)."""
    hand = int((P["step"] == 0).sum())
    rest = np.linspace(hand, len(P), n_frames - 1).astype(int)[1:]
    return np.r_[hand, rest]


class WorldFrames:
    """Points appear in the order the campaign produced them."""

    def __init__(self, P, S, n_frames=40, figsize=(11, 5.4)):
        self.P, self.S = P, S
        self.stages = frame_stages(P, n_frames)
        self.fig, self.ax = plt.subplots(figsize=figsize, constrained_layout=True)
        draw_world(self.ax)
        self.ax.scatter(S["lng"], S["lat"], s=1.2, color=GOLD, lw=0, alpha=0.8, zorder=2)
        self.sc = self.ax.scatter([], [], s=1.2, color=TEAL, lw=0, alpha=0.9, zorder=3)
        self.text = self.ax.text(-178, -50, "", fontsize=10, va="bottom")
        self.ax.scatter([], [], s=12, color=GOLD, label=f"Microsoft sites ({len(S):,})")
        self.ax.scatter([], [], s=12, color=TEAL, label="campaign positives")
        self.ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.06), fontsize=9, frameon=False)

    def draw(self, i):
        n = self.stages[i]
        Q = self.P.iloc[:n]
        self.sc.set_offsets(Q[["lng", "lat"]].to_numpy())
        last = Q.iloc[-1] if n else None
        when = "" if last is None or pd.isna(last["region_finished"]) else last["region_finished"].strftime("%d %b %Y")
        stage = "pilot regions, labelled by people" if last is not None and last["step"] == 0 else \
                ("extra passes over finished regions" if last is not None and last["step"] == 2 else f"wave {int(last['wave'])}, {when}")
        self.text.set_text(f"{n:,} solar farm locations   {stage}")
        return self.sc, self.text

    def animation(self, interval=150):
        anim = animation.FuncAnimation(self.fig, self.draw, frames=len(self.stages), interval=interval, blit=False)
        plt.close(self.fig)
        return anim


def stage_frames(P):
    """Cumulative point counts per stage: pilots, waves 1 to 6, extra passes."""
    names, counts = [], []
    hand = int((P["step"] == 0).sum())
    names.append("pilot regions, hand labelled"); counts.append(hand)
    for w in range(1, 7):
        n = int(((P["step"] == 1) & (P["wave"] <= w)).sum()) + hand
        names.append(f"wave {w}"); counts.append(n)
    names.append("extra passes"); counts.append(len(P))
    return names, counts


def plotly_animation(P, S, height=560):
    """An interactive map with a slider over the campaign stages (plotly)."""
    import plotly.graph_objects as go
    names, counts = stage_frames(P)
    ms = go.Scattermap(lat=S["lat"], lon=S["lng"], mode="markers", name=f"Microsoft sites ({len(S):,})",
                       marker=dict(size=3, color=GOLD, opacity=0.7), hoverinfo="skip")
    def ours(n):
        Q = P.iloc[:n]
        return go.Scattermap(lat=Q["lat"], lon=Q["lng"], mode="markers", name="campaign positives",
                             marker=dict(size=3, color=TEAL, opacity=0.85),
                             text=Q["region"], hovertemplate="%{text}<extra></extra>")
    fig = go.Figure(data=[ms, ours(counts[0])],
                    frames=[go.Frame(data=[ours(n)], traces=[1], name=nm,
                                     layout=go.Layout(title=f"{nm}: {n:,} solar farm locations"))
                            for nm, n in zip(names, counts)])
    steps = [dict(method="animate", label=nm, args=[[nm], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
             for nm in names]
    fig.update_layout(map=dict(style="carto-positron", center=dict(lat=22, lon=15), zoom=1.1), height=height,
                      margin=dict(l=0, r=0, t=40, b=0), title=f"{names[0]}: {counts[0]:,} solar farm locations",
                      legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
                      sliders=[dict(steps=steps, active=0, currentvalue=dict(visible=False), pad=dict(t=10))],
                      updatemenus=[dict(type="buttons", showactive=False, x=0.01, y=0.06, buttons=[
                          dict(label="play", method="animate", args=[None, dict(frame=dict(duration=900, redraw=True), fromcurrent=True, transition=dict(duration=0))])])])
    return fig
