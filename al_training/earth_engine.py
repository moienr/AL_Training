"""An EarthQuery operator session inside a notebook (notebook 04).

The model is trained inside Earth Engine on the AlphaEarth embeddings at the
labelled points, with the same settings as the EarthQuery tool: cosine
similarity to the mean positive embedding (positives only) or a random
forest (positives and negatives). Earth Engine renders the score map as map
tiles, so the whole of Austria is scored at 10 m without downloading anything.

Functions (no widgets, usable from plain code)
    init_ee, alphaearth, austria, load_seeds, train_rf, train_cosine,
    score_image, hot_peaks, refine_peaks, propose_batch, sample_embeddings
Class
    Session   the operator window: map, Classification, Similarity Search
              and Active Learning panels, laid out like the EarthQuery tool
"""
import json
import os
import time
import traceback

import numpy as np
import pandas as pd

import ee

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BANDS = [f"A{i:02d}" for i in range(64)]
# The three embedding dimensions that matter most to the Austrian solar farm model:
# mean |SHAP| of the local random forest in the three-country study (the EarthQuery tool
# analysis/data/tc_importance_vectors.npz, raw:local_shap, Austria row): A22, A01, A19.
# Stretched between the 2nd and 98th percentile of each band over the Austrian pool.
SHAP_BANDS = ["A22", "A01", "A19"]
AEE_VIS = {"bands": SHAP_BANDS, "min": [-0.13, -0.30, -0.19], "max": [0.15, 0.07, 0.26]}
SCORE_VIS = {"min": 0, "max": 1, "palette": ["440154", "3B528B", "21918C", "5DC863", "FDE725"]}
PROB_VIS = SCORE_VIS
HOT_VIS = {"min": 0, "max": 1, "palette": ["FDE725"]}
STRATA = {"exploit": "#d9a21b", "diversity": "#0e7c6b", "novelty": "#d95d39", "topup": "#9a9a9a"}
METHODS = {"cosine": "Cosine Similarity (positives only)", "rf": "Random Forest (positives + negatives)"}
# extra basemaps, the same imagery the tool's jury looks at (Google tiles, no key needed)
GOOGLE_TILES = {"Google Satellite": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                "Google Hybrid (with labels)": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}"}


# ------------------------------------------------------------- basics -----
def init_ee(project=None, force=False):
    """Connect to Earth Engine with the given Cloud project.

    Uses the stored login if there is one. Otherwise it shows a link: open
    it in a browser, allow access, and paste the code back into the
    notebook. With force=True the link is shown even if a login is stored.
    """
    project = project or os.environ.get("EE_PROJECT")
    if not project or "your-" in project:
        raise ValueError("Put the id of your Earth Engine Cloud project in EE_PROJECT (see the README).")
    if not force:
        try:
            ee.Initialize(project=project)
            print("Earth Engine ready, project", project)
            return
        except Exception:
            pass
    ee.Authenticate(auth_mode="notebook", force=True)
    ee.Initialize(project=project)
    print("Earth Engine ready, project", project)


def alphaearth(year=2024):
    """The AlphaEarth annual embedding mosaic: 64 bands A00..A63 at 10 m, unit length."""
    return (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01").mosaic())


def austria():
    """Austria as an Earth Engine polygon (from data/austria_outline.geojson)."""
    g = json.load(open(os.path.join(DATA, "austria_outline.geojson")))["geometry"]
    return ee.Geometry.Polygon(g["coordinates"])


def load_seeds(n_pos=10, n_neg=60, seed=0):
    """A small starting set: Microsoft solar farms (1) and operator negatives (0)."""
    S = pd.read_csv(os.path.join(DATA, "austria_seeds.csv"))
    rng = np.random.default_rng(seed)
    pos, neg = S[S.label == 1], S[S.label == 0]
    pos = pos.iloc[rng.permutation(len(pos))[:n_pos]]
    neg = neg.iloc[rng.permutation(len(neg))[:n_neg]]
    return pd.concat([pos, neg], ignore_index=True)


def to_fc(df, props=("label",)):
    feats = [ee.Feature(ee.Geometry.Point([float(r.lng), float(r.lat)]),
                        {p: (int(getattr(r, p)) if p == "label" else getattr(r, p)) for p in props})
             for r in df.itertuples()]
    return ee.FeatureCollection(feats)


def pool_farms():
    """The 259 solar farms of the Austrian pool of notebook 02, as lng and lat."""
    d = np.load(os.path.join(DATA, "austria_pool.npz"))
    m = d["label"] == 1
    return pd.DataFrame({"lng": d["lon"][m], "lat": d["lat"][m]})


def points_layer(df, color="ff8c1a", size=5):
    """The points of a DataFrame (lng, lat) as circles of a fixed pixel size, drawn by Earth Engine."""
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.lng), float(r.lat)])) for r in df.itertuples()])
    return fc.style(color="ffffff", fillColor=color, pointSize=size, pointShape="circle", width=1)


# ------------------------------------------------------------- model ------
def train_rf(labels, mosaic=None, n_trees=100):
    """Random forest probability map from a table with columns lng, lat, label.

    Earth Engine samples the 64 embedding values under each point, trains
    smileRandomForest (100 trees, 21 features per split) in probability mode
    and applies it to every pixel of the mosaic. Nothing is computed until a
    tile or a value is requested. The output band is called "score".
    """
    mosaic = mosaic or alphaearth()
    training = mosaic.sampleRegions(collection=to_fc(labels), properties=["label"], scale=10, geometries=False)
    clf = (ee.Classifier.smileRandomForest(numberOfTrees=n_trees, variablesPerSplit=21)
           .setOutputMode("PROBABILITY").train(training, "label", BANDS))
    return mosaic.classify(clf).rename("score")


def train_cosine(labels, mosaic=None):
    """Cosine similarity of every pixel to the mean embedding of the labelled solar farms.

    The cold-start scorer of the EarthQuery tool: it needs positives only.
    The embeddings have unit length, so the dot product with the unit mean
    positive is the cosine similarity, between -1 and 1. Band "score".
    """
    mosaic = mosaic or alphaearth()
    pos = labels[labels.label == 1]
    if len(pos) == 0:
        raise ValueError("Cosine similarity needs at least one solar farm label.")
    E = sample_embeddings(pos, mosaic, scale=10)
    E = E[~np.isnan(E).any(axis=1)]
    mean = E.mean(axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-9)
    weights = ee.Image.constant([float(v) for v in mean]).rename(BANDS)
    return mosaic.multiply(weights).reduce(ee.Reducer.sum()).rename("score")


def score_image(labels, mosaic=None, method="rf"):
    """The score map for a labelled table: "rf" or "cosine"."""
    if method == "cosine":
        return train_cosine(labels, mosaic)
    if method == "rf":
        return train_rf(labels, mosaic)
    raise ValueError(f"unknown method {method!r}; use 'rf' or 'cosine'")


def sample_embeddings(points, mosaic=None, scale=10):
    """The 64 embedding values under each point (rows of a DataFrame with lng, lat)."""
    mosaic = mosaic or alphaearth()
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.lng), float(r.lat)]), {"i": int(i)})
                               for i, r in zip(range(len(points)), points.itertuples())])
    got = mosaic.sampleRegions(collection=fc, properties=["i"], scale=scale, geometries=False).getInfo()
    E = np.full((len(points), 64), np.nan, dtype=np.float32)
    for f in got["features"]:
        E[int(f["properties"]["i"])] = [f["properties"][b] for b in BANDS]
    return E


# ------------------------------------------------------------- search -----
def hot_peaks(score, region, n=150, scale=320, min_threshold=0.2, percentile=99.9):
    """The highest-scoring blobs of the score map inside a region.

    Steps, all inside Earth Engine, one request:
      1. threshold = the 99.9th percentile of the map at the search scale (at least min_threshold)
      2. pixels above it are traced into connected blobs at that scale
      3. each blob reports its peak score and the peak's coordinates
      4. the n best blobs come back
    """
    thr = ee.Number(score.reduceRegion(ee.Reducer.percentile([percentile]), region, scale=scale,
                                       maxPixels=1e9, bestEffort=True).get("score")).max(min_threshold)
    hot = score.gte(thr).selfMask()
    blobs = hot.addBands(score.addBands(ee.Image.pixelLonLat())).reduceToVectors(
        reducer=ee.Reducer.max(3), scale=scale, geometry=region, eightConnected=False, maxPixels=1e10)
    info = blobs.limit(n, "max", False).set("threshold", thr).getInfo()
    # Reducer.max(3) over (score, longitude, latitude) reports max, max1 = longitude, max2 = latitude
    rows = [{"lng": f["properties"]["max1"], "lat": f["properties"]["max2"], "score": f["properties"]["max"]}
            for f in info["features"]]
    df = pd.DataFrame(rows, columns=["lng", "lat", "score"]).sort_values("score", ascending=False)
    df.attrs["threshold"] = info["properties"]["threshold"]
    return df.reset_index(drop=True)


def refine_peaks(score, cands, radius_m=240, scale=10):
    """Move each coarse peak to the best 10 m pixel within radius_m of it (one request)."""
    if len(cands) == 0:
        return cands
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.lng), float(r.lat)]).buffer(radius_m), {"i": int(i)})
                               for i, r in zip(range(len(cands)), cands.itertuples())])
    got = score.addBands(ee.Image.pixelLonLat()).reduceRegions(collection=fc, reducer=ee.Reducer.max(3), scale=scale).getInfo()
    out = cands.copy()
    for f in got["features"]:
        p = f["properties"]
        if p.get("max") is not None:
            out.loc[p["i"], ["lng", "lat", "score"]] = [p["max1"], p["max2"], p["max"]]
    return out


def km_coords(lng, lat, lat0=47.6):
    """Longitude and latitude as kilometres east and north (good enough for spacing rules)."""
    return np.c_[np.asarray(lng) * 111.0 * np.cos(np.radians(lat0)), np.asarray(lat) * 111.0]


def propose_batch(score, labels, region, n=12, pool=150, scale=320, mosaic=None, min_distance_km=1.0):
    """The EarthQuery thirds rule on the live map.

    exploit    the best-scoring blobs
    diversity  k-means over the candidates' embeddings, one per cluster
    novelty    score x cosine distance to the nearest labelled point
    Picks keep at least min_distance_km from each other. The winners are
    then refined to their best 10 m pixel. The search scale is 320 m by
    default: all of Austria takes about a minute at 320 m and several
    minutes at 160 m (the EarthQuery tool uses 160 m in the background).
    """
    from .strategies import EarthQueryRule
    mosaic = mosaic or alphaearth()
    cands = hot_peaks(score, region, n=pool, scale=scale)
    if len(cands) == 0:
        return cands
    E = sample_embeddings(cands, mosaic, scale=min(scale, 640))
    L = sample_embeddings(labels, mosaic, scale=10) if len(labels) else np.zeros((0, 64), np.float32)
    ok = ~np.isnan(E).any(axis=1)
    cands, E = cands[ok].reset_index(drop=True), E[ok]
    okL = ~np.isnan(L).any(axis=1)
    lab = labels[okL] if len(labels) else labels
    X = np.vstack([E, L[okL]])
    coords = np.vstack([km_coords(cands.lng, cands.lat), km_coords(lab.lng, lab.lat) if len(lab) else np.zeros((0, 2))])
    labeled = np.r_[np.zeros(len(E), bool), np.ones(int(okL.sum()), bool)]
    probs = np.r_[cands["score"].to_numpy(), np.zeros(int(okL.sum()))]
    rule = EarthQueryRule(X, pool=pool, coords=coords, min_distance=min_distance_km)
    idx = rule(dict(X=X, labeled=labeled, probs=np.c_[1 - probs, probs], rng=np.random.default_rng(0)), n)
    picked = cands.iloc[idx].copy()
    picked["source"] = rule.last_source
    picked["novelty"] = rule.last["distance"][rule.last["picked"]].round(3)
    picked = refine_peaks(score, picked.reset_index(drop=True))
    picked.attrs["threshold"] = cands.attrs.get("threshold")
    return picked


# ------------------------------------------------------------- session ----
class Session:
    """The EarthQuery operator window inside a notebook.

    Laid out like the tool: the map on the left, and on the right the three
    panels Classification (add the point at the map centre as positive or
    negative), Similarity Search (Run Similarity with Cosine Similarity or
    Random Forest, which draws the score map) and Active Learning (Find Top
    Locations with the thirds rule, a numbered list, Prev and Next).
    Every button has a plain method, so a session can also be driven from code:

        s = Session(labels=load_seeds())     # after init_ee
        s.show()
        s.run_similarity()                   # = Run Similarity
        s.find_top(n=12)                     # = Find Top Locations
        s.next_candidate(); s.add_positive() # go to a candidate, label it
        s.labels, s.candidates, s.rounds, s.save(path)
    """

    def __init__(self, labels=None, center=(47.6, 14.2), zoom=7, region=None, basemap="Esri.WorldImagery",
                 method="rf", height="600px"):
        import geemap
        import ipyleaflet as L
        import ipywidgets as W
        self.L, self.W = L, W
        self.mosaic = alphaearth()
        self.region = region or austria()
        self.labels = (labels.copy() if labels is not None else
                       pd.DataFrame(columns=["lng", "lat", "label", "source"]))
        self.labels["source"] = self.labels.get("source", "seed")
        self.labels["label"] = self.labels["label"].astype(int)
        self.score, self.candidates, self.current, self.rounds = None, None, None, []
        self.answered = set()
        # the map
        self.map = geemap.Map(center=center, zoom=zoom, basemap=basemap, height=height)
        self.map.layout = W.Layout(flex="1 1 0%", min_width="480px", height=height)
        # basemaps: the Esri imagery the map opened with, plus Google imagery to compare dates
        self.basemaps = {basemap: self.map.layers[0]}
        for name, url in GOOGLE_TILES.items():
            layer = L.TileLayer(url=url, name=name, attribution="Google", max_zoom=22, visible=False)
            self.map.add(layer)
            self.basemaps[name] = layer
        if hasattr(self.map, "add_layer_control"):
            self.map.add_layer_control()
        self.map.addLayer(self.mosaic.clip(self.region), AEE_VIS, "AlphaEarth, top 3 SHAP bands (A22 A01 A19)", False)
        self.label_layer = L.LayerGroup(name="classified points")
        self.cand_layer = L.LayerGroup(name="candidates")
        self.map.add(self.label_layer); self.map.add(self.cand_layer)
        self.map.on_interaction(self._on_click)
        # the panels
        self._build_panel(method)
        self._draw_labels()
        self._say("Add points, then press <b>Run Similarity</b> to draw the score map.")

    # ----- widgets
    def _build_panel(self, method):
        W = self.W
        wide = {"description_width": "150px"}
        self.status = W.HTML(layout=W.Layout(min_height="44px"))
        self.basemap = W.Dropdown(options=list(self.basemaps), value=list(self.basemaps)[0], description="Basemap",
                                  style=wide, layout=W.Layout(width="330px"))
        self.basemap.observe(lambda ch: self.set_basemap(ch["new"]), names="value")
        # Classification
        self.b_pos = W.Button(description="Add positive at centre", button_style="success", icon="plus",
                              tooltip="Label the map centre as a solar farm")
        self.b_neg = W.Button(description="Add negative at centre", button_style="danger", icon="minus",
                              tooltip="Label the map centre as not a solar farm")
        self.b_undo = W.Button(description="Undo last", icon="undo")
        self.click_mode = W.Dropdown(options=[("a click on the map adds a positive", 1),
                                              ("a click on the map adds a negative", 0),
                                              ("clicks on the map do nothing", -1)], value=1,
                                     layout=W.Layout(width="290px"))
        self.counts = W.HTML()
        self.b_save = W.Button(description="Save as CSV", icon="save")
        # Similarity Search
        self.method = W.Dropdown(options=[(v, k) for k, v in METHODS.items()], value=method,
                                 description="Method", style=wide, layout=W.Layout(width="330px"))
        self.b_run = W.Button(description="Run Similarity", button_style="primary", icon="play")
        self.b_clear = W.Button(description="Clear", icon="eraser")
        # Active Learning
        self.n_cands = W.BoundedIntText(value=12, min=3, max=60, description="Number of candidates", style=wide,
                                        layout=W.Layout(width="230px"))
        self.scale = W.Dropdown(options=[("640 m (fastest)", 640), ("320 m", 320), ("160 m (map view only)", 160)],
                                value=320, description="Discovery scale", style=wide, layout=W.Layout(width="330px"))
        self.area = W.Dropdown(options=[("the whole region", "region"), ("the current map view", "view")],
                               value="region", description="Search area", style=wide, layout=W.Layout(width="330px"))
        self.min_dist = W.BoundedFloatText(value=1.0, min=0.0, max=50.0, step=0.5, description="Min distance (km)",
                                           style=wide, layout=W.Layout(width="230px"))
        self.b_find = W.Button(description="Find Top Locations", button_style="primary", icon="search")
        self.b_prev = W.Button(description="Prev", icon="arrow-left", layout=W.Layout(width="90px"))
        self.b_next = W.Button(description="Next", icon="arrow-right", layout=W.Layout(width="90px"))
        self.nav_info = W.HTML()
        self.results = W.HTML()
        self.log = W.Output(layout=W.Layout(max_height="140px", overflow="auto"))
        # wiring: every handler reports what it is doing at once and shows errors in the status line
        self.b_pos.on_click(self._guard("Adding a positive ...", self.add_positive))
        self.b_neg.on_click(self._guard("Adding a negative ...", self.add_negative))
        self.b_undo.on_click(self._guard("Undoing ...", self.undo))
        self.b_save.on_click(self._guard("Saving ...", lambda: self.save()))
        self.b_run.on_click(self._guard("Training in Earth Engine and drawing the score map (5 to 15 s) ...",
                                        self.run_similarity))
        self.b_clear.on_click(self._guard("Clearing ...", self.clear))
        self.b_find.on_click(self._guard("Searching the score map for candidates (about a minute for all of Austria) ...",
                                         self.find_top))
        self.b_prev.on_click(self._guard("", self.prev_candidate))
        self.b_next.on_click(self._guard("", self.next_candidate))

        def section(title, *children):
            return W.VBox([W.HTML(f"<div style='font:bold 13px sans-serif;margin:2px 0 4px 0'>{title}</div>"), *children],
                          layout=W.Layout(border="1px solid #d8d8d8", padding="6px 8px", margin="0 0 6px 0"))
        self.panel = W.VBox([
            self.status,
            self.basemap,
            section("Classification", W.HBox([self.b_pos, self.b_neg]), W.HBox([self.b_undo, self.b_save]),
                    self.click_mode, self.counts),
            section("Similarity Search", self.method, W.HBox([self.b_run, self.b_clear])),
            section("Active Learning", self.n_cands, self.scale, self.area, self.min_dist, self.b_find,
                    W.HBox([self.b_prev, self.b_next, self.nav_info]), self.results),
            self.log,
        ], layout=W.Layout(flex="0 0 370px", padding="0 0 0 10px"))

    def show(self):
        return self.W.HBox([self.map, self.panel], layout=self.W.Layout(width="100%"))

    def set_basemap(self, name):
        """Show one basemap: Esri World Imagery, Google Satellite or Google Hybrid."""
        if name not in self.basemaps:
            raise ValueError(f"unknown basemap {name!r}; choose from {list(self.basemaps)}")
        for n, layer in self.basemaps.items():
            layer.visible = (n == name)
        if self.basemap.value != name:
            self.basemap.value = name

    def _guard(self, working, fn):
        def run(_=None):
            if working:
                self._say(working, kind="work")
            try:
                fn()
            except Exception as e:
                self._say(f"Error: {e}", kind="error")
                with self.log:
                    traceback.print_exc()
        return run

    def _say(self, html, kind="ok"):
        color = {"ok": "#1a1a1a", "work": "#7a5b00", "error": "#b00020"}[kind]
        self.status.value = f"<div style='font:13px sans-serif;color:{color};padding:4px 0'>{html}</div>"
        self._update_counts()

    def _update_counts(self):
        p, n = self._counts()
        self.counts.value = f"<div style='font:12px sans-serif;color:#444'>Classified points: {p} positive, {n} negative</div>"

    def _counts(self):
        return int((self.labels.label == 1).sum()), int((self.labels.label == 0).sum())

    # ----- map drawing
    def _draw_labels(self):
        self.label_layer.clear_layers()
        for r in self.labels.itertuples():
            col = "#ff8c1a" if int(r.label) == 1 else "#3b6ea5"
            self.label_layer.add_layer(self.L.CircleMarker(location=(float(r.lat), float(r.lng)), radius=5,
                                                           color="white", weight=1, fill_color=col, fill_opacity=0.95))

    def _replace_layer(self, name, image, vis, shown=True):
        for lyr in list(self.map.layers):
            if getattr(lyr, "name", "") == name:
                self.map.remove_layer(lyr)
        if image is not None:
            self.map.addLayer(image, vis, name, shown)

    def _draw_candidates(self):
        self.cand_layer.clear_layers()
        if self.candidates is None:
            return
        for i, r in enumerate(self.candidates.itertuples()):
            col = STRATA.get(r.source, "#9a9a9a")
            done = i in self.answered
            self.cand_layer.add_layer(self.L.CircleMarker(location=(float(r.lat), float(r.lng)), radius=10,
                                                          color=col, weight=3, fill_color="white",
                                                          fill_opacity=0.15 if done else 0.6))
            self.cand_layer.add_layer(self.L.Marker(location=(float(r.lat), float(r.lng)), draggable=False,
                                                    icon=self.L.DivIcon(html=f"<div style='font:bold 12px sans-serif;color:{col}'>{i + 1}</div>",
                                                                        icon_size=[20, 20], icon_anchor=[-12, 10])))

    def _draw_results(self):
        if self.candidates is None or len(self.candidates) == 0:
            self.results.value = ""
            self.nav_info.value = ""
            return
        rows = []
        for i, r in enumerate(self.candidates.itertuples()):
            col = STRATA.get(r.source, "#9a9a9a")
            mark = "&#9654; " if i == self.current else ""
            done = " (answered)" if i in self.answered else ""
            weight = "bold" if i == self.current else "normal"
            rows.append(f"<tr style='font-weight:{weight}'><td>{mark}{i + 1}</td>"
                        f"<td><span style='color:{col}'>&#9679;</span> {r.source}</td>"
                        f"<td>{r.score:.2f}</td><td style='color:#666'>{done}</td></tr>")
        self.results.value = ("<table style='font:12px sans-serif;border-spacing:6px 1px'>"
                              "<tr style='color:#666'><td>#</td><td>criterion</td><td>score</td><td></td></tr>"
                              + "".join(rows) + "</table>")
        self.nav_info.value = (f"<span style='font:12px sans-serif'>&nbsp;candidate {self.current + 1} of {len(self.candidates)}</span>"
                               if self.current is not None else "")

    # ----- labelling
    def add_label(self, lng, lat, label, source="click"):
        self.labels.loc[len(self.labels)] = [float(lng), float(lat), int(label), source]
        self._draw_labels()
        p, n = self._counts()
        self._say(f"Added a {'positive' if label else 'negative'} at {lat:.4f}, {lng:.4f} ({source}). "
                  f"{p} positive, {n} negative. Press <b>Run Similarity</b> to update the map.")

    def _at_centre(self, label):
        lat, lng = self.map.center
        source = "centre"
        if self.candidates is not None and self.current is not None and self.current not in self.answered:
            r = self.candidates.iloc[self.current]
            if abs(r.lat - lat) < 0.0005 and abs(r.lng - lng) < 0.0008:      # the map is still on the candidate
                lng, lat, source = float(r.lng), float(r.lat), f"candidate:{r.source}"
                self.answered.add(self.current)
        self.add_label(lng, lat, label, source)
        self._draw_candidates(); self._draw_results()

    def add_positive(self):
        """Label the map centre (or the candidate shown) as a solar farm."""
        self._at_centre(1)

    def add_negative(self):
        """Label the map centre (or the candidate shown) as not a solar farm."""
        self._at_centre(0)

    def undo(self):
        if len(self.labels):
            self.labels = self.labels.iloc[:-1].reset_index(drop=True)
            self._draw_labels(); self._say("Removed the last point.")

    def _on_click(self, **kw):
        if kw.get("type") == "click" and self.click_mode.value in (0, 1):
            lat, lng = kw["coordinates"]
            self._guard("", lambda: self.add_label(lng, lat, self.click_mode.value, "click"))()

    # ----- model
    def run_similarity(self, method=None):
        """Train the chosen scorer in Earth Engine and draw the score map (= Run Similarity)."""
        method = method or self.method.value
        p, n = self._counts()
        if method == "rf" and (p == 0 or n == 0):
            self._say("The random forest needs at least one positive and one negative.", kind="error"); return
        if method == "cosine" and p == 0:
            self._say("Cosine similarity needs at least one positive.", kind="error"); return
        t0 = time.time()
        self.score = score_image(self.labels, self.mosaic, method).clip(self.region)
        self.method_used = method
        self._replace_layer("score: " + METHODS[method], self.score, SCORE_VIS)
        self._replace_layer("score > 0.5", self.score.gt(0.5).selfMask(), HOT_VIS, False)
        self.rounds.append({"round": len(self.rounds), "method": method, "n_pos": p, "n_neg": n})
        self._say(f"{METHODS[method]} trained on {p} positives and {n} negatives in {time.time() - t0:.1f} s. "
                  f"The tiles are computed as you pan and zoom (dark = low, yellow = high).")

    train = run_similarity

    def clear(self):
        for name in list(l.name for l in self.map.layers if getattr(l, "name", "").startswith(("score", "hot spots"))):
            self._replace_layer(name, None, None)
        self.score, self.candidates, self.current, self.answered = None, None, None, set()
        self._draw_candidates(); self._draw_results()
        self._say("Cleared the score map and the candidates.")

    # ----- candidates
    def view_region(self):
        """The area currently shown on the map, for a quick local search."""
        if not self.map.bounds:                  # the map has not been displayed yet
            print("The map has no view yet; searching the whole region instead.")
            return self.region
        (south, west), (north, east) = self.map.bounds
        return ee.Geometry.Rectangle([west, south, east, north])

    def find_top(self, n=None, scale=None, region=None, min_distance_km=None):
        """Find Top Locations: the thirds rule on the current score map."""
        if self.score is None:
            self.run_similarity()
            if self.score is None:
                return
        n = int(n or self.n_cands.value)
        scale = scale or self.scale.value
        if region is None:
            region = self.view_region() if self.area.value == "view" else self.region
        min_distance_km = self.min_dist.value if min_distance_km is None else min_distance_km
        t0 = time.time()
        self.candidates = propose_batch(self.score, self.labels, region, n=n, scale=scale, mosaic=self.mosaic,
                                        min_distance_km=min_distance_km)
        self.current, self.answered = None, set()
        thr = self.candidates.attrs.get("threshold")
        if thr is not None:
            self._replace_layer("hot spots (search threshold)", self.score.gte(thr).selfMask(), HOT_VIS, False)
        self._draw_candidates(); self._draw_results()
        by = self.candidates["source"].value_counts().to_dict() if len(self.candidates) else {}
        self._say(f"{len(self.candidates)} candidates in {time.time() - t0:.0f} s at {scale} m (threshold {thr or 0:.2f}): "
                  f"{by}. Press <b>Next</b> to go to each one, look at the imagery, then <b>Add positive</b> "
                  f"or <b>Add negative at centre</b>.")
        return self.candidates

    propose = find_top

    def go_to(self, i):
        if self.candidates is None or len(self.candidates) == 0:
            self._say("No candidates yet: press <b>Find Top Locations</b>.", kind="error"); return
        self.current = int(i) % len(self.candidates)
        r = self.candidates.iloc[self.current]
        self.map.center = (float(r.lat), float(r.lng)); self.map.zoom = 16
        self._draw_results()
        self._say(f"Candidate {self.current + 1} of {len(self.candidates)}: score {r.score:.2f}, chosen by "
                  f"<b style='color:{STRATA.get(r.source, '#333')}'>{r.source}</b>. Solar farm or not? "
                  f"<b>Add positive</b> or <b>Add negative at centre</b>.")

    def next_candidate(self):
        self.go_to(0 if self.current is None else self.current + 1)

    def prev_candidate(self):
        self.go_to(-1 if self.current is None else self.current - 1)

    def save(self, path=None):
        path = path or os.path.join(DATA, "..", "outputs", "my_austria_labels.csv")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.labels.to_csv(path, index=False)
        self._say(f"Saved {len(self.labels)} points to {os.path.abspath(path)}")
        return path
