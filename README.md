# IIASA Active Learning Mini Course

Four Jupyter notebooks on active learning for Earth observation, with
examples from the EarthQuery project (solar farms on AlphaEarth satellite
embeddings). The first notebook needs no Earth observation background.

| Notebook | Content |
|---|---|
| `01_what_is_active_learning.ipynb` | The loop on a two-dimensional toy dataset: random, entropy, TypiClust, the rare-class case and the EarthQuery rule (exploit, diversity, novelty), with animations |
| `02_austria_simulation.ipynb` | The same loop on 20,663 AlphaEarth embeddings of Austria (1.25% solar farms): the probability map after each round, six strategies compared |
| `03_global_map.ipynb` | The global campaign: its method, the map of its positives by wave, the Austrian inventory of 355 solar farms |
| `04_earth_engine_operator.ipynb` | The operator window of the EarthQuery tool on the live Earth Engine map of Austria: classification, similarity search, Find Top Locations |

Each notebook opens in Google Colab from the button at its top; there its
first cell clones this repository and installs what Colab lacks. Notebooks 01
to 03 run offline from the files in `data/`. Notebook 04 needs an Earth
Engine account. Every animation is also saved as a GIF in
`outputs/`, and `outputs/preview/` holds HTML renders of the executed
notebooks for reading without running anything.

## Setup

1. Install [Miniforge](https://github.com/conda-forge/miniforge) (or any
   conda) and create the environment:

   ```
   conda env create -f environment.yml
   conda activate al_training
   ```

2. For notebook 04, register for Earth Engine (free for non-commercial use)
   at <https://code.earthengine.google.com/register>, create or choose a
   Google Cloud project there, and note its project id.

   In the notebook, put the project id in the `EE_PROJECT` line (or set the
   environment variable `EE_PROJECT`). The first time you run the notebook
   it shows a link: open it, allow access and paste the code back into the
   notebook. After that the login is stored.

3. Start Jupyter in this folder and open `notebooks/`:

   ```
   jupyter lab
   ```

The animations are drawn with matplotlib and shown as JavaScript players, so
no video codec is needed.

## What is in the repository

```
notebooks/        the four notebooks, committed without outputs
al_training/      the helper package the notebooks import
  models.py       the linear probe of the report and the random forest of the tool
  strategies.py   random, entropy, TypiClust, exploit, exploit + entropy, the EarthQuery thirds rule
  simulate.py     the active learning loop against a pool with known labels
  toy.py          the two-dimensional datasets, their plots and animations, the four-panel picture of the three criteria
  plotting.py     learning curves, the Austria probability map (frames, snapshots, picks by criterion)
  globalmap.py    the global campaign map and its animations
  earth_engine.py the Earth Engine session: scorers, hot-spot search, the thirds rule, the operator window (notebook 04)
data/
  austria_pool.npz        20,663 locations: lon, lat, label, 64-d AlphaEarth 2024 embedding
  austria_outline.geojson Austria
  austria_seeds.csv       122 Microsoft solar farms and 404 operator negatives, the starting labels of notebook 04
  austria_shap.csv        mean |SHAP| per embedding band of the Austrian solar farm model
  campaign_positives.csv  54,942 campaign positives with region, wave, pass and the region's finishing time
  microsoft_sites.csv     62,172 verified Microsoft solar farm sites
  world_outline.geojson   simplified country outlines
figures/          figures of the EarthQuery paper that the notebooks embed
outputs/          GIFs and snapshots written by the notebooks; outputs/preview/ holds HTML renders of executed copies
environment.yml
```

## Where the code and numbers come from

* The strategies and the linear probe follow the code that produced the
  simulation figures of the EarthQuery report (the entropy, TypiClust,
  max_positive and exploit_explore query strategies of the ALFM code base).
  TypiClust is from Hacohen, Dekel and Weinshall (2022), "Active learning on
  a budget: opposite strategies suit high and low budgets".
* The thirds rule (exploit, diversity, novelty) and the Earth Engine random
  forest (100 trees, 21 features per split, probability output) are the ones
  of the EarthQuery tool.
* The Austria pool is the simulation dataset of the report; the campaign
  points are the positives of the global campaign as of 15 September 2026.
* AlphaEarth: Brown et al. (2025), Google Satellite Embedding V1, available
  in Earth Engine as `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`.
* The Microsoft solar farm sites come from the Global Renewables Watch layer
  (Robinson et al., 2025), screened as described in the EarthQuery paper.

## Running the notebooks

* Notebooks 01 to 03 take about two, three and one minutes and write the GIFs
  to `outputs/`.
* In notebook 04, Find Top Locations over all of Austria takes about a
  minute; the search in the current map view takes a few seconds. Run
  Similarity takes 3 to 15 seconds, and the tiles refresh as you pan. Earth
  Engine limits interactive requests: if a request times out, run the cell
  again or search a smaller area.
* In VS Code, maps and buttons need the Jupyter extension on the machine that
  runs the kernel and the setting `jupyter.widgetScriptSources` set to
  `["jsdelivr.com", "unpkg.com"]`; JupyterLab needs nothing extra.
* To execute the notebooks from the command line, run them one at a time
  (`jupyter nbconvert --to notebook --execute` from the `notebooks/` folder;
  notebook 04 reads the project id from the environment variable
  `EE_PROJECT`). Running several at once can stall the kernels.

## License and citation

This course is by Moien Rangzan, International Institute for Applied Systems
Analysis (IIASA), 2026, and is released under the Creative Commons
Attribution 4.0 International license (CC BY 4.0, see `LICENSE`). The data
files keep the terms of their sources listed above.

To cite it: Moien Rangzan (2026). IIASA Active Learning Mini Course: active
learning for Earth observation with AlphaEarth embeddings. IIASA.
