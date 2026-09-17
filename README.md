# IIASA Active Learning Mini Course

**Author**: *Moien Rangzan, International Institute for Applied Systems Analysis (IIASA), 2026*

Mini course that introduces active learning for Earth observation.

Four Jupyter notebooks on active learning for Earth observation, with
examples from the EarthQuery project (solar farms on AlphaEarth satellite
embeddings). 

To access the notebooks, go to the [notebooks/](notebooks/) folder and click on the notebook you want to open. 

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

3. To run the project I use VS Code with the Jupyter extension, but any JupyterLab or notebook
   environment works. 





## AlphaEarth embeddings

* AlphaEarth: Brown et al. (2025), Google Satellite Embedding V1, available
  in Earth Engine as `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`.

