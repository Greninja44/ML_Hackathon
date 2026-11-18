# PathSnap — Distributed Map-Matching & Route Reconstruction

PathSnap matches noisy GPS trajectories to a road network and reconstructs the
travelled route. Machine-learning models score candidate road segments for each
GPS fix, and an HMM/Viterbi decoder turns those scores into a connected route.
A FastAPI service (Laptop 1) evaluates the reserved trips; a browser UI
(Laptop 2) lets the evaluator pick a trip and see the result.

```
GPS track + road network
  → cleaning & projection (src/data/preprocess.py, src/geo.py)
  → candidate arcs per GPS point (src/map_matching/candidates.py)
  → pair features (src/data/features.py)
  → ML candidate probabilities (src/models/inference.py)
  → HMM emissions + network transitions → Viterbi (src/map_matching/)
  → reconstructed route + server-side metrics (src/api/)
  → map in the evaluator UI (frontend/)
```

* Requirements extracted from the official guidelines: [`PROJECT_REQUIREMENTS.md`](PROJECT_REQUIREMENTS.md)
* Design decisions (for the report and viva): [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)

## Repository layout

| Path | Purpose |
|---|---|
| `configs/pipeline.json` | All tunable settings (radii, HMM params, CV folds, seed, paths) |
| `src/data/` | Loading (`io.py`), validation, cleaning, label alignment, features, EDA |
| `src/map_matching/` | Candidates, HMM emissions/transitions, Viterbi, reconstruction, route metrics |
| `src/models/` | Model registry, training + CV, evaluation metrics, Optuna optimisation, inference API |
| `src/experiments/` | Classifier-only vs HMM comparison, decoder tuning, report maps |
| `src/visualization/` | Matplotlib plots and Folium maps |
| `src/api/` | FastAPI app, schemas, service |
| `frontend/` | Evaluator UI (static HTML/JS, Leaflet vendored) |
| `notebooks/` | 01 EDA, 02 training/HPO, 03 decoder evaluation |
| `scripts/` | `run_eda.py`, `run_all.sh`, `serve_ui.py` |
| `tests/` | Unit, API and end-to-end tests (`tests/synthetic.py` = synthetic fixture for tests only) |
| `data/raw/{train,student_test,eval_test}/` | Raw trips (not committed) |
| `models/final/` | Final model + HMM config used by the API |
| `results/` | All generated tables, plots, logs and maps |

## Installation

Python 3.11.

```bash
python3.11 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

XGBoost/LightGBM are optional; they need the OpenMP runtime (`brew install libomp` on macOS).

## Data

Put the supplied trips under `data/raw/` by partition. Any sub-folder layout
works: the loader finds every `*.track` file and pairs it with the
`.nodes/.arcs/.route` files that have the same name (or the only such file in that folder).

```
data/raw/train/<trip>/<trip>.track|.nodes|.arcs|.route        (80 trips)
data/raw/student_test/...                                    (10 trips)
data/raw/eval_test/...                                       (10 trips — API only)
```

Use `PATHSNAP_DATA_DIR` to point somewhere else. Development code **cannot** read
`eval_test` (`iter_partition` raises an error); only the API service reads it.

## Running the development pipeline

```bash
python scripts/run_eda.py                          # EDA tables, plots, Folium maps → results/eda/
python -m src.models.train                         # ≥5 models, GroupKFold CV, student-test comparison
python -m src.models.optimize                      # Optuna on the selected model → models/final/
python -m src.experiments.compare_decoders --tune  # HMM tuning + classifier vs HMM + report maps
python -m pytest -q
```

Or run everything with `scripts/run_all.sh`. Key outputs:

| File | Report section |
|---|---|
| `results/eda/*` | 2.2 EDA (tables, plots, Folium maps) |
| `results/model_comparison.csv`, `results/plots/*` | 3.1 comparison table, 3.3 plots |
| `results/models/<model>/` | 3.2 per-model metrics, confusion matrix, importance, loss curves |
| `results/optuna/<model>/`, `results/tuned_vs_untuned.csv` | 4 Bayesian optimisation |
| `results/decoders/*` | 5 HMM vs classifier, 6 Folium maps |
| `results/logs/*.log` | experiment logs |

## Deployment (evaluation day)

**Laptop 1: API**

```bash
export PATHSNAP_SERVE_PARTITION=eval_test       # student_test for rehearsal
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Swagger docs are at `http://<laptop1-ip>:8000/docs`. Keep this terminal visible: every
request is logged there and in `results/logs/api.log`. Allow port 8000 through the firewall.

**Laptop 2: UI**

```bash
python scripts/serve_ui.py --port 5173
# open http://localhost:5173/?api=http://<laptop1-ip>:8000
```

Both laptops must be on the same network. The UI needs only Python's standard
library. Map tiles come from OpenStreetMap; without internet the route and GPS
layers still draw, just without a basemap.

### API reference

| Method | Path | Body / result |
|---|---|---|
| GET | `/health` | model loaded, partition served, number of trips |
| GET | `/trips` | `[{trip_id, n_observations, duration_s, bbox}]` |
| GET | `/trips/{trip_id}/network` | road segments near the trajectory |
| POST | `/reconstruct` | `{"trip_id": "...", "mode": "hmm" \| "classifier", "compare": true}` |

Example:

```bash
curl -X POST http://localhost:8000/reconstruct -H 'Content-Type: application/json' \
     -d '{"trip_id": "<id from /trips>", "mode": "hmm", "compare": true}'
```

The response contains each observation's predicted arc and snapped position, the
route arc ids and polylines, server-side metrics (Top-1 accuracy, candidate miss
rate, route continuity, sequence precision/recall/F1, edit similarity), the
optional comparison of both modes, and timings. Reference-route data is never
included (a test enforces this).

## Data-leakage safeguards

* Splits are by trip; CV uses `GroupKFold(groups=trip_id)`.
* Scalers are fitted inside each CV fold's pipeline.
* Features are computed from GPS + network only; a test checks that changing the route does not change the features.
* The decoder is tuned on out-of-fold probabilities for training trips.
* `eval_test` is blocked for all development code; the API keeps the reference route in-process.

## Two workflows

| | **CSV workflow** (`src/csv_pipeline/`) | **Raw-data workflow** (`src/data/`, `src/map_matching/`, `src/api/`) |
|---|---|---|
| Input | Person 1's 4 CSVs in `data/person1/` | `.track/.nodes/.arcs/.route` in `data/raw/` |
| Available now | ✅ | ❌ raw files not yet received |
| Labels | route sequence aligned to points (monotone DP on `distance_m`) | same idea on true geometry |
| Models, CV, Optuna | ✅ | ✅ (code ready) |
| HMM | topology-only (node connectivity, no distances) | full (network driving distance vs GPS distance) |
| Maps / API / UI | ❌ CSVs have no coordinates | ✅ (code ready) |

```bash
python -m src.csv_pipeline.experiments compare     # ≥5 models, CV + student-test
python -m src.csv_pipeline.experiments optimize    # Optuna on the selected model
python -m src.csv_pipeline.experiments decoders    # topology HMM tuning + comparison
```

Results go to `results/csv_pipeline/`. Data findings and limitations are in
[`docs/CSV_DATA_REPORT.md`](docs/CSV_DATA_REPORT.md). What we still need from
Person 1 is in [`PERSON1_INTEGRATION.md`](PERSON1_INTEGRATION.md).

## Status

* The CSV workflow has run on the real CSVs; see `results/csv_pipeline/`.
* The raw-data workflow (maps, full HMM, API, UI) has only been tested on the
  **synthetic** fixture in `tests/synthetic.py`. Those numbers are not results.
  It needs the raw files before it can be run for real.

## AI-assistance disclosure

Parts of this codebase were developed with the help of an AI coding assistant.
The team is responsible for verifying, running, and being able to explain all
code and every reported result (Deliverables §8, Assessment Policy §5).
