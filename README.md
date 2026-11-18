# PathSnap

GPS is noisy. In a city a car's position can drift onto the parallel street,
jump across a block, or vanish for a few seconds. PathSnap takes those noisy
points and works out which roads the car actually drove on.

It does this in two steps. First a classifier looks at each GPS point and
scores the nearby road segments it could belong to. Then a hidden Markov model
strings those guesses together, using the road network to rule out routes that
are impossible to drive, such as jumping between two roads that don't connect.

This is our submission for the UE24CS352A PathSnap hackathon.

## How well it works

All numbers are from the 10 student-testing trips (19,160 GPS points), which
the models never trained on. Every figure comes from a file in `results/`.

| Method | Points on the right road | Route continuity | Route F1 |
|---|---|---|---|
| Snap to the nearest road | 54.4 % | 27.8 % | 61.6 % |
| Classifier only | 87.8 % | 78.0 % | 81.4 % |
| **Classifier + HMM** | **88.5 %** | **89.5 %** | **93.0 %** |

The HMM's main contribution is a route that holds together. The classifier
alone often hops to a wrong road for one or two points, which barely affects
point accuracy but breaks the route into pieces.

Full results, the comparison of all eight models and the tuning runs are in
[`docs/model_evaluation.md`](docs/model_evaluation.md).

## How it fits together

```
GPS track + road network for one trip
  -> clean the track, project to metres (UTM)
  -> find candidate road segments near each point     src/map_matching/candidates.py
  -> describe each (point, road) pair with 19 features src/data/features.py
  -> classifier scores every candidate                src/models/inference.py
  -> HMM + Viterbi picks the most likely route         src/map_matching/
  -> API returns the route and scores it              src/api/
  -> UI draws it on a map                             frontend/
```

On evaluation day this runs on two laptops, as the guidelines ask. Laptop 1
runs the API, which holds the model, the road networks and the reference
routes. Laptop 2 runs the UI. The evaluator picks a trip in the UI, the API
reconstructs it and scores it against the reference route, and only the score
goes back. The reference route itself never leaves the API laptop.

## Repository layout

```
configs/pipeline.json   every tunable setting (search radius, HMM parameters, CV folds, seed)
src/data/               loading, cleaning, validation, EDA, labels, features
src/features/           Person 1's preprocessing scripts (unchanged)
src/map_matching/       candidates, HMM, Viterbi, route reconstruction, route metrics
src/models/             model zoo, training and cross-validation, Optuna tuning, inference
src/csv_pipeline/       experiments on Person 1's preprocessed CSVs
src/experiments/        classifier vs HMM comparison and decoder tuning
src/visualization/      plots and Folium maps
src/api/                FastAPI backend
frontend/               evaluator UI (plain HTML and JS, Leaflet included)
notebooks/              EDA, training and evaluation walkthroughs
scripts/                one-off entry points (EDA, full pipeline, UI server, evaluator check)
tests/                  unit, API and end-to-end tests
models/final/           the model and HMM settings the API loads
results/                every table, plot, map and log the experiments produce
docs/                   methodology, evaluation, API reference and data notes
```

## Getting started

You need Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The dataset isn't in the repository. Unzip it so each trip sits in its own folder:

```
data/raw/train/<trip_id>/<trip_id>.track  .nodes  .arcs  .route
data/raw/student_test/<trip_id>/...
data/raw/eval_test/<trip_id>/...           only on the API laptop
```

Person 1's preprocessed CSVs, if you want to rerun the CSV experiments, go in `data/person1/`.

## Running the experiments

```bash
python scripts/run_eda.py                          # dataset exploration -> results/eda/
python -m src.models.train                         # train and compare 8 models
python -m src.models.optimize                      # Bayesian tuning of the chosen model -> models/final/
python -m src.experiments.compare_decoders --tune  # tune the HMM, compare it with classifier-only
python -m pytest -q
```

`scripts/run_all.sh` runs all of this in order. On a 12-core laptop the whole
thing takes about half an hour.

The experiments on Person 1's CSVs are kept separate:

```bash
python -m src.csv_pipeline.experiments compare
python -m src.csv_pipeline.experiments optimize
python -m src.csv_pipeline.experiments decoders
```

## Evaluation day

On **Laptop 1** (API):

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

It serves `data/raw/eval_test/` by default. To rehearse on the student-testing
trips instead, set `PATHSNAP_SERVE_PARTITION=student_test`. Keep this terminal on
screen: every request is logged there, and the evaluator will want to see it.
Port 8000 needs to be open in the firewall.

On **Laptop 2** (UI):

```bash
python scripts/serve_ui.py
```

Then open `http://localhost:5173/?api=http://<laptop-1-ip>:8000`. Both laptops
must be on the same network. The UI needs nothing but Python. Without internet
the background map tiles won't load, but the route and GPS points still draw.

The endpoints, with real example responses, are documented in [`docs/api.md`](docs/api.md).
Interactive docs are at `http://<laptop-1-ip>:8000/docs` while the API is running.

Optional settings, all environment variables:

| Variable | Default | What it does |
|---|---|---|
| `PATHSNAP_SERVE_PARTITION` | `eval_test` | which trips the API offers |
| `PATHSNAP_DATA_DIR` | `data/raw` | where the trip folders live |
| `PATHSNAP_MODEL_DIR` | `models/final` | which model the API loads |
| `PATHSNAP_CORS_ORIGINS` | `*` | which UI addresses may call the API |

## Keeping the evaluation honest

* Trips are never split across training and testing. Cross-validation keeps all
  points of a trip in the same fold.
* The data provides an ordered route per trip, not a road per GPS point. We
  derive per-point labels by aligning the route to the points in order. This is
  explained, and checked for stability, in [`docs/methodology.md`](docs/methodology.md).
* Features are built from GPS and road data only. A test checks that changing
  the reference route can't change them.
* Model choice, hyperparameters and HMM settings were all decided on training
  data. The student-testing trips were used only to compare.
* The training code refuses to read the evaluator trips. We ran the pipeline
  on their inputs once, to check it doesn't crash, without opening their
  reference routes or computing any score. See [`docs/evaluator_data.md`](docs/evaluator_data.md).

## Known limitations

* Two training trips, 00000054 and 00000033, have GPS that sits 120–140 m away
  from their reference route. We kept them, so the cross-validation scores
  include them, and we flagged them for Person 1.
* Roads are stored as straight segments between nodes, so curved roads with few
  nodes look further from the GPS than they really are.
* The HMM helps 9 of the 10 test trips. On trip 00000007 it costs 5 points of
  accuracy, which we haven't fully explained yet.
* Route continuity and the sequence scores are our own definitions, because the
  guidelines don't define them precisely.

## Further reading

| Document | What's in it |
|---|---|
| [`docs/model_evaluation.md`](docs/model_evaluation.md) | all results: model comparison, tuning, HMM vs classifier |
| [`docs/methodology.md`](docs/methodology.md) | why each design decision was made |
| [`docs/api.md`](docs/api.md) | API reference |
| [`docs/person1_integration.md`](docs/person1_integration.md) | how Person 1's data and scripts fit in, and what we found in them |
| [`docs/csv_data_report.md`](docs/csv_data_report.md) | checks on Person 1's CSVs |
| [`docs/evaluator_data.md`](docs/evaluator_data.md) | how the evaluator data was handled |
| [`docs/requirements_checklist.md`](docs/requirements_checklist.md) | the official requirements and where each is met |

## Team

| Role | Who | Main areas |
|---|---|---|
| Person 1: data processing | Anwesha Manik | dataset exploration, candidate generation, feature CSVs (`src/features/`) |
| Person 2: machine learning | Adarsh D| labels, features, model training and tuning, evaluation |
| Person 3: map matching and system | Bhavish Arun | HMM and Viterbi, API, UI, maps |

We used an AI coding assistant while building this. Every result here comes
from code in this repository that we ran ourselves, and we can explain all of it.
