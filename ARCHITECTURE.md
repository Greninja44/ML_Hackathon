# PathSnap architecture

## Deployment (Main Guidelines §3, MT §6)

```
 Laptop 2 (UI)                               Laptop 1 (model + API)
 ┌──────────────────────────┐   HTTP/JSON    ┌──────────────────────────────────────────────┐
 │ frontend/ (static HTML,  │ ─────────────► │ FastAPI  src/api/                            │
 │ Leaflet vendored)        │  GET  /trips   │  ├─ loads models/final/ once at start-up     │
 │ • lists served trips     │  POST /recon-  │  ├─ reads data/raw/eval_test/ (route stays   │
 │ • sends trip_id          │       struct   │  │  server-side; only metrics leave)         │
 │ • draws returned route   │ ◄───────────── │  └─ logs every request (show to evaluator)   │
 └──────────────────────────┘  route + metrics└──────────────────────────────────────────────┘
```

## Prediction pipeline (Workflow B, raw data, used by the API)

```
TripInputs (track + nodes + arcs; never the route)
  │ src/data/prepare.py
  ├─ build_network ─► RoadNetwork (UTM projection, KD-tree over densified arcs,
  │                   adjacency, bounded Dijkstra)            src/data/network.py, src/geo.py
  ├─ clean_track   ─► CleanTrack (drop invalid, sort, de-dup, flag speed outliers,
  │                   ±2-point heading)                       src/data/preprocess.py
  ├─ generate_candidates ─► CandidateSet (≤8 arcs within 50 m, 150 m fallback)
  │                                                           src/map_matching/candidates.py
  └─ build_features ─► one row per (point, candidate), 19 features
                                                              src/data/features.py
  │ src/models/inference.py
  CandidateScorer.predict_candidates(features) ─► P(candidate = reference arc)
  │ src/map_matching/hmm.py
  emission = log p  (or hybrid with Gaussian distance; tuned)
  transition = −|network driving distance − GPS distance| / β·√Δt (bounded Dijkstra)
  │ src/map_matching/viterbi.py
  log-space Viterbi with segment breaks for missing candidates / impossible transitions
  │ src/map_matching/reconstruction.py
  ReconstructionResult: arc per observation, connected route (gaps bridged by
  bounded shortest paths), continuity, timings
  │ src/api/service.py
  server-side metrics vs reference route (alignment → Top-1, continuity, sequence metrics)
```

## Training pipelines

| | Workflow A (CSV) | Workflow B (raw) |
|---|---|---|
| Input | `data/person1/*.csv` (Person 1's `run_full_pipeline.py`) | `data/raw/{train,student_test}` |
| Labels | `src/csv_pipeline/labels.py` (monotone route alignment using CSV distances + lower bounds) | `src/data/labels.py` (same alignment on exact geometry) |
| Features | `src/csv_pipeline/features.py` (16) | `src/data/features.py` (19) |
| Models / CV / Optuna | shared: `src/models/registry.py`, `train.cross_validate`, `optimize.search_space` | same |
| Decoder | topology-only HMM `src/csv_pipeline/hmm.py` | full HMM `src/map_matching/` |
| CLI | `python -m src.csv_pipeline.experiments {compare,optimize,decoders}` | `python -m src.models.train`, `python -m src.models.optimize`, `python -m src.experiments.compare_decoders --tune` |
| Results | `results/csv_pipeline/` | `results/` |
| Used by API | no | **yes** (`models/final/`) |

## Module dependency rules

* `src/data` ← `src/map_matching` ← `src/models` ← `src/experiments` / `src/api` (no cycles).
* Features are computed in exactly one place per workflow and reused by training,
  evaluation and the API.
* `iter_partition` refuses `eval_test` unless `allow_eval=True`, which only `src/api/service.py` sets.
* Person 1's scripts (`src/features/*`, `src/data/{explore_dataset,analyze_timestamps,visualize_*}.py`)
  are kept unchanged and are not imported by the API.
