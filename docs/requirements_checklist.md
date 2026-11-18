# PathSnap — Extracted Project Requirements

Source documents (in `../PathSnap_student_materials/`):

| Code | Document |
|------|----------|
| MG   | Main Guidelines.pdf |
| DE   | Dataset Exploration.pdf |
| MT   | Models and Training Guidelines.pdf |
| DL   | Deliverables.pdf |
| AP   | Assessment and Plagiarism Policy.pdf |

The guidelines are the source of truth. Where they are silent or ambiguous, the
ambiguity is listed in the final section rather than silently resolved.

---

## 1. Problem statement

| Requirement | Source | How it is satisfied |
|---|---|---|
| Reconstruct realistic vehicle trajectories by matching noisy GPS observations to a road network. | MG §1 | `src/map_matching/` — candidate generation → ML candidate scoring → HMM/Viterbi decoding. |
| End-to-end: raw data → processing → models → route reconstruction → deployed system. | MG §1, §4 | Pipeline in `src/`, API in `src/api/`, UI in `frontend/`. |
| Two-laptop deployment: Laptop 1 hosts classifier + HMM/Viterbi + road network + server-side evaluation; Laptop 2 hosts the UI. | MG §3, MT §6 | API binds `0.0.0.0` (configurable); UI is a static site with a configurable API base URL. |

## 2. Dataset

| Requirement | Source | How it is satisfied |
|---|---|---|
| Data organised by trip; each trip has GPS + road-network files. | DE §2 | `src/data/io.py` discovers trips by `.track` file and finds sibling files. |
| `.track`: 3 whitespace columns, no header: **longitude, latitude, timestamp (s)**. ~97 % of steps are 1 s. | DE §3 | `load_track()` reads exactly these columns. |
| `.nodes`: 2 columns, no header: **longitude, latitude**; 0-based row index is the node id. | DE §3 | `load_nodes()`. |
| `.arcs`: 2 integer columns: **from_node_id, to_node_id**; 0-based row index is the arc_id. | DE §3 | `load_arcs()`. |
| `.route`: 1 integer column: chronological sequence of ground-truth **arc_id** values, "where supplied". | DE §3 | `load_route()`; route is optional at load time. |
| A GPS row does **not** correspond to a road-network row. | DE §3 | Per-observation reference labels are *derived* by aligning the route to the track (`src/data/labels.py`). |
| Verify naming, row counts, dtypes per trip; do not assume all trips identical. | DE §2 | `src/data/validation.py`, `scripts/run_eda.py`, `notebooks/01_data_exploration.ipynb`. |
| EDA checklist: trips per partition, obs per trip, file availability & sizes, dtypes, missing/invalid GPS, timestamp ordering & gaps, geographic extent, #nodes/#arcs, connectivity, isolated nodes, components, anomalies, route structure. | DE §4 | `src/data/exploration.py` computes every item; outputs to `results/eda/`. |
| Lon/lat are angular; do not use Euclidean degrees; justify projection/geodesic method; check coordinate order and plausibility. | DE §5 | `src/geo.py` projects each trip to its local UTM zone (metres) via `pyproj`; plausibility checks in validation. |
| Reference routes are not error-free ground truth; document limitations. | DE §6 | Alignment residuals reported; limitations in `docs/` and README. |
| Partitions: 80 training / 10 student-testing / 10 evaluator-testing trips. | DE §7 | Configurable partition dirs: `data/raw/train`, `data/raw/student_test`, `data/raw/eval_test`. |
| No trip split across partitions. | DE §7 | Splits are by trip; CV uses `GroupKFold` on `trip_id`. |
| Evaluator data never used for training, feature selection, algorithm/hyperparameter/threshold choice. | DE §7, MT §7, AP §7 | Training/optimisation code can only read `train` and `student_test`; the `eval_test` path is read only by the API service. A test checks this. |
| Fit learned preprocessing on training data only. | DE §7 | Scaling lives inside sklearn `Pipeline`s fitted per CV fold / on train only. |
| Keep evaluator ground truth separate from prediction pipeline. | DE §7, MT §6 | `TripInputs` (track+network) is what the predictor receives; the route is held in a separate object used only by the scorer. |
| Exploration outputs: inventory, summary stats, plots, Folium maps, network tables, data-quality log. | DE §8 | `scripts/run_eda.py` → `results/eda/*.csv`, `*.png`, `*.html`. |

## 3. Models and training

| Requirement | Source | How it is satisfied |
|---|---|---|
| Train and evaluate **≥ 5 distinct models** from multiple families. | MT §2, §4, DL §3, AP §3 | `src/models/registry.py`: Logistic Regression (linear), Gaussian NB (probabilistic), KNN (instance), Decision Tree, Random Forest, Extra Trees (ensemble), HistGradientBoosting (boosting), MLP (neural). XGBoost/LightGBM added automatically if importable. |
| If an approach does not produce candidate probabilities, explain adaptation. Clustering ids are not labels. | MT §3 | All chosen models expose `predict_proba`. No clustering used. |
| Record config of each model; use 80 training trips for fitting + CV; 10 student-testing trips for comparison. | MT §4 | `results/models/<name>/config.json`, `cv_folds.csv`, `student_test_metrics.json`. |
| Consistent preprocessing. | MT §4 | One feature builder (`src/data/features.py`) used by training, evaluation, and the API. |
| Do not select by training accuracy. | MT §4, §6 | Selection uses CV and student-test metrics only. |
| Record training & inference time; keep logs, seeds, configs. | MT §4 | Timings stored per model; seed in `configs/pipeline.json`; logs in `results/logs/`. |
| Metrics: Accuracy, Precision, Recall, F1, Confusion matrix, Log loss, **Observation-level Top-1**, **Candidate Miss Rate**, **Route Continuity**, **Sequence-level performance**, training/inference time. | MT §5 | `src/models/evaluate.py` (candidate-level) and `src/map_matching/metrics.py` (observation/route-level). |
| Explain primary metric used for selection. | MT §5 | Primary metric = student-test / CV **observation-level Top-1 accuracy** (see `docs/METHODOLOGY.md`). |
| Comparison table: name, family, key config, CV Macro F1, Top-1, route continuity, student-test Macro F1, log loss, inference time. | MT §6, DL §3.1 | `results/model_comparison.csv` with exactly these columns (+ extras). |
| Bayesian HPO on chosen model: search space, objective, group-aware CV on 80 trips, trial log, best params, refit, evaluate on student-test, compare tuned vs untuned, report cost. | MT §7, DL §4 | `src/models/optimize.py` (Optuna TPE + MedianPruner); `results/optuna/`. |
| HMM: hidden-state representation, preserve candidate→segment correspondence, emission & transition choices, Viterbi recurrence, missing candidates, invalid transitions, GPS gaps, a predicted segment per observation, classifier-only vs classifier+HMM comparison, pointwise + sequence metrics, continuity analysis. | MT §8–9, DL §5 | `src/map_matching/hmm.py`, `viterbi.py`, `reconstruction.py`; comparison script `src/experiments/compare_decoders.py`. |
| Optimise decoder parameters using training and student-testing data only. | MT §9 | `compare_decoders.py --tune` grid search reads only `train`/`student_test`. |

## 4. System (API / UI / evaluation)

| Requirement | Source | How it is satisfied |
|---|---|---|
| FastAPI backend that processes evaluation requests and loads the model. | MT §5, DL §1 | `src/api/` — model loaded once at startup. |
| UI displays the **ten reserved evaluation trips**; evaluator selects one; UI sends **trip ID** to API. | MT §6 | `GET /trips` lists trips; `POST /reconstruct {trip_id}`. |
| API runs the full prediction pipeline and **computes accuracy using its internal reference data**; UI displays predicted route and results. | MT §6 | Service computes metrics server-side and returns only metrics + predicted route. |
| Must work for all ten trips, not one demo. | MT §6 | No per-trip special casing; test iterates every served trip. |
| Ground truth stays server-side; never exposed to model or UI. | MT §6, AP §7 | Response schema has no reference-route fields; tested. |
| No hardcoded outputs, no bypassing API; show API logs on Laptop 1. | MT §6, AP §7 | Every request logged (trip id, timings, metrics) to console and `results/logs/api.log`. |

## 5. Deliverables

| Requirement | Source | How it is satisfied |
|---|---|---|
| Public GitHub repo with data processing, feature engineering, training for every model, evaluation & plots & confusion matrices, Bayesian optimisation + results, HMM/Viterbi, API, UI, deployment config, Folium code, supporting CSVs/logs. | DL §1 | Folder layout in README. |
| Include saved model weights + configs (subject to size). | DL §1 | `models/final/`. |
| README with install, execution, deployment. | DL §1 | `README.md`. |
| No secrets committed. | DL §1 | `.gitignore`; configuration via environment variables documented in the README. |
| Technical report PDF (sections 2.2–7) with real results, tables, graphs, Folium map screenshot, UI + API-log screenshots. | DL §2–7 | **Written by the team.** Repo produces every table/graph it needs (`results/`). Not auto-generated (AP §5). |
| Plots: CV Macro F1 across models; student-test Top-1 & Macro F1; log loss; confusion matrices; loss curves **only** for models with epoch histories. | DL §3.3 | `src/visualization/plots.py`; loss curves only for MLP (`loss_curve_`) and HistGradientBoosting (`train_score_`/`validation_score_`). |
| Folium maps: GPS points, road network, reference route, predicted route; legend; layer control; sensible zoom. | DL §6 | `src/visualization/folium_maps.py`, `scripts/make_maps.py` (offline/report only — reference route never sent to UI). |

## 6. Academic integrity

| Requirement | Source | How it is satisfied |
|---|---|---|
| No sharing of code/models/configs/results between teams; artifacts compared across teams. | MT §11, AP §4 | — (team responsibility). |
| No fabricated metrics, graphs, results; every graph from real experiments. | DL §3.3, AP §5 | All numbers are written by scripts to `results/`; nothing is hand-entered. Synthetic data exists **only** inside `tests/` for unit tests and is never reported. |
| Disclose AI assistance; every member must be able to explain the code. | DL §8, AP §5–6 | Team responsibility; `docs/METHODOLOGY.md` explains every design decision. |

---

## 7. Ambiguities and recorded assumptions (updated 2026-11-18 with the real data)

1. ~~Dataset not present.~~ **Resolved:** `raw_dataset.zip` (80 train + 10 student-test trips) is extracted to `data/raw/`. Person 1's CSVs are in `data/person1/`. No evaluator trips have been received, which is expected.
2. **Data layout, resolved:** `<partition>/<trip_id>/<trip_id>.{track,nodes,arcs,route}`, exactly what the loader expects.
3. **Per-observation labels** are still not supplied. They are derived by monotone route alignment. The labels are robust: ≥ 98.4 % agreement under extreme cost assumptions (`docs/CSV_DATA_REPORT.md` §3).
4. **Arc direction, resolved:** arcs are directed. Consecutive route arcs chain to→from in 99.96 % of cases, and 95 % of arcs have a reverse twin (`results/eda/validation.csv`). `network.directed = true` is correct.
5. **"Macro F1"** = mean of the class-0 and class-1 F1 on candidate pairs (unchanged).
6. **Route continuity** = share of consecutive distinct predicted arcs that are directly connected (unchanged; our definition).
7. **Sequence-level performance** = arc-set precision/recall/F1, length-weighted precision/recall and edit similarity vs `.route` (unchanged; our definition).
8. **Evaluator `.route` availability:** still assumed (MT §6 says the API scores against "internal reference data"). The API returns `metrics = null` if a route is absent.
9. **Person 1 integration, resolved:** all 8 scripts are present and run unchanged (the script inventory in `docs/person1_integration.md`). The production pipeline uses our candidate generation because of measured recall and reproducibility (`docs/person1_integration.md`).
10. **Known data anomalies (kept, documented):** trips 00000054 and 00000033 (GPS ≈ 120–140 m from the reference route); runs of > 50 m/s speeds in trips 00000032 and 00000001; a 153,064 s maximum time gap.

## 8. Implementation status

| Area | Status |
|---|---|
| Raw-data loading, validation, EDA (DE §4, §8) | on real data, `results/eda/` |
| ≥ 5 models, GroupKFold CV, metrics (MT §2–6) | Workflow A (CSV) complete; Workflow B (raw) running |
| Bayesian optimisation (MT §7) | code; running on both workflows |
| HMM + Viterbi, classifier-only vs HMM (MT §8–9) | code; raw decoder tuning pending the final model |
| Folium maps (DL §6) | EDA + data-quality maps; prediction maps produced by `compare_decoders` |
| FastAPI + UI (MT §6) | code + tests; real-data rehearsal pending the final model |
| Technical report PDF (DL §2–7) | Pending: team-written, from `results/` |
