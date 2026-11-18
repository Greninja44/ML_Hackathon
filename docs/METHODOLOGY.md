# PathSnap — Methodology and design decisions

This document explains *why* the pipeline is built the way it is, so every team
member can defend it in the viva. It contains **no results**: every number for
the report must come from the files the scripts write to `results/`.

## 1. Data model

| File | Content (Dataset Exploration §3) | Code |
|---|---|---|
| `.track` | lon, lat, t [s] — ~1 Hz | `src/data/io.py::load_track` |
| `.nodes` | lon, lat; row = node id | `load_nodes` |
| `.arcs` | from_node, to_node; row = arc id | `load_arcs` |
| `.route` | ordered arc ids (reference) | `load_route` |

Relationships: GPS rows are **not** aligned with any network row. Arcs are
straight segments between two nodes. Each trip has its own node/arc tables, so
arc ids are only meaningful within a trip.

### Coordinates
Each trip is projected to the UTM zone of its median GPS position
(`src/geo.py`). Inside one trip, UTM distortion is < 0.1 %, so planar
distances in metres are accurate and cheap. Euclidean distance in degrees is
never used.

### Cleaning (`src/data/preprocess.py`)
Non-finite / out-of-range rows are dropped; rows are stably time-sorted if
needed; exact duplicates are removed; implausible-speed points are *flagged*
(kept, and the flag is a feature). The original `.track` row index is kept so
every prediction maps back to the raw file.

### Per-observation labels (`src/data/labels.py`)
The `.route` file gives the arc sequence, not which GPS point is on which arc.
We assign every observation an index into the route such that the index never
decreases over time and the sum of point-to-arc distances is minimal (exact DP,
O(N·L)). Residuals are reported — the reference route is treated as
reference information, not error-free truth (DE §6).

**Viva point:** labels are derived *only* for training/evaluation. The
prediction pipeline (`TripInputs`) never contains the route.

## 2. Candidate generation (`src/map_matching/candidates.py`)
All arcs within `radius_m` (default 50 m) of a GPS point, closest
`max_candidates` (8) kept, using a KD-tree over arc points densified every
10 m. If none: widen once to `fallback_radius_m` (150 m). Still none → the
observation is unmatched and the decoder starts a new segment.

**Candidate miss rate** = share of observations whose reference arc is not a
candidate. It is an upper bound on achievable Top-1 accuracy, so radius and K
should be chosen by looking at miss rate vs. candidate count on training trips.

## 3. ML formulation
*Binary classification of (observation, candidate arc) pairs*: is this
candidate the reference arc? This turns a variable-sized, per-trip label space
(arc ids differ per trip) into a fixed feature space that generalises across
trips, and produces a probability per candidate — exactly what the HMM needs
as an emission score.

Features (`src/data/features.py`) — all computed from GPS + network only:
distance, distance rank and margin, number of candidates, heading difference
to the *directed* arc (crucial when two-way roads are stored as two arcs),
axis-only heading difference, speed, arc length, projection position, distance
to junction, presence of a reverse twin arc, neighbouring-point distances to
the arc (±3 observations), link to the previous point's nearest arc, out-degree,
time gap, outlier flag.

### Metrics
* Pair level (threshold 0.5): accuracy, precision, recall, F1, **macro F1**
  (mean of class-0 and class-1 F1), log loss, ROC-AUC, confusion matrix.
* Observation level: **Top-1 accuracy** (highest-probability candidate =
  reference arc; observations without candidates count as wrong), candidate
  miss rate.
* Route level: **route continuity** (share of arc changes that follow a network
  connection), **sequence metrics** (set precision/recall/F1 of arcs,
  length-weighted precision/recall, normalised edit similarity vs. `.route`).

**Primary selection metric: CV observation-level Top-1 accuracy.** Pair-level
accuracy is dominated by the many easy negatives and depends on an arbitrary
0.5 threshold, while map matching only needs the right candidate *ranked
first* for each point. Top-1 measures exactly that, and CV over 80 trips is
more stable than the 10 student-testing trips. Log loss is the tie-breaker
because the HMM consumes probabilities.

### Validation design
`GroupKFold` with `groups = trip_id` (5 folds): all points of a trip are in the
same fold, so consecutive near-identical GPS points cannot leak between train
and validation. Scaling lives inside each sklearn `Pipeline`, so it is fitted
on training folds only. Student-testing trips are only used for comparison;
evaluator trips are blocked in code (`iter_partition` raises).

### Models (`src/models/registry.py`)
Logistic Regression (linear), Gaussian NB (probabilistic), KNN (instance-based,
150k-row training subsample), Decision Tree, Random Forest, Extra Trees,
HistGradientBoosting (boosting), MLP (neural). XGBoost/LightGBM are added
automatically when the OpenMP runtime is available. All produce probabilities.
Loss curves are drawn **only** for MLP (`loss_curve_`) and HistGradientBoosting
(`train_score_`/`validation_score_`), the models that actually record histories.

### Bayesian optimisation (`src/models/optimize.py`)
Optuna TPE sampler (seeded), MedianPruner on the running fold mean, objective =
mean CV Top-1 on the 80 training trips. Search spaces are model-specific and
listed in `search_space()`. The tuned model is refitted and evaluated with the
same code as the untuned one (`results/tuned_vs_untuned.csv`). The final model is
the tuned one only if its CV Top-1 is at least the untuned CV Top-1.

## 4. HMM and Viterbi

* **Hidden state** at step t: one candidate arc of observation t plus the
  point's projection offset on that arc. The candidate list *is* the state
  list, so state → arc id correspondence is exact.
* **Emissions** (`hmm.py::emission_scores`): `log p` from the classifier
  (`ml`), a Gaussian on perpendicular distance (`gaussian`, no ML), or a
  weighted mix (`hybrid`). Justification: P(z|s) ∝ P(s|z)/P(s); with a uniform
  prior over an observation's candidates the classifier posterior is a scaled
  likelihood, and the per-step constant does not change the Viterbi argmax.
  Scores are floored at `min_log_prob` so a 0 probability never makes a path
  impossible on its own.
* **Transitions** (`TransitionModel`): driving distance between the two
  projected points (bounded Dijkstra on the directed arc graph) compared with
  the straight-line distance between the GPS fixes:
  `log A = −|d_route − d_gc| / β_eff`, `β_eff = β·sqrt(max(1, Δt))` (Newson &
  Krumm 2009). Unreachable within `4·d_gc + 200 m` → `−inf`. Moving backwards on
  the same directed arc (> 5 m) is not allowed. **Classifier probabilities are
  never used as transitions.**
* **Viterbi** (`viterbi.py`): log-space recurrence
  `δ_t(j) = e_t(j) + max_i[δ_{t−1}(i) + A_t(i,j)]`, so no underflow on long trips.
* **Missing candidates / disconnected segments / gaps:** an observation with no
  candidates, or a step where every transition is `−inf`, closes the current
  segment (back-tracked) and restarts decoding at the next decodable
  observation. Large time gaps loosen β and the Dijkstra bound rather than
  forcing a break.
* **Route output:** consecutive duplicate arcs are collapsed; gaps between
  consecutive distinct arcs are bridged with the shortest network path only if
  one exists within a bounded distance (never a fabricated long detour).
* **Decoder tuning** (`src/experiments/compare_decoders.py --tune`): grid over
  emission type, ML weight, σ, β using **out-of-fold** probabilities on 20
  training trips (so probabilities are not over-confident), objective = pooled
  Top-1. The comparison is then reported on student-testing trips:
  nearest arc, distance-only HMM, classifier-only, classifier + HMM.

## 5. API and UI
* `GET /health`, `GET /trips`, `GET /trips/{id}/network`, `POST /reconstruct`
  (`{trip_id, mode: "hmm"|"classifier", compare}`); Swagger at `/docs`.
* The service loads the model and tuned HMM config once at start-up, runs the
  full pipeline per request on the trip's GPS + network, and computes metrics
  against the internal `.route` file, which never leaves the server. A test
  asserts no reference-route fields appear in responses.
* Every request is logged (trip, mode, model, obs, matches, Top-1, time) to the
  console and `results/logs/api.log` — show this terminal to the evaluator.
* The UI is a static page (Leaflet, vendored locally) that lists the served
  trips, sends the selected trip id, and draws only what the API returns.

## 6. Known limitations
* Labels are derived by alignment; where the reference route itself is wrong
  or GPS is very noisy, labels inherit that error.
* Arcs are straight segments; curved roads represented by few nodes will have
  larger point-to-arc distances.
* Directed-network assumption must be confirmed on the real data
  (`share_arcs_with_reverse` in `results/eda/validation.csv`). Set
  `network.directed=false` if arcs are undirected.
* Route continuity and sequence metrics are our own definitions (the guidelines
  do not define them formally) — state this in the report.
* HMM decode time grows with candidates²; K=8 keeps it bounded.
