# Person 1 CSVs — verification, target derivation and data-quality findings

Partitions: **train** (80 trips) and **student_test** (10 trips). No evaluator data.
Every number below was computed from the files by code in `src/csv_pipeline/`
(outputs in `results/csv_pipeline/`).

## 1. Files

| File | Rows | SHA-256 |
|---|---|---|
| train_features.csv | 804,416 | `047f1ddb…a9f123b6` |
| train_labels.csv | 31,336 | `9d926427…f51d9eb` |
| student_test_features.csv | 79,669 | `067f3be2…58e2de959` |
| student_test_labels.csv | 3,477 | `8a23c31b…4f660985` |

## 2. What a row is (verified)

* `*_features.csv`: one row per **(GPS point, candidate arc) pair**. Points are
  `(trip_id, gps_index)`; `gps_index` is 0..n−1 with no gaps. The per-point
  columns (`timestamp, gps_distance_m, gps_bearing, time_gap_sec,
  gps_speed_mps`) are identical across a point's candidates.
* Candidates per point: 1–5, sorted by `distance_m`. Train: 5 candidates for
  55 % of points, 1 for 10 %.
* `arc_id` is a **per-trip** id: the same id has different nodes in different
  trips, but nodes and length are consistent within a trip.
* `*_labels.csv`: the reference route as an ordered arc sequence
  (`route_position` 0..L−1). Consecutive route arcs chain `to_node → from_node`
  in **99.96 %** (train) / **99.97 %** (student) of cases where both are known,
  so arcs are **directed**. About 75 % of candidate arcs have their reverse
  twin among the trip's candidates (two-way roads = two arcs).
* No missing values, no duplicate (trip, point, arc) rows, trip sets identical
  between feature and label files, no trip in both partitions.

## 3. Supervised target

**Rejected:** "candidate appears anywhere in the route". 53 % (train) / 60 %
(student) of points have more than one on-route candidate (junctions, loops),
so it is not a per-point label.

**Used:** a monotone alignment of points to route positions
(`src/csv_pipeline/labels.py`). Each point gets the route position minimising the
total point-to-arc distance, subject to never moving backwards along the route.
Distances to route arcs that are not among a point's candidates are replaced by
a **lower bound**: the point's farthest candidate distance + 5 m when it has 5
candidates, otherwise max(farthest, 50 m) + 5 m.

**Robustness** (label agreement with the default when the assumptions change):

| Variant | Train | Student-test |
|---|---|---|
| margin 0 m | 98.42 % | 98.74 % |
| margin 20 m | 99.90 % | 99.96 % |
| unknown radius 30 m | 99.998 % | 100 % |
| unknown radius 100 m | 99.74 % | 99.99 % |

The target is therefore well determined by the data. Remaining uncertainty is
≤ 1.6 % of labels under the most extreme assumption. The median distance from a
point to its label arc is 4.9 m (train) and 5.3 m (student).

**Candidate misses:** points whose aligned arc is not among their candidates
stay in every evaluation and count as wrong. Top-1 accuracy cannot exceed
1 − miss rate.

| | Train | Student-test |
|---|---|---|
| Miss rate (aligned label not a candidate) | 13.73 % | 3.21 % |
| Miss rate excluding trips 00000054 and 00000033 | ≈ 4.1 % | — |

## 4. Why train misses 13.7 % but student-test 3.2 %

* Trip **00000054** (19,313 points, 9 % of all training points): 88.9 % missed.
  The median distance to the *nearest* candidate is **84 m**, against 4–7 m in
  every other trip. It accounts for about 60 % of all training misses.
* Trip **00000033** (5,844 points): 68.2 % missed, about 14 % of training misses.
* Misses form long runs: 79.5 % of missed training points sit in runs of ≥ 30
  consecutive points (longest 5,243). They are systematic, not random noise.
  Student-test misses are short (longest run 25).

This is consistent with a mismatch between those trips' GPS and their network
extract or route, which can only be checked with the raw files. Both trips are
**kept**: they stay in CV, and in the primary metric they count against the
model. A sensitivity metric that excludes trips with > 50 % misses is reported
separately and never used for selection.

## 5. Speeds and GPS jumps

* 1,530 training points (0.73 %) have speed > 50 m/s (180 km/h), almost all
  with a 1 s gap. They are not isolated: 972 are followed by another spike, and
  **trips 00000032 (892) and 00000001 (340)** hold 80 % of them. This points
  to a timestamp or sampling problem in those trips rather than single GPS
  jumps. Student-test has only 8 such points.
* These points are kept and flagged (`speed_spike` feature). The guidelines
  do not allow silently altering data, and the cause is unknown until the raw
  files are inspected.

## 6. Heading

`gps_bearing` comes from the previous point only. For the true arc (where a
reverse twin exists), the movement agrees with the arc's direction
(difference < 90°) in only 82 % of cases, with a median difference of 29°. A heading
smoothed over several points would separate two-way twins better. That is a
candidate feature-engineering experiment (it can be computed from the CSV by a
circular mean of neighbouring `gps_bearing`).

## 7. What can and cannot be done with the CSVs

| Can (done) | Cannot (needs raw files) |
|---|---|
| Per-point target via route alignment | Folium maps, any coordinates or geometry |
| ≥ 5-model comparison, GroupKFold CV, log loss, Macro F1, Top-1, miss rate | Our own candidate generation (radius/K are fixed by Person 1) |
| Optuna on the selected model | Distance-based HMM transitions (network driving distance) |
| Topology-only HMM (node connectivity) + classifier vs HMM comparison | Bridging gaps with shortest paths; route geometry for the UI |
| Route continuity (to→from) and sequence metrics vs route | Running on evaluator trips (they arrive as raw files) |

## 8. Experiment results

See section 8 of this report (filled from `results/csv_pipeline/` after the runs).
