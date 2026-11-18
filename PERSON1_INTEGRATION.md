# Person 1 ↔ Person 2/3 integration

Updated 2026-11-18 after syncing GitHub `origin/main` @ `47d6dac` and inspecting
`preprocessing_code.zip` and `raw_dataset.zip` (both in `Downloads/ML/`, unmodified).

## What Person 1 delivered

| Item | Where | Status |
|---|---|---|
| 8 preprocessing / EDA / visualisation scripts | `src/data/`, `src/features/` (GitHub; zip copy identical) | ✅ all run unchanged, see `PERSON1_SCRIPTS_INVENTORY.md` |
| `PART_1_REPORT.md` | repo root | ✅ |
| EDA outputs | `outputs/dataset_inventory.csv`, `outputs/timestamp_analysis.csv` | ✅ reproduced byte-identically |
| Trip-1 visualisations | `visualizations/` | ✅ |
| Raw dataset (80 train + 10 student-test trips) | `raw_dataset.zip` → extracted to `data/raw/{train,student_test}/<id>/<id>.{track,nodes,arcs,route}` (git-ignored) | ✅ 360 files, 94 MB |
| 4 feature/label CSVs | copied to `data/person1/` (git-ignored) | ✅ verified |
| Evaluator trips | — | correctly **not** included |

## Raw file formats (verified against the data)

| File | Columns | Notes |
|---|---|---|
| `.track` | lon, lat, timestamp (s) | whitespace-separated, no header; row number = `gps_index` in the CSVs (1:1 for all 90 trips) |
| `.nodes` | lon, lat | row = node id |
| `.arcs` | from_node_id, to_node_id | row = arc id; **directed** (consecutive route arcs chain to→from in 99.96 %); two-way roads = two arcs |
| `.route` | arc id | ordered reference route; identical to `true_arc_id` in `*_labels.csv` |

Arc and node ids are **per trip**: each trip has its own network extract.

## How the pieces connect

```
data/raw/<partition>/<trip>/*.{track,nodes,arcs,route}
   │
   ├── Person 1: src/features/run_full_pipeline.py ──► data/person1/*_features.csv, *_labels.csv
   │        (candidates: 5 nearest within 50 m via midpoint KD-tree; 15 columns)
   │        └─► Workflow A: src/csv_pipeline/ (route alignment labels, CSV features,
   │                         8 models, Optuna, topology-only HMM) → results/csv_pipeline/
   │
   └── Person 2/3: src/data/io.py → prepare.py (UTM, cleaning, candidates: 8 nearest within
            50 m via full-segment KD-tree, 150 m fallback; route alignment on true geometry)
            └─► Workflow B: src/models/ (8 models, Optuna) + src/map_matching/ (full HMM)
                 → models/final/ → src/api/ → frontend/
```

**Why the API uses Workflow B's candidates and features.** Measured on the same
student-test points:

* candidate miss rate: Person 1's generation **3.21 %**, ours **0.42 %**;
* Person 1's generation is not deterministic across environments (unstable
  tie-breaking between twin arcs; see the inventory).

Person 1's code is kept unchanged, and their CSVs remain a full, documented
experiment track (Workflow A). Both feature sets share the same core quantities
(point-to-arc distance, bearing difference, speed, time gap, arc length).

## Data-quality findings (please review)

1. **Trip 00000054**: the GPS follows Amsterdam's A10 ring, but the median
   distance from points to their reference arcs is **140 m**. The mean offset is
   ~118 m north with a large spread, so it is not a constant shift. Map:
   `results/csv_pipeline/maps/trip_00000054_quality.html`. This matches Person 1's
   own observation (report §9).
2. **Trip 00000033**: median 122 m from the reference route.
3. Together these two trips cause ~74 % of the CSV training misses. They are
   **kept**, and a sensitivity metric excluding them is reported separately.
4. **Speeds > 50 m/s**: 1,530 training points, 80 % in trips 00000032 and
   00000001, in runs rather than single jumps. They are kept and flagged.
5. The CSV `direction_difference` uses a single-step bearing (noisy at 1 Hz).
   Our raw pipeline uses a ±2-point window.

## Remaining requests to Person 1

* Is there an explanation for trips 54 and 33 (extraction or projection issue)?
* Do you agree with the stable-sort fix for twin arcs? It would make the CSVs reproducible.
* Please confirm the evaluator input format. Our assumption (MT §6): the
  evaluator trips arrive as the same raw files on the API laptop.
