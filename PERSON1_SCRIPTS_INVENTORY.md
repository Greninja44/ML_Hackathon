# Person 1 scripts — inventory

Checked 2026-11-18 against GitHub `origin/main` @ `47d6dac` and `preprocessing_code.zip`
(9,081 bytes, in `Downloads/ML/`, left unmodified).

**All eight scripts exist.** The copies in the zip are identical to the GitHub
copies apart from Windows (CRLF) line endings, so the GitHub versions in `src/`
are authoritative. They were **not modified**.

Every script was run unchanged in an isolated scratch directory with `data/raw`
linked in, so the committed `outputs/` and `visualizations/` were not overwritten.
Python 3.11, numpy 2.4.6, pandas 3.1.1, pyproj 3.7.2.

| # | Script | Location | In zip | Runs? | What it does | Reads | Writes |
|---|---|---|---|---|---|---|---|
| 1 | `explore_dataset.py` | `src/data/` | ✓ | ✓ (1 s) | Per-trip inventory: #points, #nodes, #arcs, route length, missing values, duplicates, timestamp range | `data/raw/{train,student_test}/*/*.{track,nodes,arcs,route}` | `outputs/dataset_inventory.csv` — **identical** to committed |
| 2 | `analyze_timestamps.py` | `src/data/` | ✓ | ✓ (<1 s) | Per-trip sampling-interval stats, gaps > 1 s, negative intervals | `*.track` | `outputs/timestamp_analysis.csv` — **identical** to committed |
| 3 | `visualize_trip.py` | `src/data/` | ✓ | ✓ | Matplotlib + Folium plot of trip 00000001 GPS (hard-coded trip) | trip 1 `.track` | `visualizations/trip_00000001.{png,html}` |
| 4 | `visualize_road_network.py` | `src/data/` | ✓ | ✓ | Road network of trip 1 (lon/lat line collection) | trip 1 `.nodes/.arcs` | `visualizations/road_network_00000001.png` |
| 5 | `visualize_map_matching.py` | `src/data/` | ✓ | ✓ | GPS over road network for trip 1 (no matching; overlay only) | trip 1 `.track/.nodes/.arcs` | `visualizations/gps_road_overlay.png` |
| 6 | `candidate_generation.py` | `src/features/` | ✓ | ✓ | Single-trip candidate generation (trip 1); same algorithm as #8 | trip 1 raw | `data/processed/candidates_00000001.csv` — same rows, **different twin order** (see below) |
| 7 | `feature_engineering.py` | `src/features/` | ✓ | ✓ | Single-trip features from #6 output | trip 1 raw + candidates CSV | `data/processed/features_00000001.csv` |
| 8 | `run_full_pipeline.py` | `src/features/` | ✓ | ✓ (~2 min) | **Produces the four CSVs** for all 90 trips: UTM projection, candidate generation, 15-column features, labels file | `data/raw/{train,student_test}/` | `data/processed/{train,student_test}_{features,labels}.csv` |

Imports used: pandas, numpy, scipy (`cKDTree`), pyproj, matplotlib, folium. All are in `requirements.txt`.

## How the CSVs are produced (`run_full_pipeline.py`)

1. Load `.track` (lon, lat, t), `.nodes` (lon, lat) and `.arcs` (from, to). Arc id = row index; node id = row index.
2. Project to the UTM zone of the **mean node position**.
3. Candidates: build a KD-tree over arc **midpoints**, query the **20** nearest
   midpoints, compute exact point-to-segment distances, `np.argsort`, keep up to
   **5** arcs with distance ≤ **50 m**; if none, keep the single nearest of the 20.
4. Per point: `gps_distance_m`, `gps_bearing` from the previous point (0 for the
   first point), `time_gap_sec` = Δtimestamp, `gps_speed_mps` = distance/gap
   (0 when gap = 0).
5. Per arc: `road_length_m`, `road_bearing`; `direction_difference` =
   min(|Δbearing|, 360 − |Δbearing|).
6. Labels: the `.route` file copied as (`route_position`, `true_arc_id`).

`gps_index` is the row number in `.track`. We verified that every trip's
`.track` row count equals its number of CSV points (1 : 1).

## Reproducibility finding (important)

Re-running `run_full_pipeline.py` here gives the same number of rows (804,416 /
79,669) and **identical labels files**. The feature files differ:

* 314,466 train rows have a different `arc_id` at the same position. 96 % of them
  are the **reverse twin** at exactly the same distance.
* The candidate **set** differs for 54,278 train points (26 %) and 4,849
  student-test points.

Cause: two-way roads are stored as two arcs with identical geometry, so they tie
on distance. `np.argsort` (default quicksort, not stable) breaks ties differently
across numpy versions or platforms (Person 1 used Python 3.13). With
`MAX_CANDIDATES = 5`, an odd number, the 5th slot often holds an arbitrary
direction of a two-way road.

Measured effect: in **22.9 %** of student-test candidate misses and 4.1 % of
training misses, the true arc's reverse twin was kept and the true arc dropped.

Suggested fix for Person 1 (not applied — their code is unchanged): use
`np.argsort(distances, kind="stable")` and keep both twins when one is selected
(or use an even K / a radius-only rule).

## Other limitations (Person 1 also notes some of these)

* Midpoint KD-tree shortlist (k = 20) can miss long arcs whose midpoint is far
  from the point.
* The projection is centred on the mean node position. Fine at city scale.
* The bearing comes from one step, so it is noisy at 1 Hz. The first point's bearing is 0.
* The trip-specific scripts (#3–#7) hard-code trip `00000001`.
