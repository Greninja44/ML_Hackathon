# Evaluator data audit

Date: 2026-11-18. Archive: `Downloads/ML/evaluator_test.zip`. It sits outside the repo and was not modified.
SHA-256 `a425b35e6e6f8d3a919690c5df3a9a7df2f7c2de7343a01c50ccd6ecb45178a7`, checked again after extraction.

## 1. Archive contents (listing only)

* 2,431,095 bytes compressed, 7,643,166 uncompressed. 51 entries: 1 top folder, 10 trip folders, 40 files.
* Layout: `evaluator_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}`. This is the same
  per-trip structure as `train/` and `student_test/`, but the top-level folder has a different name.
* Extensions: 10 × `.track`, 10 × `.nodes`, 10 × `.arcs`, 10 × `.route`. No CSV, JSON or other files.
* Trips (10): 00000015, 00000017, 00000023, 00000030, 00000049, 00000059, 00000067,
  00000073, 00000087, 00000094. **None** overlap the training or student-test trips.
* The archive holds **both inputs** (`.track/.nodes/.arcs`) **and reference labels** (`.route`).
* No absolute or `../` paths. The CRC check (`unzip -t`) passed.

## 2. Permitted use (official guidelines)

| Rule | Source | How we comply |
|---|---|---|
| Laptop 1 hosts the model, HMM, road network **and the server-side evaluation process** | Main Guidelines §3 | Trips extracted to `data/raw/eval_test/` on the API laptop |
| The UI lists the ten reserved trips; the API runs the full pipeline and **calculates accuracy using its internal reference data** | MT §6 | `GET /trips`, then `POST /reconstruct`, which scores server-side; the route is never in a response |
| Must not be used for training, feature selection, choosing algorithms, hyperparameters or thresholds | DE §7, MT §7 | No training, tuning or selection code can read `eval_test` (`iter_partition` raises; tested) |
| Ground truth never exposed to the prediction model or UI | MT §6, AP §7 | The predictor receives `TripInputs` (no route); responses contain metrics only |
| "Evaluator-testing data must remain untouched during development" | AP §7 | Agreed with the team: deployment plus an **input-only** check. No accuracy was computed and no `.route` was read (§4) |

**Not done, deliberately:** no metrics against evaluator labels, no `/reconstruct`
calls on evaluator trips, and no model, feature or parameter changes based on anything observed here.

## 3. Repository protection

* Extracted to `data/raw/eval_test/<trip_id>/` (the partition folder the API serves by default).
* `.gitignore`: `data/raw/` (Person 1's rule) plus explicit entries `evaluator_test.zip`,
  `evaluator_test/` and `data/raw/eval_test/*`. `git status` shows no evaluator files.

## 4. Inference compatibility: input-only check

`python scripts/check_eval_inputs.py`. In this script the label loader
(`src.data.io.load_route`) is replaced by a function that raises, so a `.route` file
cannot be opened. It runs the **final saved model** (`hist_gradient_boosting_tuned`) with the
saved, tuned HMM config, in both decoder modes. Nothing is retrained or tuned. Output:
`results/eval_input_check.json` (`"labels_read": false, "metrics_computed": false`).

| Stage | Result |
|---|---|
| Raw parsing (`.track`, `.nodes`, `.arcs`) | ✅ 10/10, same format as training data |
| Road network build, UTM projection, KD-tree | ✅ 10/10 |
| Track cleaning | ✅ |
| Candidate generation | ✅ every point has ≥ 1 candidate (0 points without candidates) |
| Feature preprocessing (training-time `build_features`) | ✅ |
| ML inference (`CandidateScorer.predict_candidates`) | ✅ |
| HMM emissions/transitions + Viterbi | ✅ every point matched in every trip |
| Route reconstruction (connected arc sequence) | ✅ |
| Runtime, both modes | 144 ms – 1.1 s per trip |
| API `/health`, `/trips`, `/trips/{id}/network` with label loader disabled | ✅ 10 trips listed, all network calls 200 |

Structural note, recorded and **not** acted on: trip 00000015 (6,854 points) produced
138 decoder segment breaks, and trips 00000067 and 00000094 produced 6 each. A segment break restarts
Viterbi when no transition is reachable within the search bound. Changing decoder
parameters because of this would be tuning on evaluator data, so the configuration stays
as selected on training data.

## 5. Issue found and fixed by the check (code design, not evaluator results)

With the label loader disabled, `GET /trips/{id}/network` failed. The service loaded
the trip **with** its `.route` file, although drawing roads does not need it. Nothing was
exposed (the route was never returned), but the endpoint read labels for no reason. Fix in
`src/api/service.py`: trips load **without** the route by default, and only
`reconstruct` loads it, solely for server-side scoring. A regression test
(`test_only_reconstruct_reads_reference_routes`) enforces this. All 41 tests pass.

## 6. Evaluation-day procedure (Laptop 1)

```bash
export PATHSNAP_SERVE_PARTITION=eval_test    # the default
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
python scripts/check_eval_inputs.py          # optional: input-only health check, reads no labels
```

Restart any API process that was started before this fix so it picks up the service change.

## 7. Open points

* Confirm with the organisers that keeping the evaluator `.route` files on the API
  laptop is the intended "internal reference data" (MT §6). We assume yes.
* The archive's top folder is `evaluator_test/`; our partition folder is `eval_test/`. The
  trips were moved into it, so no code change is needed (configurable in `configs/pipeline.json`).
