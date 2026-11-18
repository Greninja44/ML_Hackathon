# PathSnap API

FastAPI service for Laptop 1. Interactive docs: `http://<host>:8000/docs` (Swagger) and `/redoc`.

```bash
export PATHSNAP_SERVE_PARTITION=eval_test        # student_test to rehearse
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

| Variable | Default | Meaning |
|---|---|---|
| `PATHSNAP_SERVE_PARTITION` | `eval_test` | which `data/raw/<partition>/` the API serves |
| `PATHSNAP_DATA_DIR` | `data/raw` | root of the partition folders |
| `PATHSNAP_MODEL_DIR` | `models/final` | `model.joblib`, `metadata.json`, `hmm_config.json` |
| `PATHSNAP_CORS_ORIGINS` | `*` | comma-separated allowed UI origins |

At start-up the service loads the final model and the tuned HMM config once.
Each request runs the **full pipeline** on the trip's raw `.track`, `.nodes`
and `.arcs`. The `.route` file is read only inside the server to compute
metrics. No response field contains reference-route data; `tests/test_models_api.py` enforces this.

Every request is logged to the console and `results/logs/api.log`, e.g.

```
… | pathsnap.api | POST /reconstruct from 127.0.0.1 trip=00000052 mode=hmm compare=True
… | pathsnap.api | reconstruct trip=00000052 mode=hmm model=hist_gradient_boosting_tuned emission=hybrid obs=262 matched=262 breaks=0 top1=1.0000 continuity=0.8902 total=117ms
… | pathsnap.api | [a5e586e5] POST /reconstruct -> 200 (120 ms)
```

## Endpoints

### `GET /health`
```json
{"status":"ok","model_loaded":true,"model_name":"hist_gradient_boosting_tuned","emission":"hybrid",
 "served_partition":"student_test","n_trips":10,"messages":[]}
```
`status` is `degraded` when no model or no trips are found; `messages` explains why.

### `GET /trips`
The trips the evaluator can choose from (no reference data).
```json
{"partition":"student_test","trips":[
  {"trip_id":"00000000","n_observations":2503,"duration_s":2503.0,"bbox":[56.08056,54.248491,56.254566,54.963519]}, …]}
```
`bbox` = `[min_lat, min_lon, max_lat, max_lon]`.

### `GET /trips/{trip_id}/network`
Road segments (arcs) within 150 m of the trajectory, for drawing context:
`{"trip_id": "...", "n_arcs": 13145, "arcs": [[[lat, lon], [lat, lon]], …]}`. 404 for an unknown trip.

### `POST /reconstruct`
Request:
```json
{"trip_id": "00000052", "mode": "hmm", "compare": true}
```
* `mode`: `"hmm"` (classifier + HMM/Viterbi) or `"classifier"` (per-point argmax).
* `compare`: also run the other mode and return its metrics.

Response (real output for student-test trip 00000052, lists shortened):
```json
{
  "trip_id": "00000052", "mode": "hmm",
  "model_name": "hist_gradient_boosting_tuned", "emission_source": "hybrid",
  "n_observations_raw": 262, "n_observations_used": 262, "n_matched": 262, "n_segment_breaks": 0,
  "route_arc_ids": [1680, 1671, 3559, 1685, 3558, "… 93 total"],
  "route_polylines": [[[56.7994, 35.8577], [56.7986, 35.8604], "…"]],
  "observations": [
    {"index": 0, "lat": 56.7991, "lon": 35.8584, "t": 1.0, "arc_id": 1680,
     "snapped": [56.7992, 35.8584], "score": 0.9859},
    "… 261 more"],
  "metrics": {"top1_accuracy": 1.0, "candidate_miss_rate": 0.0, "route_continuity": 0.8902,
              "seq_precision": 1.0, "seq_recall": 0.9789, "seq_f1": 0.9894,
              "length_precision": 1.0, "length_recall": 0.9969, "edit_similarity": 0.9789,
              "n_labelled_observations": 262},
  "comparison": {"hmm": {"top1_accuracy": 1.0, "…": "…"}, "classifier": {"top1_accuracy": 0.9695, "…": "…"}},
  "timings_ms": {"prepare_ms": "…", "score_ms": "…", "decode_ms": "…", "assemble_ms": "…", "request_total_ms": 117},
  "cleaning_report": {"n_raw": 262, "n_invalid_dropped": 0, "n_duplicates_dropped": 0, "was_time_sorted": true,
                      "n_outliers_flagged": 0, "n_kept": 262}
}
```

| Field | Meaning |
|---|---|
| `observations[].index` | row in the raw `.track` file |
| `observations[].arc_id` | predicted road segment (row in `.arcs`); `null` if no road within 150 m |
| `observations[].snapped` | GPS point projected onto the predicted arc, `[lat, lon]` |
| `observations[].score` | classifier probability of the chosen candidate |
| `route_arc_ids` | connected predicted route (consecutive duplicates removed; short gaps bridged by shortest network paths) |
| `route_polylines` | the route as `[lat, lon]` polylines for drawing |
| `metrics` | computed server-side against the internal reference route (definitions: `docs/methodology.md`); `null` if the trip has no `.route` |

Errors:

| Status | When | Body |
|---|---|---|
| 404 | unknown `trip_id` | `{"detail":"Unknown trip_id 'nope'"}` |
| 422 | invalid body (e.g. `mode: "magic"`, missing `trip_id`) | FastAPI validation detail |
| 422 | trip files present but unusable | `{"detail":"Trip data could not be processed: …"}` |
| 500 | unexpected error (logged with a request id) | `{"detail":"Internal server error"}` |

## Measured latency (real student-test trips, this laptop, Apple Silicon 12-core)

117 ms (262 points) to 1.54 s (3,958 points) per `/reconstruct` with `compare: true`.
