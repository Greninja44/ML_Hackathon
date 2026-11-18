"""Input-only deployment check for the evaluator partition.

    python scripts/check_eval_inputs.py

Confirms the final model and decoder can process every evaluator trip, without
touching any reference label:

* `.route` files are never opened: ``src.data.io.load_route`` is replaced by a
  function that raises, so even an accidental call would fail loudly;
* no accuracy or other label-based metric is computed;
* nothing is retrained or tuned, and the output is pass/fail plus input statistics.

This complies with Dataset Exploration §7 / Assessment Policy §7: evaluator data
is not used for training, selection or tuning, and its labels stay unread until
the API scores a trip during the official evaluation.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import src.data.io as io  # noqa: E402
from src.config import load_config, resolve  # noqa: E402


def _forbidden(*_a, **_k):
    raise PermissionError("Reference labels (.route) must not be read by the input-only check")


io.load_route = _forbidden  # guard: label files are unreadable from here on

from src.map_matching.reconstruction import RouteReconstructor  # noqa: E402
from src.models.inference import CandidateScorer  # noqa: E402


def main():
    cfg = load_config()
    final = resolve(cfg["paths"]["final_model_dir"])
    scorer = CandidateScorer.load(final)
    hmm_path = final / "hmm_config.json"
    if hmm_path.exists():
        cfg["hmm"] = json.loads(hmm_path.read_text())
    rec = RouteReconstructor(cfg, scorer)

    files = io.list_partition(cfg, "eval_test")
    rows, ok = [], True
    for tid in sorted(files):
        f = files[tid]
        row = {"trip_id": tid, "has_track_nodes_arcs": all(e in f for e in (".track", ".nodes", ".arcs")),
               "route_file_present": ".route" in f}
        try:
            trip = io.load_trip(f, tid, "eval_test", with_route=False)
            assert trip.route is None
            t0 = time.perf_counter()
            for mode in ("hmm", "classifier"):
                r = rec.reconstruct(trip.inputs(), mode)
                if mode == "hmm":
                    c = r.prepared.cands
                    row.update({
                        "n_track_rows": int(len(trip.track)), "n_nodes": int(len(trip.nodes)),
                        "n_arcs": int(len(trip.arcs)), "n_points_used": int(len(r.pred_arc)),
                        "n_points_matched": int((r.pred_arc >= 0).sum()),
                        "points_without_candidates": int((c.counts() == 0).sum()),
                        "points_needing_fallback_radius": int(len(np.unique(c.obs[c.fallback]))),
                        "candidates_per_point_mean": round(float(c.counts().mean()), 2) if len(c.offsets) > 1 else 0,
                        "segment_breaks": int(r.n_breaks), "route_arcs": len(r.route_arcs),
                        "emission_source": r.emission_source, "cleaning": r.prepared.track.report,
                    })
            row["both_modes_ms"] = round((time.perf_counter() - t0) * 1e3)
            row["status"] = "ok" if row["n_points_matched"] > 0 else "no matches"
        except Exception as exc:  # report, keep checking the rest
            ok = False
            row["status"] = f"error: {type(exc).__name__}: {exc}"
            row["trace"] = traceback.format_exc(limit=3)
        rows.append(row)
        print(f"{tid}: {row['status']} | points {row.get('n_points_matched')}/{row.get('n_points_used')} matched"
              f" | no-candidate {row.get('points_without_candidates')} | breaks {row.get('segment_breaks')}"
              f" | {row.get('both_modes_ms')} ms")

    out = resolve(cfg["paths"]["results_dir"]) / "eval_input_check.json"
    out.write_text(json.dumps({"model": scorer.name, "hmm": cfg["hmm"], "labels_read": False,
                               "metrics_computed": False, "trips": rows}, indent=2, default=str))
    print(f"\n{sum(r['status'] == 'ok' for r in rows)}/{len(rows)} trips OK -> {out}")
    sys.exit(0 if ok and len(rows) == 10 else 1)


if __name__ == "__main__":
    main()
