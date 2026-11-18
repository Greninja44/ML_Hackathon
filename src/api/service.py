"""Server-side application logic (Laptop 1).

* loads the final model + tuned HMM config once at start-up;
* serves the trips of one partition (``eval_test`` by default — set
  ``PATHSNAP_SERVE_PARTITION=student_test`` to rehearse);
* for each request runs the *full* pipeline on the trip's GPS track and road
  network only, then scores the prediction against the reference route that
  never leaves this process.
"""
from __future__ import annotations

import json
import logging
import os
import time

import numpy as np

from src.config import load_config, resolve
from src.data.io import TripData, list_partition, load_trip
from src.map_matching.metrics import candidate_miss_rate, sequence_metrics, top1_accuracy
from src.map_matching.reconstruction import ReconstructionResult, RouteReconstructor
from src.models.inference import CandidateScorer, ModelArtifactError
from src.visualization.folium_maps import _route_polyline

log = logging.getLogger("pathsnap.api")


class TripNotFound(KeyError):
    pass


class PathSnapService:
    def __init__(self, cfg: dict | None = None, partition: str | None = None):
        self.cfg = cfg or load_config()
        self.partition = partition or os.environ.get("PATHSNAP_SERVE_PARTITION", "eval_test")
        self.messages: list[str] = []
        final_dir = resolve(self.cfg["paths"]["final_model_dir"])
        self.scorer = None
        try:
            self.scorer = CandidateScorer.load(final_dir)
        except ModelArtifactError as exc:
            self.messages.append(f"No trained model found ({exc}); using distance-only HMM emissions.")
            log.warning(self.messages[-1])
        hmm_path = final_dir / "hmm_config.json"
        if hmm_path.exists():
            self.cfg["hmm"] = json.loads(hmm_path.read_text())
            log.info("Loaded tuned HMM config from %s", hmm_path)
        self.reconstructor = RouteReconstructor(self.cfg, self.scorer)
        self.refresh_trips()

    # ------------------------------------------------------------------ trips
    def refresh_trips(self):
        self._files = list_partition(self.cfg, self.partition)
        if not self._files:
            self.messages.append(f"No trips found for partition '{self.partition}'.")
            log.warning(self.messages[-1])
        log.info("Serving %d trips from partition '%s'", len(self._files), self.partition)

    @property
    def trip_ids(self) -> list[str]:
        return sorted(self._files)

    def _load(self, trip_id: str, with_route: bool = False) -> TripData:
        """Reference routes are read only when a trip is scored (``reconstruct``)."""
        if trip_id not in self._files:
            raise TripNotFound(trip_id)
        return load_trip(self._files[trip_id], trip_id, self.partition, with_route=with_route)

    def list_trips(self) -> list[dict]:
        out = []
        for tid in self.trip_ids:
            tr = load_trip(self._files[tid], tid, self.partition, with_route=False).track.dropna()
            out.append({"trip_id": tid, "n_observations": len(tr),
                        "duration_s": float(tr["t"].max() - tr["t"].min()) if len(tr) else 0.0,
                        "bbox": [float(tr["lat"].min()), float(tr["lon"].min()),
                                 float(tr["lat"].max()), float(tr["lon"].max())] if len(tr) else [0, 0, 0, 0]})
        return out

    def network(self, trip_id: str, buffer_m: float = 150.0) -> dict:
        from src.data.prepare import build_network
        trip = self._load(trip_id)
        net = build_network(trip.inputs(), self.cfg)
        tr = trip.track.dropna()
        xy = net.proj.to_xy(tr["lon"].to_numpy(), tr["lat"].to_numpy())
        near = set()
        for p in xy[:: max(1, len(xy) // 400)]:
            near.update(net.arcs_near(p, buffer_m)[0].tolist())
        arcs = []
        for a in sorted(near):
            f, t = net.arcs[a]
            arcs.append([(float(net.node_lonlat[f, 1]), float(net.node_lonlat[f, 0])),
                         (float(net.node_lonlat[t, 1]), float(net.node_lonlat[t, 0]))])
        return {"trip_id": trip_id, "n_arcs": int(net.n_arcs), "arcs": arcs}

    # ---------------------------------------------------------- reconstruction
    def _metrics(self, res: ReconstructionResult, trip: TripData):
        """Server-side scoring against the internal reference route."""
        if not trip.has_route:
            return None, None
        from src.data.labels import align_route
        prep = res.prepared
        al = align_route(prep.track.xy, trip.route, prep.net)
        sm = sequence_metrics(res.route_arcs, trip.route, prep.net)
        lab = al.label_arc

        def f(x):
            return None if x is None or not np.isfinite(x) else float(x)

        return {
            "top1_accuracy": f(top1_accuracy(res.pred_arc, lab)),
            "candidate_miss_rate": f(candidate_miss_rate(prep.cands.obs, prep.cands.arc, lab)),
            "route_continuity": f(res.continuity["route_continuity"]),
            **{k: f(v) for k, v in sm.items()},
            "n_labelled_observations": int((lab >= 0).sum()),
        }, lab

    def reconstruct(self, trip_id: str, mode: str = "hmm", compare: bool = False) -> dict:
        t0 = time.perf_counter()
        trip = self._load(trip_id, with_route=True)  # scoring only; route never leaves the server
        inputs = trip.inputs()  # GPS + network only: the reference route is not passed on
        res = self.reconstructor.reconstruct(inputs, mode)
        metrics, _ = self._metrics(res, trip)

        comparison = None
        if compare:
            other = "classifier" if mode == "hmm" else "hmm"
            res2 = self.reconstructor.reconstruct(inputs, other, prepared=res.prepared, proba=res.proba)
            comparison = {mode: metrics, other: self._metrics(res2, trip)[0]}

        prep = res.prepared
        net, c = prep.net, prep.cands
        snapped_xy = np.full((len(res.pred_arc), 2), np.nan)
        ok = res.chosen >= 0
        k = res.chosen[ok]
        a, b = net.arc_a[c.arc[k]], net.arc_b[c.arc[k]]
        snapped_xy[ok] = a + c.t_along[k][:, None] * (b - a)
        snapped_ll = np.full_like(snapped_xy, np.nan)
        if ok.any():
            snapped_ll[ok] = net.proj.to_lonlat(snapped_xy[ok])
        tr = prep.track
        observations = [{
            "index": int(tr.obs_index[i]), "lat": float(tr.lat[i]), "lon": float(tr.lon[i]), "t": float(tr.t[i]),
            "arc_id": int(res.pred_arc[i]) if res.pred_arc[i] >= 0 else None,
            "snapped": (float(snapped_ll[i, 1]), float(snapped_ll[i, 0])) if ok[i] else None,
            "score": float(res.pred_score[i]) if np.isfinite(res.pred_score[i]) else None,
        } for i in range(len(tr))]
        polylines = [[(float(la), float(lo)) for la, lo in line] for line in _route_polyline(net, res.route_arcs)]

        timings = dict(res.timings_ms)
        timings["request_total_ms"] = (time.perf_counter() - t0) * 1e3
        log.info("reconstruct trip=%s mode=%s model=%s emission=%s obs=%d matched=%d breaks=%d top1=%s "
                 "continuity=%s total=%.0fms", trip_id, mode, self.scorer.name if self.scorer else None,
                 res.emission_source, len(tr), int(ok.sum()), res.n_breaks,
                 f"{metrics['top1_accuracy']:.4f}" if metrics and metrics["top1_accuracy"] is not None else "n/a",
                 f"{res.continuity['route_continuity']:.4f}", timings["request_total_ms"])
        return {
            "trip_id": trip_id, "mode": mode, "model_name": self.scorer.name if self.scorer else None,
            "emission_source": res.emission_source, "n_observations_raw": int(tr.report["n_raw"]),
            "n_observations_used": len(tr), "n_matched": int(ok.sum()), "n_segment_breaks": int(res.n_breaks),
            "route_arc_ids": [int(x) for x in res.route_arcs], "route_polylines": polylines,
            "observations": observations, "metrics": metrics, "comparison": comparison,
            "timings_ms": timings, "cleaning_report": tr.report,
        }
