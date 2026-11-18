import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.data.io import iter_partition
from src.data.prepare import prepare_trip
from src.models.inference import CandidateScorer, ModelArtifactError
from src.models.registry import default_specs


def test_registry_has_at_least_five_probabilistic_models():
    specs = default_specs(1)
    assert len(specs) >= 5
    assert len({s.family for s in specs.values()}) >= 5


def test_model_load_and_inference(cfg, trained_model_dir):
    scorer = CandidateScorer.load(trained_model_dir)
    trip = next(iter_partition(cfg, "student_test"))
    feats = prepare_trip(trip.inputs(), cfg).features
    p = scorer.predict_candidates(feats)
    assert p.shape == (len(feats),) and ((p >= 0) & (p <= 1)).all()
    assert scorer.predict_candidates(feats.iloc[:0]).shape == (0,)
    with pytest.raises(ValueError):
        scorer.predict_candidates(feats.drop(columns=["dist_m"]))


def test_missing_artifact(tmp_path):
    with pytest.raises(ModelArtifactError):
        CandidateScorer.load(tmp_path)


@pytest.fixture(scope="module")
def client(cfg, trained_model_dir):
    from src.api.main import create_app
    from src.api.service import PathSnapService
    svc = PathSnapService(cfg=dict(cfg), partition="eval_test")
    with TestClient(create_app(svc)) as c:
        yield c


def test_health_and_trips(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_loaded"] and h["n_trips"] == 3
    trips = client.get("/trips").json()["trips"]
    assert len(trips) == 3 and all(t["n_observations"] > 0 for t in trips)


def test_reconstruct_every_served_trip(client):
    for t in client.get("/trips").json()["trips"]:
        r = client.post("/reconstruct", json={"trip_id": t["trip_id"], "mode": "hmm", "compare": True})
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["observations"]) == t["n_observations"]
        assert body["metrics"]["top1_accuracy"] is not None
        assert set(body["comparison"]) == {"hmm", "classifier"}
        assert body["route_polylines"] and body["emission_source"] in ("ml", "hybrid")


def test_ground_truth_not_exposed(client, cfg):
    tid = client.get("/trips").json()["trips"][0]["trip_id"]
    body = client.post("/reconstruct", json={"trip_id": tid}).json()
    forbidden = {"route", "reference", "reference_route", "label", "label_arc", "ground_truth"}
    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield k
                yield from keys(v)
        elif isinstance(o, list):
            for v in o:
                yield from keys(v)
    assert not forbidden & set(keys(body))


def test_api_errors(client):
    assert client.post("/reconstruct", json={"trip_id": "does-not-exist"}).status_code == 404
    assert client.post("/reconstruct", json={"trip_id": "x", "mode": "magic"}).status_code == 422
    assert client.post("/reconstruct", json={}).status_code == 422
    assert client.get("/trips/nope/network").status_code == 404


def test_network_endpoint(client):
    tid = client.get("/trips").json()["trips"][0]["trip_id"]
    net = client.get(f"/trips/{tid}/network").json()
    assert net["arcs"] and len(net["arcs"][0]) == 2


def test_service_without_model_falls_back(cfg, tmp_path):
    from src.api.service import PathSnapService
    c2 = {**cfg, "paths": {**cfg["paths"], "final_model_dir": str(tmp_path / "none")}}
    svc = PathSnapService(cfg=c2, partition="eval_test")
    out = svc.reconstruct(svc.trip_ids[0])
    assert out["emission_source"] == "gaussian" and out["model_name"] is None


def test_training_pipeline_end_to_end(cfg):
    """Train → CV → student-test evaluation on synthetic data (pipeline check, not a result)."""
    from src.experiments.trip_eval import load_labelled_trips
    from src.models.dataset import build_partition
    from src.models.train import run_model
    spec = default_specs(1)["logistic_regression"]
    train = build_partition(cfg, "train", n_jobs=1)
    student = load_labelled_trips(cfg, "student_test")
    row = run_model(spec, dict(spec.params), train, student, cfg, "lr_e2e")
    for k in ("cv_macro_f1_mean", "cv_top1_mean", "student_top1", "student_hmm_top1", "student_log_loss",
              "student_route_continuity", "inference_ms_per_1k_pairs"):
        assert np.isfinite(row[k]), k


def test_only_reconstruct_reads_reference_routes(client, monkeypatch):
    """/health, /trips and /trips/{id}/network must work without ever opening a .route file."""
    import src.data.io as io

    def forbidden(*_a, **_k):
        raise AssertionError(".route read by an endpoint that does not score")

    monkeypatch.setattr(io, "load_route", forbidden)
    tid = client.get("/trips").json()["trips"][0]["trip_id"]
    assert client.get("/health").status_code == 200
    assert client.get(f"/trips/{tid}/network").status_code == 200
