from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import (HealthResponse, NetworkResponse, ReconstructRequest, ReconstructResponse,
                             TripListResponse)
from src.api.service import PathSnapService, TripNotFound

log = logging.getLogger("pathsnap.api")
router = APIRouter()


def _svc(request: Request) -> PathSnapService:
    return request.app.state.service


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    s = _svc(request)
    return HealthResponse(
        status="ok" if s.scorer is not None and s.trip_ids else "degraded",
        model_loaded=s.scorer is not None, model_name=s.scorer.name if s.scorer else None,
        emission=s.cfg["hmm"]["emission"] if s.scorer else "gaussian",
        served_partition=s.partition, n_trips=len(s.trip_ids), messages=s.messages)


@router.get("/trips", response_model=TripListResponse)
def trips(request: Request):
    s = _svc(request)
    log.info("list trips (%d)", len(s.trip_ids))
    return TripListResponse(partition=s.partition, trips=s.list_trips())


@router.get("/trips/{trip_id}/network", response_model=NetworkResponse)
def network(trip_id: str, request: Request):
    try:
        return _svc(request).network(trip_id)
    except TripNotFound:
        raise HTTPException(status_code=404, detail=f"Unknown trip_id '{trip_id}'")


@router.post("/reconstruct", response_model=ReconstructResponse)
def reconstruct(body: ReconstructRequest, request: Request):
    client = request.client.host if request.client else "?"
    log.info("POST /reconstruct from %s trip=%s mode=%s compare=%s", client, body.trip_id, body.mode, body.compare)
    try:
        return _svc(request).reconstruct(body.trip_id, body.mode, body.compare)
    except TripNotFound:
        raise HTTPException(status_code=404, detail=f"Unknown trip_id '{body.trip_id}'")
    except ValueError as exc:
        log.warning("Bad data for trip %s: %s", body.trip_id, exc)
        raise HTTPException(status_code=422, detail=f"Trip data could not be processed: {exc}")
