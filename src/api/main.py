"""FastAPI application (Laptop 1).

Run:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
or  python -m src.api.main

Environment:
    PATHSNAP_SERVE_PARTITION   eval_test (default) | student_test
    PATHSNAP_CORS_ORIGINS      comma-separated origins allowed to call the API (default "*")
    PATHSNAP_DATA_DIR / PATHSNAP_MODEL_DIR / PATHSNAP_CONFIG   see src/config.py

Swagger UI: http://<host>:8000/docs
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.api.service import PathSnapService
from src.logging_utils import setup_logging

log = logging.getLogger("pathsnap.api")


def create_app(service: PathSnapService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is None:
            setup_logging("api")
        app.state.service = service or PathSnapService()
        log.info("PathSnap API ready")
        yield

    app = FastAPI(title="PathSnap API", version="1.0.0", lifespan=lifespan,
                  description="Map-matching and route reconstruction for noisy GPS trajectories.")
    origins = [o.strip() for o in os.environ.get("PATHSNAP_CORS_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                       allow_headers=["*"])

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        rid = uuid.uuid4().hex[:8]
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("[%s] %s %s failed", rid, request.method, request.url.path)
            return JSONResponse({"detail": "Internal server error"}, status_code=500)
        log.info("[%s] %s %s -> %d (%.0f ms)", rid, request.method, request.url.path, response.status_code,
                 (time.perf_counter() - t0) * 1e3)
        return response

    app.include_router(router)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host=os.environ.get("PATHSNAP_HOST", "0.0.0.0"),
                port=int(os.environ.get("PATHSNAP_PORT", "8000")))
