"""Krishi.AI ML service.

Owns every model file. Loads each one once at startup, exposes a small typed
HTTP surface, and never lets one broken model take the others down.
Only the Krishi.AI backend is expected to call it.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import registry
from .config import LOG_LEVEL
from .routers import crop, disease, fertilizer, rainfall, water

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
)
log = logging.getLogger("krishi.ml")


@asynccontextmanager
async def lifespan(_: FastAPI):
    started = time.perf_counter()
    registry.load_all()
    log.info("model registry ready in %.1fs: %s",
             time.perf_counter() - started, registry.status())
    yield


app = FastAPI(
    title="Krishi.AI ML Service",
    version="1.0.0",
    description="Inference layer for the Krishi.AI platform.",
    lifespan=lifespan,
)

# The ML service is an internal component. By default only the backend may
# reach it; ML_CORS_ORIGINS exists for debugging against a local frontend.
_origins = [o.strip() for o in os.getenv("ML_CORS_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

for module in (crop, fertilizer, water, rainfall, disease):
    app.include_router(module.router)


@app.exception_handler(StarletteHTTPException)
async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Normalise every error to {"error": ..., "detail": ...}."""
    detail = exc.detail
    body = detail if isinstance(detail, dict) else {"error": str(detail), "detail": None}
    return JSONResponse(status_code=exc.status_code, content=body)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    problems = "; ".join(
        f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors()
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Some inputs are outside the range this model supports.",
                 "detail": problems},
    )


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    # Log the traceback for developers, return nothing sensitive to the caller.
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal ML service error.", "detail": None},
    )


@app.get("/health", tags=["meta"])
def health() -> dict:
    models = registry.status()
    return {
        "status": "ok" if any(m["ready"] for m in models.values()) else "degraded",
        "service": "krishi-ml",
        "models": models,
    }
