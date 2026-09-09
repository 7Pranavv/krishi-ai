"""Krishi.AI backend.

Serves the API under /api and the frontend from /. Sits between the browser
and the ML service so no model file, API key or database is ever reachable
from a page.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import BASE_DIR, get_settings
from .db import init_db
from .routers import auth, chat, dashboard, market, ml
from .services import ml_client

settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
)
log = logging.getLogger("krishi.api")

FRONTEND_DIR = BASE_DIR.parent / "frontend"


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    if not settings.ai_enabled:
        log.warning("GEMINI_API_KEY is not set - the AI assistant will be unavailable.")
    log.info("Krishi.AI backend ready (env=%s, ml=%s)",
             settings.environment, settings.ml_service_url)
    yield
    await ml_client.close_client()


app = FastAPI(
    title="Krishi.AI",
    description="An AI-powered digital assistant for farmers.",
    version="1.0.0",
    lifespan=lifespan,
    # Interactive docs are useful in development, noise in production.
    docs_url=None if settings.is_production else "/api/docs",
    openapi_url=None if settings.is_production else "/api/openapi.json",
)

# The bundled frontend is same-origin and needs no CORS entry. This is only
# for hosting the frontend on a separate domain - never "*", because the API
# is token-authenticated.
if settings.cors_origin_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

for module in (auth, chat, ml, dashboard, market):
    app.include_router(module.router, prefix="/api")


# There is no build step, so asset URLs carry no content hash and a browser
# would happily keep an old copy. ES modules make that fatal rather than
# cosmetic: one stale file that no longer exports what a fresh sibling imports
# fails the whole module graph, and the page renders nothing at all.
#
# "no-cache" does not mean "do not store" - the browser still caches, it just
# revalidates. With the ETag that StaticFiles already sends, an unchanged file
# costs one 304 and no body.
_REVALIDATE_PREFIXES = ("/js/", "/css/", "/i18n/")


@app.middleware("http")
async def revalidate_code_assets(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.startswith(_REVALIDATE_PREFIXES) or response.headers.get(
        "content-type", ""
    ).startswith("text/html"):
        response.headers["Cache-Control"] = "no-cache"
    return response


# ------------------------------------------------------- error handling
@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
    """API errors come back as {"error": ..., "detail": ...}; page requests get
    the friendly 404 page.

    Note: re-raising here to "fall through" to Starlette's default does not
    work - the exception escapes the handler and becomes a 500. Every branch
    must return a response.
    """
    if request.url.path.startswith("/api"):
        detail = exc.detail
        body = detail if isinstance(detail, dict) else {"error": str(detail), "detail": None}
        return JSONResponse(status_code=exc.status_code, content=body,
                            headers=getattr(exc, "headers", None))

    if exc.status_code == status.HTTP_404_NOT_FOUND and (FRONTEND_DIR / "404.html").is_file():
        return FileResponse(FRONTEND_DIR / "404.html", status_code=404)
    return PlainTextResponse(str(exc.detail), status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    problems = "; ".join(
        f"{'.'.join(str(p) for p in e['loc'][1:]) or 'request'}: {e['msg']}"
        for e in exc.errors()
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Please check the values you entered.", "detail": problems},
    )


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    # Full traceback to the log, nothing internal to the client.
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Something went wrong on our side. Please try again.",
                 "detail": None},
    )


# --------------------------------------------------------------- health
@app.get("/api/health", tags=["meta"])
async def health() -> dict:
    ml_status = await ml_client.health()
    return {
        "status": "ok",
        "service": "krishi-api",
        "environment": settings.environment,
        "assistant": {"ready": settings.ai_enabled, "model": settings.gemini_model},
        "market": {"ready": settings.market_enabled},
        "ml": ml_status,
    }


# ------------------------------------------------------------- frontend
if FRONTEND_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")
    app.mount("/css", StaticFiles(directory=FRONTEND_DIR / "css"), name="css")
    app.mount("/js", StaticFiles(directory=FRONTEND_DIR / "js"), name="js")
    app.mount("/i18n", StaticFiles(directory=FRONTEND_DIR / "i18n"), name="i18n")

    # HEAD as well as GET: uptime monitors and link checkers use it, and
    # FastAPI does not add it automatically the way plain Starlette does.
    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.api_route("/{page}", methods=["GET", "HEAD"], include_in_schema=False)
    async def page(page: str) -> FileResponse:
        """Serve /chat, /crop, ... as their .html files.

        Constrained to a bare filename so the path can never escape the
        frontend directory.
        """
        name = Path(page).name
        candidate = (FRONTEND_DIR / f"{name}.html").resolve()
        if candidate.parent == FRONTEND_DIR.resolve() and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIR / "404.html", status_code=404)
else:  # pragma: no cover - only when the API is deployed on its own
    log.warning("frontend directory not found at %s - serving API only", FRONTEND_DIR)
