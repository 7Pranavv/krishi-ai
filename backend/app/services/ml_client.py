"""Thin HTTP client for the ML service.

Every failure mode the ML service can have - down, slow, model missing, bad
input - is translated into an HTTPException the frontend can render, so one
unavailable model never breaks the rest of Krishi.AI.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import HTTPException, status

from ..config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

# One pooled client for the process; opening a connection per request is the
# most common source of latency in a proxy like this.
_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=settings.ml_service_url,
            timeout=httpx.Timeout(settings.ml_timeout_seconds, connect=5.0),
        )
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _translate(response: httpx.Response) -> dict[str, Any]:
    if response.is_success:
        return response.json()

    try:
        body = response.json()
    except ValueError:
        body = {}
    detail = {
        "error": body.get("error") or "The prediction service returned an error.",
        "detail": body.get("detail"),
    }
    # 4xx from the ML service means the user's input was wrong - pass it
    # through. 5xx is our problem, so report it as a gateway failure.
    code = response.status_code if response.status_code < 500 else status.HTTP_502_BAD_GATEWAY
    if response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
        code = status.HTTP_503_SERVICE_UNAVAILABLE
    raise HTTPException(status_code=code, detail=detail)


def _unreachable(exc: Exception) -> HTTPException:
    log.error("ML service unreachable at %s: %s", settings.ml_service_url, exc)
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "error": "The prediction service is not responding right now. "
                     "Please try again in a moment.",
            "detail": None,
        },
    )


async def post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        response = await get_client().post(path, json=payload)
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={"error": "The prediction took too long. Please try again.", "detail": None},
        ) from exc
    except httpx.HTTPError as exc:
        raise _unreachable(exc) from exc
    return _translate(response)


async def post_image(path: str, filename: str, content: bytes, content_type: str,
                     data: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        response = await get_client().post(
            path,
            files={"file": (filename, content, content_type)},
            # Extra multipart fields alongside the file, e.g. heatmap=true.
            data={k: str(v).lower() if isinstance(v, bool) else str(v)
                  for k, v in (data or {}).items()},
            timeout=httpx.Timeout(settings.ml_upload_timeout_seconds, connect=5.0),
        )
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={"error": "Analysing the image took too long. Please try a smaller photo.",
                    "detail": None},
        ) from exc
    except httpx.HTTPError as exc:
        raise _unreachable(exc) from exc
    return _translate(response)


async def get_json(path: str) -> dict[str, Any]:
    try:
        response = await get_client().get(path)
    except httpx.HTTPError as exc:
        raise _unreachable(exc) from exc
    return _translate(response)


async def health() -> dict[str, Any]:
    """Never raises - the dashboard uses this to grey out unavailable tools."""
    try:
        response = await get_client().get("/health", timeout=5.0)
        response.raise_for_status()
        return response.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("ML health check failed: %s", exc)
        return {"status": "unreachable", "models": {}}
