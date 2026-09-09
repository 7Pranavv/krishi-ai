"""Mandi prices from the Government of India's open data platform.

Source: data.gov.in resource "Current Daily Price of Various Commodities from
Various Markets (Mandi)", which republishes Agmarknet's daily arrivals.

These are real reported prices, not estimates. Where a mandi has not reported,
this module returns nothing rather than interpolating - a farmer deciding when
to sell must not be shown an invented number. Every figure carries the date and
market it came from so it can be checked.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from fastapi import HTTPException, status

from ..config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

# Prices are published once a day, so a short cache spares the upstream quota
# without ever showing yesterday's data as today's.
_CACHE: dict[tuple, tuple[float, list[dict]]] = {}
CACHE_TTL_SECONDS = 900

# data.gov.in's own filtering breaks above ~300 rows per page: ask for 350 with
# filters[state]=Madhya Pradesh and it starts returning Uttar Pradesh rows
# alongside, while still reporting the filtered total. Verified 2026-09-09 -
# 250 and 300 are clean, 350 leaks 45 rows, 500 leaks 195. Do not raise this
# without re-testing, and note the client-side filter in fetch_prices() that
# backs it up.
MAX_UPSTREAM_PAGE = 300

# data.gov.in sits behind a WAF that silently BLACK-HOLES requests whose
# User-Agent is the httpx default - the connection opens, nothing ever comes
# back, and the request dies on the read timeout. Any recognisable agent gets
# an immediate response. Identify ourselves properly; do not remove this.
REQUEST_HEADERS = {
    "User-Agent": "Krishi.AI/1.0 (+agriculture advisory; contact via deployment owner)",
    "Accept": "application/json",
}

_NOT_CONFIGURED = HTTPException(
    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    detail={
        "error": "Market prices are not configured on this server.",
        "detail": "Set DATA_GOV_API_KEY to enable them. Get a free key at "
                  "https://data.gov.in/apis - the rest of Krishi.AI works without it.",
    },
)


def _to_float(value: Any) -> float | None:
    """Upstream sends numbers as strings, and uses placeholders for missing."""
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    # Agmarknet uses 0 and -1 to mean "not reported".
    return number if number > 0 else None


def _normalise(record: dict) -> dict | None:
    """Map one upstream row to our shape, dropping rows with no usable price.

    Field names are read defensively: the resource has changed their casing
    and spelling between revisions.
    """
    def field(*names: str) -> Any:
        for name in names:
            if name in record and record[name] not in (None, ""):
                return record[name]
        return None

    modal = _to_float(field("modal_price", "Modal_Price", "modal_x0020_price"))
    if modal is None:
        return None  # no price reported - omit the row entirely

    return {
        "commodity": field("commodity", "Commodity"),
        "variety": field("variety", "Variety"),
        "grade": field("grade", "Grade"),
        "state": field("state", "State"),
        "district": field("district", "District"),
        "market": field("market", "Market"),
        "arrival_date": field("arrival_date", "Arrival_Date"),
        "min_price": _to_float(field("min_price", "Min_Price")),
        "max_price": _to_float(field("max_price", "Max_Price")),
        "modal_price": modal,
        # The resource quotes rupees per quintal (100 kg).
        "unit": "INR per quintal",
    }


async def fetch_prices(commodity: str | None = None, state: str | None = None,
                       limit: int = 50) -> list[dict]:
    """Recent mandi prices, newest first. Raises 503 when unconfigured."""
    if not settings.data_gov_api_key:
        raise _NOT_CONFIGURED

    key = (commodity or "", state or "", limit)
    cached = _CACHE.get(key)
    if cached and time.time() - cached[0] < CACHE_TTL_SECONDS:
        return cached[1]

    params: dict[str, Any] = {
        "api-key": settings.data_gov_api_key,
        "format": "json",
        "limit": min(max(limit, 1), MAX_UPSTREAM_PAGE),
    }
    if commodity:
        params["filters[commodity]"] = commodity
    if state:
        params["filters[state]"] = state

    response = None
    try:
        async with httpx.AsyncClient(timeout=30.0, headers=REQUEST_HEADERS) as client:
            # data.gov.in throttles aggressively, especially on shared keys.
            # One short retry turns most 429s into a normal answer instead of
            # an error the farmer has to act on.
            for attempt in range(2):
                response = await client.get(settings.market_api_url, params=params)
                if response.status_code != 429:
                    break
                if attempt == 0:
                    log.info("data.gov.in rate-limited; retrying once")
                    await asyncio.sleep(1.5)
    except httpx.HTTPError as exc:
        log.error("mandi price request failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "Could not reach the government price service. "
                             "Please try again shortly.", "detail": None},
        ) from exc

    if response.status_code == 429:
        log.warning("data.gov.in rate limit exceeded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "The government price service is rate-limiting this server.",
                "detail": "This usually means the API key is shared. Register your "
                          "own free key at https://data.gov.in/apis for reliable access.",
            },
        )

    if response.status_code in (401, 403):
        log.error("data.gov.in rejected the API key (%s)", response.status_code)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "The market price service rejected this server's API key.",
                    "detail": None},
        )
    if not response.is_success:
        log.error("data.gov.in returned %s", response.status_code)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"error": "The government price service returned an error.",
                    "detail": None},
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"error": "The price service sent an unreadable response.",
                    "detail": None},
        ) from exc

    rows = [r for r in (_normalise(x) for x in payload.get("records", [])) if r]

    # Second line of defence against the upstream filter bug described above.
    # Showing a Punjab farmer Uttar Pradesh rates would be worse than showing
    # fewer rows, so anything that does not match what was asked for is dropped.
    before = len(rows)
    if commodity:
        rows = [r for r in rows if (r["commodity"] or "").strip().lower() == commodity.strip().lower()]
    if state:
        rows = [r for r in rows if (r["state"] or "").strip().lower() == state.strip().lower()]
    if len(rows) != before:
        log.warning("upstream returned %d rows not matching the requested filter; dropped",
                    before - len(rows))

    _CACHE[key] = (time.time(), rows)
    return rows


async def options() -> dict[str, list[str]]:
    """Commodities and states actually present in today's feed.

    Read from live data rather than hardcoded, so the dropdowns can never offer
    something with no prices behind it.
    """
    rows = await fetch_prices(limit=500)
    return {
        "commodities": sorted({r["commodity"] for r in rows if r["commodity"]}),
        "states": sorted({r["state"] for r in rows if r["state"]}),
    }


def estimate_value(modal_price_per_quintal: float, yield_tonnes_per_ha: float,
                   area_hectares: float) -> dict:
    """Gross value of a harvest at the quoted mandi price.

    Deliberately arithmetic on two real numbers - a reported price and the
    farmer's own expected yield. It is not a profit model: it does not know
    their input costs, so it never claims to.
    """
    quintals = yield_tonnes_per_ha * 10 * area_hectares  # 1 tonne = 10 quintal
    return {
        "quintals": round(quintals, 2),
        "price_per_quintal": round(modal_price_per_quintal, 2),
        "gross_value": round(quintals * modal_price_per_quintal, 2),
        "basis": "Gross value at the quoted modal price, before input costs, "
                 "transport, commission and taxes.",
    }
