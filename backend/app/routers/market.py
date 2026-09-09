"""Mandi prices and harvest value."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session

from ..db import get_session
from ..models import Prediction, User
from ..schemas import HarvestValueRequest, HarvestValueResponse, MarketPricesResponse
from ..security import current_user
from ..services import market

router = APIRouter(prefix="/market", tags=["market"])

# Above this max/min ratio the reported prices are too scattered for a median
# to represent any individual farmer's sale.
WIDE_SPREAD_RATIO = 1.5


@router.get("/options")
async def options() -> dict:
    """Commodities and states present in the current government feed."""
    return await market.options()


@router.get("/prices", response_model=MarketPricesResponse)
async def prices(
    commodity: str | None = Query(None, max_length=64),
    state: str | None = Query(None, max_length=64),
    limit: int = Query(50, ge=1, le=500),
    _: User = Depends(current_user),
) -> MarketPricesResponse:
    rows = await market.fetch_prices(commodity, state, limit)
    return MarketPricesResponse(
        count=len(rows),
        source="Agmarknet via data.gov.in",
        records=rows,
    )


@router.post("/harvest-value", response_model=HarvestValueResponse)
async def harvest_value(
    payload: HarvestValueRequest,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> HarvestValueResponse:
    """What a harvest is worth at today's reported mandi price."""
    # Pull a full page: the median is only meaningful over a broad sample, and
    # an arbitrary first-100 slice can be dominated by a single state.
    rows = await market.fetch_prices(payload.commodity, payload.state, limit=300)
    if not rows:
        # No reported price means no answer. Estimating one would be inventing
        # the single number the farmer is here to find out.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": f"No mandi has reported a price for {payload.commodity} "
                         f"in {payload.state or 'any state'} recently.",
                "detail": "Try a nearby state, a different commodity spelling, "
                          "or check back after the next market day.",
            },
        )

    # Use the median modal price so one outlier mandi cannot skew the figure.
    ordered = sorted(rows, key=lambda r: r["modal_price"])
    reference = ordered[len(ordered) // 2]

    estimate = market.estimate_value(
        reference["modal_price"], payload.yield_tonnes_per_hectare, payload.area_hectares
    )

    low = ordered[0]["modal_price"]
    high = ordered[-1]["modal_price"]

    # Mandi prices vary by variety, grade and distance from the buyer. When the
    # spread is this wide a single median is a poor guide to what any one
    # farmer will actually be paid, so the answer says so rather than letting
    # the headline figure stand unqualified.
    wide_spread = high > low * WIDE_SPREAD_RATIO

    result = HarvestValueResponse(
        commodity=reference["commodity"],
        state=reference["state"],
        market=reference["market"],
        arrival_date=reference["arrival_date"],
        markets_compared=len(rows),
        price_range={"min": low, "max": high},
        wide_spread=wide_spread,
        **estimate,
    )

    try:
        session.add(Prediction(
            user_id=user.id, tool="market",
            summary=f"{reference['commodity']}: Rs {estimate['gross_value']:,.0f} gross "
                    f"at Rs {estimate['price_per_quintal']:,.0f}/quintal",
            inputs=payload.model_dump(), result=result.model_dump(mode="json"),
        ))
        session.commit()
    except Exception:  # noqa: BLE001 - history must not fail a good answer
        session.rollback()

    return result
