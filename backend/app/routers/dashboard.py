"""Dashboard and prediction history."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session, func, select

from ..config import get_settings
from ..db import get_session
from ..models import Conversation, Message, Prediction, User
from ..schemas import (ConversationOut, DashboardResponse, PredictionOut, UserOut)
from ..security import current_user
from ..services import ml_client

settings = get_settings()
router = APIRouter(tags=["dashboard"])

TOOLS = {"crop", "fertilizer", "water", "rainfall", "disease"}


@router.get("/dashboard", response_model=DashboardResponse)
async def dashboard(
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> DashboardResponse:
    predictions = session.exec(
        select(Prediction)
        .where(Prediction.user_id == user.id)
        .order_by(Prediction.created_at.desc())
        .limit(5)
    ).all()

    conversations = session.exec(
        select(Conversation)
        .where(Conversation.user_id == user.id, Conversation.is_active)
        .order_by(Conversation.updated_at.desc())
        .limit(5)
    ).all()

    ml_health = await ml_client.health()

    return DashboardResponse(
        user=UserOut.model_validate(user, from_attributes=True),
        total_predictions=session.exec(
            select(func.count(Prediction.id)).where(Prediction.user_id == user.id)
        ).one(),
        total_conversations=session.exec(
            select(func.count(Conversation.id)).where(Conversation.user_id == user.id)
        ).one(),
        recent_predictions=[
            PredictionOut.model_validate(p, from_attributes=True) for p in predictions
        ],
        recent_conversations=[
            ConversationOut(
                session_id=c.session_id,
                title=c.title,
                created_at=c.created_at,
                updated_at=c.updated_at,
                message_count=session.exec(
                    select(func.count(Message.id))
                    .where(Message.conversation_id == c.id)
                ).one(),
            )
            for c in conversations
        ],
        services={
            "ml": ml_health,
            "assistant": {"ready": settings.ai_enabled},
            "market": {"ready": settings.market_enabled},
        },
    )


@router.get("/predictions", response_model=list[PredictionOut])
def list_predictions(
    tool: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> list[PredictionOut]:
    if tool is not None and tool not in TOOLS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": f"Unknown tool {tool!r}.",
                    "detail": f"Expected one of: {', '.join(sorted(TOOLS))}"},
        )
    query = select(Prediction).where(Prediction.user_id == user.id)
    if tool:
        query = query.where(Prediction.tool == tool)
    rows = session.exec(query.order_by(Prediction.created_at.desc()).limit(limit)).all()
    return [PredictionOut.model_validate(p, from_attributes=True) for p in rows]


@router.delete("/predictions/{prediction_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_prediction(
    prediction_id: int,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> None:
    prediction = session.get(Prediction, prediction_id)
    if prediction is None or prediction.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail={"error": "Prediction not found.", "detail": None})
    session.delete(prediction)
    session.commit()
