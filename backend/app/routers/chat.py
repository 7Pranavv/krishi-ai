"""Chat API: one conversation per session_id, history preserved per user."""
from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session, func, select

from ..db import get_session
from ..models import Conversation, Message, User, utcnow
from ..schemas import (ChatRequest, ChatResponse, ConversationDetail,
                       ConversationOut, MessageOut)
from ..security import current_user
from ..services import assistant

router = APIRouter(prefix="/chat", tags=["chat"])

TITLE_MAX = 60


def _find_conversation(session: Session, user: User, session_id: str | None) -> Conversation | None:
    """The caller's existing thread, or None to start a new one.

    An unknown id means a stale tab or a cleared database; starting a fresh
    thread is friendlier than a 404 mid-conversation.
    """
    if not session_id:
        return None
    return session.exec(
        select(Conversation).where(
            Conversation.session_id == session_id,
            Conversation.user_id == user.id,  # never another user's thread
        )
    ).first()


@router.post("", response_model=ChatResponse)
async def send_message(
    payload: ChatRequest,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> ChatResponse:
    started = time.perf_counter()
    conversation = _find_conversation(session, user, payload.session_id)

    # Answer first. Creating the row up front would litter the sidebar with
    # empty "Untitled chat" entries every time the assistant is unavailable.
    answer = await assistant.answer(
        session,
        conversation.id if conversation else None,
        payload.message,
        assistant.language_name(user.language),
    )

    if conversation is None:
        conversation = Conversation(session_id=str(uuid.uuid4()), user_id=user.id)
        session.add(conversation)
        session.commit()
        session.refresh(conversation)

    session.add(Message(conversation_id=conversation.id, role="user",
                        content=payload.message))
    session.add(Message(conversation_id=conversation.id, role="bot",
                        content=answer.text, meta={"source": answer.source}))

    if not conversation.title:
        title = payload.message.strip()
        conversation.title = title[:TITLE_MAX] + ("..." if len(title) > TITLE_MAX else "")
    conversation.updated_at = utcnow()
    session.add(conversation)
    session.commit()

    return ChatResponse(
        reply=answer.text,
        session_id=conversation.session_id,
        source=answer.source,
        response_time_ms=round((time.perf_counter() - started) * 1000, 1),
    )


def _summarise(session: Session, conversation: Conversation) -> ConversationOut:
    count = session.exec(
        select(func.count(Message.id)).where(Message.conversation_id == conversation.id)
    ).one()
    return ConversationOut(
        session_id=conversation.session_id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        message_count=count,
    )


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(
    limit: int = Query(20, ge=1, le=100),
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> list[ConversationOut]:
    rows = session.exec(
        select(Conversation)
        .where(Conversation.user_id == user.id, Conversation.is_active)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
    ).all()
    return [_summarise(session, c) for c in rows]


def _owned_conversation(session: Session, user: User, session_id: str) -> Conversation:
    conversation = session.exec(
        select(Conversation).where(
            Conversation.session_id == session_id,
            Conversation.user_id == user.id,
        )
    ).first()
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "Conversation not found.", "detail": None},
        )
    return conversation


@router.get("/conversations/{session_id}", response_model=ConversationDetail)
def conversation_detail(
    session_id: str,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> ConversationDetail:
    conversation = _owned_conversation(session, user, session_id)
    messages = session.exec(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at)
    ).all()
    return ConversationDetail(
        session_id=conversation.session_id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        message_count=len(messages),
        messages=[MessageOut.model_validate(m, from_attributes=True) for m in messages],
    )


@router.delete("/conversations/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    session_id: str,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> None:
    conversation = _owned_conversation(session, user, session_id)
    session.delete(conversation)
    session.commit()
