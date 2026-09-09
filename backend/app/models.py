"""Database schema.

Consolidated from the two chat projects that were merged in:

* Conversation / Message / UserProfile come from the Krishi.AI Django backend
  (chat/models.py) - the better-designed of the two, so it is the source of
  truth for chat history.
* FaqEntry is the Shoora chatbot's ChatBotModel: a curated question/answer
  table checked before the LLM is called.
* Prediction is new - it is what makes the dashboard's "recent activity" and
  each tool's history work across all five models.
"""
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Column, Index
from sqlalchemy.types import JSON
from sqlmodel import Field, Relationship, SQLModel

# NOTE: do not add "from __future__ import annotations" here - it turns the
# Relationship annotations into plain strings and SQLAlchemy can no longer see
# the generic argument, which fails mapper configuration at runtime.


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True, max_length=64)
    password_hash: str = Field(max_length=128)
    # Guests are auto-provisioned by the browser so a farmer can use every
    # tool without a signup wall; they can claim a real account later.
    is_guest: bool = Field(default=False)
    full_name: str | None = Field(default=None, max_length=120)
    pincode: str | None = Field(default=None, max_length=10)
    language: str = Field(default="en", max_length=8)
    created_at: datetime = Field(default_factory=utcnow)
    last_active: datetime = Field(default_factory=utcnow)

    conversations: list["Conversation"] = Relationship(back_populates="user")
    predictions: list["Prediction"] = Relationship(back_populates="user")


class Conversation(SQLModel, table=True):
    __tablename__ = "conversations"

    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True, unique=True, max_length=64)
    user_id: int = Field(foreign_key="users.id", index=True)
    title: str | None = Field(default=None, max_length=120)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow, index=True)
    is_active: bool = Field(default=True)

    user: User | None = Relationship(back_populates="conversations")
    messages: list["Message"] = Relationship(
        back_populates="conversation",
        sa_relationship_kwargs={"cascade": "all, delete-orphan",
                                "order_by": "Message.created_at"},
    )


class Message(SQLModel, table=True):
    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_conversation_created", "conversation_id", "created_at"),)

    id: int | None = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversations.id")
    role: str = Field(max_length=10)  # "user" | "bot"
    content: str
    created_at: datetime = Field(default_factory=utcnow)
    meta: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    conversation: Conversation | None = Relationship(back_populates="messages")


class Prediction(SQLModel, table=True):
    """One run of any ML tool, so the dashboard can show recent activity."""

    __tablename__ = "predictions"
    __table_args__ = (Index("ix_predictions_user_created", "user_id", "created_at"),)

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id")
    tool: str = Field(max_length=32, index=True)  # crop | fertilizer | water | rainfall | disease
    summary: str = Field(max_length=200)  # human-readable headline for the card
    inputs: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    result: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow)

    user: User | None = Relationship(back_populates="predictions")


class FaqEntry(SQLModel, table=True):
    """Curated answers served before the LLM is consulted.

    Carried over from the Shoora chatbot's ChatBotModel; `hits` replaces its
    ChatHistory.count so popular questions are still visible.
    """

    __tablename__ = "faq_entries"

    id: int | None = Field(default=None, primary_key=True)
    question: str = Field(index=True, unique=True, max_length=500)
    answer: str
    hits: int = Field(default=0)
    created_at: datetime = Field(default_factory=utcnow)
