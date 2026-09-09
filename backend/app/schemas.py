"""Public API contracts for the Krishi.AI backend."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .security import MAX_PASSWORD_BYTES, MIN_PASSWORD_LENGTH

USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,64}$")


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ auth
def _check_password_bytes(value: str) -> str:
    """bcrypt refuses anything over 72 BYTES, not characters.

    A 70-character Devanagari password is 210 bytes, so a character-only limit
    let it through and bcrypt then raised ValueError - surfacing as a 500. This
    matters here because the app is used in Indian languages.
    """
    if len(value.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise ValueError(
            f"Password is too long. Use at most {MAX_PASSWORD_BYTES} characters "
            "(fewer if you use non-English letters, which take more space)."
        )
    return value


class RegisterRequest(Base):
    username: str
    password: str = Field(..., min_length=MIN_PASSWORD_LENGTH)
    full_name: str | None = Field(None, max_length=120)

    @field_validator("username")
    @classmethod
    def _valid_username(cls, value: str) -> str:
        value = value.strip()
        if not USERNAME_RE.match(value):
            raise ValueError(
                "Username must be 3-64 characters, letters/digits/._- only."
            )
        return value

    @field_validator("password")
    @classmethod
    def _password_fits_bcrypt(cls, value: str) -> str:
        return _check_password_bytes(value)


class LoginRequest(Base):
    username: str = Field(..., max_length=64)
    # No byte check here: a wrong password should fail as a wrong password, not
    # as a validation error that tells an attacker anything about the format.
    password: str = Field(..., max_length=256)


class UserOut(Base):
    id: int
    username: str
    is_guest: bool
    full_name: str | None
    pincode: str | None
    language: str
    created_at: datetime
    last_active: datetime


class TokenResponse(Base):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class ProfileUpdate(Base):
    full_name: str | None = Field(None, max_length=120)
    pincode: str | None = Field(None, max_length=10)
    language: str | None = Field(None, max_length=8)

    @field_validator("pincode")
    @classmethod
    def _valid_pincode(cls, value: str | None) -> str | None:
        if value in (None, ""):
            return None
        if not re.fullmatch(r"\d{6}", value):
            raise ValueError("Pincode must be 6 digits.")
        return value


# ------------------------------------------------------------------ chat
class ChatRequest(Base):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(None, max_length=64)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message cannot be empty.")
        return value


class ChatResponse(Base):
    reply: str
    session_id: str
    source: str = Field(..., description="'faq' for a curated answer, 'ai' for the model")
    response_time_ms: float


class MessageOut(Base):
    id: int
    role: str
    content: str
    created_at: datetime


class ConversationOut(Base):
    session_id: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    message_count: int


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]


# -------------------------------------------------------------- history
class PredictionOut(Base):
    id: int
    tool: str
    summary: str
    inputs: dict[str, Any]
    result: dict[str, Any]
    created_at: datetime


class DashboardResponse(Base):
    user: UserOut
    total_predictions: int
    total_conversations: int
    recent_predictions: list[PredictionOut]
    recent_conversations: list[ConversationOut]
    services: dict[str, Any]


# --------------------------------------------------------------- market
class MarketPricesResponse(Base):
    count: int
    source: str
    records: list[dict[str, Any]]


class HarvestValueRequest(Base):
    commodity: str = Field(..., max_length=64)
    state: str | None = Field(None, max_length=64)
    yield_tonnes_per_hectare: float = Field(..., gt=0, le=200)
    area_hectares: float = Field(..., gt=0, le=500)


class HarvestValueResponse(Base):
    commodity: str | None
    state: str | None
    market: str | None
    arrival_date: str | None
    markets_compared: int
    price_range: dict[str, float]
    wide_spread: bool = Field(
        False, description="Reported prices vary too much for the median to be a "
                           "reliable guide to one farmer's sale")
    quintals: float
    price_per_quintal: float
    gross_value: float
    basis: str


# Error bodies are built by the exception handlers in main.py, which is the
# single place that shape is defined - no schema class is needed for them.
