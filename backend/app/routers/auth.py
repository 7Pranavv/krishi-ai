"""Registration, login, guest sessions and profile."""
from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from ..db import get_session
from ..models import User, utcnow
from ..schemas import (LoginRequest, ProfileUpdate, RegisterRequest, TokenResponse,
                       UserOut)
from ..security import (create_access_token, hash_password, current_user,
                        user_by_username, verify_password)

router = APIRouter(prefix="/auth", tags=["auth"])


def _token_for(user: User) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(user),
        user=UserOut.model_validate(user, from_attributes=True),
    )


@router.post("/guest", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def create_guest(session: Session = Depends(get_session)) -> TokenResponse:
    """Provision a throwaway account.

    A farmer opening the site should be able to use every tool immediately.
    The browser calls this once and keeps the token, so history and the
    dashboard work without a signup form. The password is random and never
    leaves the server - a guest upgrades by registering, not by logging in.
    """
    user = User(
        username=f"guest_{secrets.token_hex(8)}",
        password_hash=hash_password(secrets.token_urlsafe(32)),
        is_guest=True,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return _token_for(user)


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, session: Session = Depends(get_session)) -> TokenResponse:
    if user_by_username(session, payload.username):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": "That username is already taken.", "detail": None},
        )
    user = User(
        username=payload.username,
        password_hash=hash_password(payload.password),
        full_name=payload.full_name,
        is_guest=False,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return _token_for(user)


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, session: Session = Depends(get_session)) -> TokenResponse:
    user = user_by_username(session, payload.username.strip())
    # Same message either way so the endpoint cannot be used to enumerate users.
    if user is None or user.is_guest or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Incorrect username or password.", "detail": None},
        )
    user.last_active = utcnow()
    session.add(user)
    session.commit()
    session.refresh(user)
    return _token_for(user)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)) -> UserOut:
    return UserOut.model_validate(user, from_attributes=True)


@router.patch("/me", response_model=UserOut)
def update_me(
    payload: ProfileUpdate,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> UserOut:
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(user, field, value)
    session.add(user)
    session.commit()
    session.refresh(user)
    return UserOut.model_validate(user, from_attributes=True)
