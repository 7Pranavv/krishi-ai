"""Password hashing and JWT issue/verify."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlmodel import Session, select

from .config import get_settings
from .db import get_session
from .models import User, utcnow

settings = get_settings()
# auto_error=False so optional-auth routes can fall through instead of 403ing.
_bearer = HTTPBearer(auto_error=False)

# bcrypt truncates at 72 bytes; reject longer rather than silently ignoring
# the tail (which would make two different passwords equivalent).
MAX_PASSWORD_BYTES = 72
MIN_PASSWORD_LENGTH = 8


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """False for a wrong password, and for anything bcrypt cannot process.

    An over-long or malformed input is not a match, so treating it as a failed
    login is correct - and it keeps the response identical to a wrong password,
    which is what stops the endpoint leaking whether an account exists.
    """
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(user: User) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user.id),
        "username": user.username,
        "guest": user.is_guest,
        "iat": now,
        "exp": now + timedelta(days=settings.access_token_days),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


_CREDENTIALS_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Your session has expired. Please sign in again.",
    headers={"WWW-Authenticate": "Bearer"},
)


def _decode(token: str) -> dict:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError as exc:
        raise _CREDENTIALS_ERROR from exc


def current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session: Session = Depends(get_session),
) -> User:
    """Require a valid token. Use on everything that touches user data."""
    if credentials is None:
        raise _CREDENTIALS_ERROR

    payload = _decode(credentials.credentials)
    user = session.get(User, int(payload.get("sub", 0)))
    if user is None:
        raise _CREDENTIALS_ERROR

    # Cheap activity tracking; the dashboard shows it on the profile page.
    user.last_active = utcnow()
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def user_by_username(session: Session, username: str) -> User | None:
    return session.exec(select(User).where(User.username == username)).first()
