"""Browser-session protection for public QFind API routes."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import Cookie, HTTPException, Request, Response

SESSION_COOKIE_NAME = "qfind_session"
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(60 * 60 * 24)))
SESSION_RATE_LIMIT_PER_MINUTE = int(os.getenv("SESSION_RATE_LIMIT_PER_MINUTE", "5"))
SESSION_RATE_LIMIT_PER_DAY = int(os.getenv("SESSION_RATE_LIMIT_PER_DAY", "50"))

_FALLBACK_SECRET = secrets.token_urlsafe(32)


@dataclass
class SlidingWindowRateLimiter:
    """Small in-memory limiter for demo-scale Cloud Run instances."""

    per_minute: int = SESSION_RATE_LIMIT_PER_MINUTE
    per_day: int = SESSION_RATE_LIMIT_PER_DAY
    _events: dict[str, list[float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def check(self, key: str, *, now: float | None = None) -> tuple[bool, int]:
        current_time = time.time() if now is None else now
        day_start = current_time - 86400
        minute_start = current_time - 60
        with self._lock:
            events = [
                event_time
                for event_time in self._events.get(key, [])
                if event_time >= day_start
            ]
            minute_count = sum(1 for event_time in events if event_time >= minute_start)
            if minute_count >= self.per_minute:
                retry_after = max(1, int(60 - (current_time - min(
                    event_time for event_time in events if event_time >= minute_start
                ))))
                self._events[key] = events
                return False, retry_after
            if len(events) >= self.per_day:
                retry_after = max(1, int(86400 - (current_time - min(events))))
                self._events[key] = events
                return False, retry_after
            events.append(current_time)
            self._events[key] = events
            return True, 0


def _signing_secret() -> str:
    return os.getenv("SESSION_SIGNING_SECRET") or _FALLBACK_SECRET


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def _signature(message: str) -> str:
    digest = hmac.new(
        _signing_secret().encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return _b64url(digest)


def create_session_token(*, now: float | None = None) -> str:
    current_time = int(time.time() if now is None else now)
    session_id = secrets.token_urlsafe(18)
    body = _b64url(f"{session_id}:{current_time}".encode())
    return f"{body}.{_signature(body)}"


def verify_session_token(token: str | None, *, now: float | None = None) -> str:
    if not token or "." not in token:
        raise HTTPException(
            status_code=401,
            detail="Browser session required. Load /api/session before calling this endpoint.",
        )
    body, supplied_signature = token.rsplit(".", 1)
    expected_signature = _signature(body)
    if not hmac.compare_digest(supplied_signature, expected_signature):
        raise HTTPException(status_code=401, detail="Invalid browser session.")
    try:
        decoded = _b64url_decode(body).decode("utf-8")
        session_id, issued_at_text = decoded.rsplit(":", 1)
        issued_at = int(issued_at_text)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid browser session.") from exc
    current_time = int(time.time() if now is None else now)
    if current_time - issued_at > SESSION_TTL_SECONDS:
        raise HTTPException(
            status_code=401,
            detail="Browser session expired. Reload the page and try again.",
        )
    return session_id


def _origin_host(value: str) -> str:
    parsed = urlsplit(value)
    return parsed.netloc.lower()


def validate_origin(request: Request) -> None:
    if request.method.upper() not in {"POST", "PUT", "PATCH", "DELETE"}:
        return
    origin = request.headers.get("origin")
    if not origin:
        raise HTTPException(status_code=401, detail="Browser origin header required.")
    allowed_origin = (os.getenv("ALLOWED_ORIGIN") or "").strip()
    if allowed_origin:
        if origin.rstrip("/") != allowed_origin.rstrip("/"):
            raise HTTPException(status_code=401, detail="Request origin is not allowed.")
        return
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    if _origin_host(origin) != host.lower():
        raise HTTPException(status_code=401, detail="Request origin is not allowed.")


def client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    return request.client.host if request.client else "unknown"


def issue_session_cookie(response: Response) -> dict[str, object]:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=create_session_token(),
        httponly=True,
        secure=os.getenv("SESSION_COOKIE_SECURE", "false").strip().lower()
        in {"1", "true", "yes", "on"},
        samesite="lax",
        max_age=SESSION_TTL_SECONDS,
        path="/",
    )
    return {"ok": True, "cookie": SESSION_COOKIE_NAME, "max_age": SESSION_TTL_SECONDS}


def make_session_dependency(limiter: SlidingWindowRateLimiter):
    def require_browser_session(
        request: Request,
        response: Response,
        token: Annotated[str | None, Cookie(alias=SESSION_COOKIE_NAME)] = None,
    ) -> str:
        validate_origin(request)
        session_id = verify_session_token(token)
        key = f"{session_id}:{client_ip(request)}"
        allowed, retry_after = limiter.check(key)
        if not allowed:
            response.headers["Retry-After"] = str(retry_after)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )
        return session_id

    return require_browser_session
