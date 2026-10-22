import hashlib
import hmac

from fastapi import HTTPException, Request

from app.config import get_settings


def signature_is_valid(raw_body: bytes, header: str | None, secret: str) -> bool:
    if not header:
        return False
    # Accept both "abc123..." and "sha256=abc123...".
    received = header.removeprefix("sha256=").strip()
    expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    # compare_digest takes the same time wherever the strings differ, so an attacker
    # cannot guess the signature byte by byte from response timing.
    return hmac.compare_digest(received.encode(), expected.encode())


async def verify_signature(request: Request) -> bytes:
    """Dependency: returns the raw body if the signature is valid, else 401.

    The signature covers the exact bytes sent. We must read them before any JSON
    parsing, because parsing and re-serialising can change spacing/key order.
    """
    raw_body = await request.body()
    header = request.headers.get("X-Signature")
    if not signature_is_valid(raw_body, header, get_settings().webhook_secret):
        raise HTTPException(status_code=401, detail="Invalid signature")
    return raw_body
