import os
import time
import hmac
import hashlib
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel

logger = logging.getLogger("app.routers.admin_auth")

router = APIRouter(prefix="/admin/auth", tags=["Admin Authentication"])

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "jatayu2026")
ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "jatayu_cyber_secret_key_98765")

class LoginRequest(BaseModel):
    username: str
    password: str

def generate_admin_token(username: str) -> str:
    timestamp = str(int(time.time()))
    payload = f"{username}:{timestamp}"
    signature = hmac.new(ADMIN_SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}:{signature}"

def verify_admin_token(token: str) -> bool:
    try:
        parts = token.split(":")
        if len(parts) != 3:
            return False
        username, timestamp_str, signature = parts
        
        # Check token expiration (24 hours = 86400 seconds)
        ts = int(timestamp_str)
        if time.time() - ts > 86400:
            return False
            
        payload = f"{username}:{timestamp_str}"
        expected_sig = hmac.new(ADMIN_SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(signature, expected_sig)
    except Exception as exc:
        logger.warning(f"[AdminAuth] Token verification failed: {exc}")
        return False

@router.post("/login")
async def admin_login(payload: LoginRequest):
    """Authenticates admin credentials and returns a secure bearer access token."""
    username = payload.username.strip()
    password = payload.password.strip()

    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        token = generate_admin_token(username)
        logger.info(f"[AdminAuth] Successful login for admin user '{username}'")
        return {
            "status": "success",
            "access_token": token,
            "token_type": "bearer",
            "expires_in": 86400,
            "user": {
                "username": username,
                "role": "administrator"
            }
        }
    else:
        logger.warning(f"[AdminAuth] Invalid login attempt for username '{username}'")
        raise HTTPException(status_code=401, detail="Invalid admin username or password.")

@router.get("/verify")
async def verify_token(authorization: Optional[str] = Header(None)):
    """Verifies whether the provided Bearer token is valid and active."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing.")

    token = authorization.replace("Bearer ", "").strip()
    if verify_admin_token(token):
        return {"status": "success", "authenticated": True, "username": ADMIN_USERNAME}
    else:
        raise HTTPException(status_code=401, detail="Invalid or expired admin session token.")
