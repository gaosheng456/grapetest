from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from .auth_store import ensure_default_user, register_user, verify_user_password
from .auth_tokens import create_token, verify_token

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    identifier: str
    password: str


class LoginRequest(BaseModel):
    identifier: str
    password: str


def require_user(authorization: Optional[str] = Header(default=None)) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="未登录")

    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="登录信息无效")

    token = parts[1]
    try:
        payload = verify_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="登录已失效")

    subject = str(payload.get("sub") or "").strip()
    if not subject:
        raise HTTPException(status_code=401, detail="登录信息无效")
    return subject


@router.on_event("startup")
def _init_default_user() -> None:
    ensure_default_user()


@router.post("/register")
def register(req: RegisterRequest) -> Dict[str, Any]:
    try:
        register_user(req.identifier, req.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}


@router.post("/login")
def login(req: LoginRequest) -> Dict[str, Any]:
    if not verify_user_password(req.identifier, req.password):
        raise HTTPException(status_code=401, detail="账号或密码错误")

    token = create_token(req.identifier)
    return {"token": token, "identifier": req.identifier}


@router.get("/me")
def me(user: str = Depends(require_user)) -> Dict[str, Any]:
    return {"identifier": user}
