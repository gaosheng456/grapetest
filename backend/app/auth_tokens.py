from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any, Dict


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("ascii"))


def _secret() -> bytes:
    """获取签名密钥。

    优先级：
    1) 环境变量 GRAPE_AUTH_SECRET
    2) 本地持久化文件 backend/app/auth_secret.txt（首次登录/签发 token 时自动生成）
    3) 兜底常量（仅在无法写入文件时使用）
    """

    env = (os.getenv("GRAPE_AUTH_SECRET") or "").strip()
    if env:
        return env.encode("utf-8")

    secret_file = Path(__file__).resolve().with_name("auth_secret.txt")
    try:
        if secret_file.exists():
            v = secret_file.read_text(encoding="utf-8").strip()
            if v:
                return v.encode("utf-8")

        v = secrets.token_urlsafe(48)
        secret_file.write_text(v, encoding="utf-8")
        return v.encode("utf-8")
    except Exception:
        return b"grape-auth-secret"


def create_token(subject: str, ttl_seconds: int = 24 * 3600) -> str:
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"sub": subject, "iat": now, "exp": now + int(ttl_seconds)}

    h = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    p = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    msg = f"{h}.{p}".encode("utf-8")
    sig = hmac.new(_secret(), msg, hashlib.sha256).digest()
    s = _b64url_encode(sig)
    return f"{h}.{p}.{s}"


def verify_token(token: str) -> Dict[str, Any]:
    parts = (token or "").split(".")
    if len(parts) != 3:
        raise ValueError("invalid token")

    h, p, s = parts
    msg = f"{h}.{p}".encode("utf-8")
    expected = hmac.new(_secret(), msg, hashlib.sha256).digest()
    actual = _b64url_decode(s)
    if not hmac.compare_digest(expected, actual):
        raise ValueError("bad signature")

    payload = json.loads(_b64url_decode(p).decode("utf-8"))
    exp = int(payload.get("exp") or 0)
    if exp and int(time.time()) > exp:
        raise ValueError("token expired")
    return payload
