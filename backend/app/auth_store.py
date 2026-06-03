from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Tuple

_USERS_FILE = Path(__file__).resolve().with_name("password_store.json")
_LOCK = threading.Lock()

_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_PHONE_RE = re.compile(r"^1\d{10}$")

_PBKDF2_ITERS = 120_000


def _b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def _now_ts() -> int:
    return int(time.time())


def _load_store() -> Dict[str, Any]:
    if not _USERS_FILE.exists():
        return {"version": 1, "users": {}}
    data = json.loads(_USERS_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"version": 1, "users": {}}
    data.setdefault("version", 1)
    data.setdefault("users", {})
    if not isinstance(data["users"], dict):
        data["users"] = {}
    return data


def _save_store(store: Dict[str, Any]) -> None:
    tmp = _USERS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_USERS_FILE)


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERS)


def _make_user_record(password: str) -> Dict[str, Any]:
    salt = os.urandom(16)
    pwd_hash = _hash_password(password, salt)
    return {
        "salt_b64": _b64e(salt),
        "hash_b64": _b64e(pwd_hash),
        "iters": _PBKDF2_ITERS,
        "created_at": _now_ts(),
    }


def ensure_default_user() -> None:
    """确保默认账号存在：grape / 123。"""
    with _LOCK:
        store = _load_store()
        users = store["users"]
        if "grape" not in users:
            users["grape"] = _make_user_record("123")
            _save_store(store)


def is_valid_registration_identifier(identifier: str) -> Tuple[bool, str]:
    """注册只允许手机号或邮箱。默认账号 grape 不走注册校验。"""
    ident = (identifier or "").strip()
    if not ident:
        return False, "账号不能为空"
    if _EMAIL_RE.match(ident) or _PHONE_RE.match(ident):
        return True, ""
    return False, "账号必须是手机号(11位, 1开头)或邮箱"


def register_user(identifier: str, password: str) -> None:
    ident = (identifier or "").strip()
    pwd = password or ""

    ok, msg = is_valid_registration_identifier(ident)
    if not ok:
        raise ValueError(msg)
    if len(pwd) < 3:
        raise ValueError("密码长度至少 3 位")

    with _LOCK:
        store = _load_store()
        users = store["users"]
        if ident in users:
            raise ValueError("账号已存在")
        users[ident] = _make_user_record(pwd)
        _save_store(store)


def verify_user_password(identifier: str, password: str) -> bool:
    ident = (identifier or "").strip()
    pwd = password or ""
    if not ident or not pwd:
        return False

    with _LOCK:
        store = _load_store()
        users = store.get("users", {})
        rec = users.get(ident)
        if not isinstance(rec, dict):
            return False

        try:
            salt = _b64d(str(rec.get("salt_b64", "")))
            expected = _b64d(str(rec.get("hash_b64", "")))
            iters = int(rec.get("iters") or _PBKDF2_ITERS)
        except Exception:
            return False

    actual = hashlib.pbkdf2_hmac("sha256", pwd.encode("utf-8"), salt, iters)
    return hmac.compare_digest(actual, expected)
