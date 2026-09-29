"""API key authentication middleware for aiohttp."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass, field, asdict

from aiohttp import web

from ainode.core.config import AINODE_HOME


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


AUTH_FILE = AINODE_HOME / "auth.json"

SKIP_PATHS: set[str] = {"/", "/onboarding", "/api/health"}
SKIP_PREFIXES: tuple[str, ...] = ("/static/", "/api/onboarding/")


@dataclass
class AuthConfig:
    enabled: bool = False
    api_keys: list[dict] = field(default_factory=list)
    # Each key entry: {"id": "<short-id>", "key": "<hex>"}

    # -- persistence ----------------------------------------------------------

    def save(self) -> None:
        AINODE_HOME.mkdir(parents=True, exist_ok=True)
        AUTH_FILE.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls) -> "AuthConfig":
        if AUTH_FILE.exists():
            data = json.loads(AUTH_FILE.read_text())
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        return cls()

    def generate_key(self, name: str = "") -> dict:
        """Mint a key. The plaintext is returned ONCE and never stored.

        ``name`` is a label so an operator can tell keys apart when revoking
        one — "anna's laptop" rather than a hex id. Optional and free-form.
        """
        key = secrets.token_hex(16)
        key_id = secrets.token_hex(4)
        key_hash = _hash_key(key)
        self.api_keys.append({
            "id": key_id,
            "key_hash": key_hash,
            "name": str(name or "").strip(),
            "created_at": time.time(),
        })
        self.save()
        return {"id": key_id, "key": key, "name": str(name or "").strip()}

    def revoke_key(self, key_id: str) -> bool:
        before = len(self.api_keys)
        self.api_keys = [k for k in self.api_keys if k["id"] != key_id]
        if len(self.api_keys) != before:
            self.save()
            return True
        return False

    def validate_token(self, token: str) -> bool:
        token_hash = _hash_key(token)
        for entry in self.api_keys:
            stored = entry.get("key_hash") or entry.get("key", "")
            if hmac.compare_digest(token_hash, stored):
                return True
        return False

    def enable(self) -> dict:
        self.enabled = True
        if not self.api_keys:
            entry = self.generate_key()
        else:
            entry = self.api_keys[0]
            self.save()
        return entry

    def disable(self) -> None:
        self.enabled = False
        self.save()


def _should_skip(request: web.Request) -> bool:
    path = request.path
    if path in SKIP_PATHS:
        return True
    if path.startswith(SKIP_PREFIXES):
        return True
    if request.method == "GET" and path.startswith("/api/onboarding"):
        return True
    return False


@web.middleware
async def auth_middleware(request: web.Request, handler):
    auth_cfg: AuthConfig | None = request.app.get("auth_config")
    if auth_cfg is None or not auth_cfg.enabled:
        return await handler(request)
    if _should_skip(request):
        return await handler(request)
    # API keys govern the OpenAI-compatible proxy and nothing else. They used
    # to cover every route, and the dashboard never sends one — so switching
    # them on returned 401 to every request the UI made, the switch to turn
    # them off included, and the only way back was editing auth.json on the
    # host. The UI and /api/* are behind the web sign-in now
    # (auth/web_routes.py), which also accepts a valid key for automation.
    if not request.path.startswith("/v1/"):
        return await handler(request)
    # The dashboard's own chat and model list call /v1/* too, from a browser
    # that is signed in but holds no API key. A valid web session counts.
    web_login = request.app.get("web_login")
    if web_login is not None:
        from ainode.auth.web_login import COOKIE_NAME

        try:
            if web_login.check(request.cookies.get(COOKIE_NAME, "")):
                return await handler(request)
        except Exception:
            pass
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return web.json_response(
            {"error": {"message": "Missing or invalid Authorization header", "type": "auth_error"}},
            status=401,
        )
    token = auth_header[7:]
    if not auth_cfg.validate_token(token):
        return web.json_response(
            {"error": {"message": "Invalid API key", "type": "auth_error"}},
            status=401,
        )
    return await handler(request)
