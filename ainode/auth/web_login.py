"""The password in front of the web UI.

Before this, anything on the LAN could reach every ``/api/*`` route without
signing in — update the node from git, delete a model, rewrite a secret, start
a container. The container holds the host's docker socket and the operator's
SSH keys, so "can call the API" was "is root on all three nodes".

The UI stays reachable from the LAN, as asked. What changes is that reaching it
now means signing in first. Out of scope, on purpose:

* ``/v1/*`` — the OpenAI-compatible proxy. Its clients (Open WebUI, opencode)
  cannot sign in to a web page. It keeps the optional API keys it already had
  (auth/middleware.py), which now apply to it alone.
* calls from the node itself (loopback) — the CLI, the update script, the
  diagnostics script.
* calls from the other nodes — they present the cluster key
  (auth/cluster_key.py).

Storage is ``~/.ainode/web-auth.json``: the user, a salted scrypt hash, and the
secret that signs session cookies. A fresh install starts as admin/admin and
says so on every page until it is changed. Changing the password rotates the
signing secret, which signs every other browser out.

Sessions are stateless signed cookies rather than a table, so they survive a
restart of the container — having to sign in again after every update would be
the kind of friction that gets a login switched off.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["WebLogin", "COOKIE_NAME", "SESSION_SECONDS", "DEFAULT_USER",
           "DEFAULT_PASSWORD", "MIN_PASSWORD_LENGTH", "LoginThrottle"]

COOKIE_NAME = "ainode_session"
#: Thirty days. Long enough that a node you look at weekly does not ask every
#: time; short enough that a stolen cookie expires on its own.
SESSION_SECONDS = 30 * 24 * 3600
DEFAULT_USER = "admin"
DEFAULT_PASSWORD = "admin"
#: For a NEW password. The default is shorter than this, which is part of why
#: the UI will not stop mentioning it.
MIN_PASSWORD_LENGTH = 8

# scrypt at these costs takes ~50 ms on a GB10 core: nothing for a login,
# a lot for someone trying a word list against a copied file.
_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1, "dklen": 32}


def _hash(password: str, salt: bytes) -> str:
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt,
                            maxmem=64 * 1024 * 1024, **_SCRYPT)
    return digest.hex()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


@dataclass
class _Stored:
    user: str = DEFAULT_USER
    salt: str = ""
    password_hash: str = ""
    secret: str = ""
    is_default: bool = True
    changed_at: float = 0.0


class WebLogin:
    """Credentials and sessions for one node."""

    def __init__(self, path: Optional[Path] = None) -> None:
        if path is None:
            from ainode.core.config import AINODE_HOME

            path = Path(AINODE_HOME) / "web-auth.json"
        self.path = Path(path)
        self._lock = threading.Lock()
        self._unsaved = False
        self._stored = self._load()

    # -- storage ------------------------------------------------------------

    def _load(self) -> _Stored:
        try:
            blob = json.loads(self.path.read_text())
            stored = _Stored(**{k: v for k, v in blob.items()
                                if k in _Stored.__dataclass_fields__})
            if stored.salt and stored.password_hash and stored.secret:
                return stored
            logger.warning("%s is incomplete; resetting the web login to "
                           "its defaults", self.path)
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError):
            logger.warning("could not read %s; resetting the web login to "
                           "its defaults", self.path, exc_info=True)
        # Not written yet: a node nobody has signed in to should not grow a
        # credentials file (nor should a test suite write one into the real
        # home). The first sign-in persists it, so its session secret survives
        # a restart from then on.
        self._unsaved = True
        return self._fresh(DEFAULT_USER, DEFAULT_PASSWORD, is_default=True)

    @staticmethod
    def _fresh(user: str, password: str, *, is_default: bool) -> _Stored:
        salt = os.urandom(16)
        return _Stored(user=user, salt=salt.hex(),
                       password_hash=_hash(password, salt),
                       secret=secrets.token_hex(32), is_default=is_default,
                       changed_at=time.time())

    def _save(self, stored: _Stored) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as sink:
                json.dump(asdict(stored), sink, indent=2)
            tmp.replace(self.path)
        except OSError:
            logger.exception("could not save the web login to %s", self.path)

    # -- questions ----------------------------------------------------------

    @property
    def user(self) -> str:
        return self._stored.user

    @property
    def is_default(self) -> bool:
        return bool(self._stored.is_default)

    def verify(self, user: str, password: str) -> bool:
        stored = self._stored
        try:
            candidate = _hash(str(password or ""), bytes.fromhex(stored.salt))
        except (ValueError, MemoryError):
            return False
        # Both compared in constant time, and both always compared, so the
        # answer takes as long for a wrong user as for a wrong password.
        user_ok = hmac.compare_digest(str(user or "").encode(),
                                      stored.user.encode())
        pass_ok = hmac.compare_digest(candidate, stored.password_hash)
        return user_ok and pass_ok

    # -- changes ------------------------------------------------------------

    def set_password(self, current: str, new: str,
                     user: Optional[str] = None) -> Tuple[bool, str]:
        """(changed, why not). Signs every other session out."""
        if not self.verify(self._stored.user, current):
            return False, "The current password is not right."
        problem = password_problem(new)
        if problem:
            return False, problem
        new_user = (user or self._stored.user).strip() or self._stored.user
        with self._lock:
            self._stored = self._fresh(new_user, new, is_default=False)
            self._save(self._stored)
            self._unsaved = False
        return True, ""

    def export_hash(self) -> Dict[str, str]:
        """What another node needs to accept the same password. Never the
        password itself, and never the session secret — each node signs its
        own cookies."""
        return {"user": self._stored.user, "salt": self._stored.salt,
                "password_hash": self._stored.password_hash,
                "is_default": self._stored.is_default}

    def import_hash(self, blob: Dict[str, object]) -> bool:
        """Adopt another node's credentials. For the cluster fan-out only."""
        try:
            user = str(blob["user"])
            salt = str(blob["salt"])
            digest = str(blob["password_hash"])
            bytes.fromhex(salt)
            bytes.fromhex(digest)
        except (KeyError, ValueError, TypeError):
            return False
        with self._lock:
            self._stored = _Stored(user=user, salt=salt, password_hash=digest,
                                   secret=secrets.token_hex(32),
                                   is_default=bool(blob.get("is_default")),
                                   changed_at=time.time())
            self._save(self._stored)
            self._unsaved = False
        return True

    # -- sessions -----------------------------------------------------------

    def issue(self, now: Optional[float] = None) -> str:
        if self._unsaved:
            with self._lock:
                self._save(self._stored)
                self._unsaved = False
        expires = int((time.time() if now is None else now) + SESSION_SECONDS)
        body = f"{self._stored.user}|{expires}|{secrets.token_hex(8)}"
        mac = hmac.new(bytes.fromhex(self._stored.secret), body.encode(),
                       hashlib.sha256).digest()
        return f"{_b64(body.encode())}.{_b64(mac)}"

    def check(self, token: str, now: Optional[float] = None) -> bool:
        if not token or "." not in token:
            return False
        try:
            body_part, mac_part = token.split(".", 1)
            body = _unb64(body_part)
            mac = _unb64(mac_part)
            expected = hmac.new(bytes.fromhex(self._stored.secret), body,
                                hashlib.sha256).digest()
            if not hmac.compare_digest(mac, expected):
                return False
            user, expires, _nonce = body.decode().split("|", 2)
        except (ValueError, UnicodeDecodeError):
            return False
        if user != self._stored.user:
            return False
        return int(expires) > (time.time() if now is None else now)


def password_problem(password: str) -> str:
    """"" or what is wrong with a new password."""
    password = str(password or "")
    if len(password) < MIN_PASSWORD_LENGTH:
        return (f"Use at least {MIN_PASSWORD_LENGTH} characters.")
    if password == DEFAULT_PASSWORD:
        return "That is the default password."
    return ""


@dataclass
class LoginThrottle:
    """Failed sign-ins per address, in a sliding window.

    In memory and per node: this is a brake on guessing, not an audit trail.
    """

    limit: int = 10
    window: float = 300.0
    _failures: Dict[str, List[float]] = field(default_factory=dict)

    def blocked(self, address: str, now: Optional[float] = None) -> float:
        """Seconds until ``address`` may try again; 0 when it may now."""
        now = time.time() if now is None else now
        recent = [t for t in self._failures.get(address, [])
                  if now - t < self.window]
        self._failures[address] = recent
        if len(recent) < self.limit:
            return 0.0
        return max(0.0, self.window - (now - recent[0]))

    def failed(self, address: str, now: Optional[float] = None) -> None:
        self._failures.setdefault(address, []).append(time.time() if now is None else now)

    def succeeded(self, address: str) -> None:
        self._failures.pop(address, None)
