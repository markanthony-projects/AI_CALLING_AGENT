"""Password hashing for dashboard users.

Uses hashlib.scrypt from the standard library rather than pulling in passlib or
argon2-cffi. scrypt is a memory-hard KDF that OWASP lists as an acceptable choice, and
keeping it in the stdlib means no compiled wheel has to resolve on both the Windows dev
machines and the Linux image — a build failure here would block deploys over a login form.

Stored format is a single self-describing string:

    scrypt$<n>$<r>$<p>$<salt_b64>$<hash_b64>

The parameters travel with the hash so they can be raised later without invalidating
every existing password: verification reads the cost out of the stored value, and
needs_rehash tells the caller when to re-hash on the next successful login.
"""

import base64
import hashlib
import secrets
from typing import Tuple

# ~64 MiB and roughly 100ms on a droplet vCPU. Raising N is the lever; r and p are left at
# the values scrypt's author recommends for interactive logins.
_N = 2**16
_R = 8
_P = 1
_DKLEN = 32
_SALT_BYTES = 16
_PREFIX = "scrypt"

# scrypt needs about 128 * N * r bytes. OpenSSL's default maxmem is 32 MiB and would refuse
# the parameters above, so the requirement is stated explicitly with headroom.
_MAXMEM = 128 * _N * _R * 2


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _unb64(raw: str) -> bytes:
    return base64.b64decode(raw.encode("ascii"))


def _derive(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        dklen=_DKLEN,
        maxmem=128 * n * r * 2,
    )


def hash_password(password: str) -> str:
    if not password:
        raise ValueError("password must not be empty")
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = _derive(password, salt, _N, _R, _P)
    return f"{_PREFIX}${_N}${_R}${_P}${_b64(salt)}${_b64(digest)}"


def _parse(stored: str) -> Tuple[int, int, int, bytes, bytes]:
    scheme, n, r, p, salt, digest = stored.split("$")
    if scheme != _PREFIX:
        raise ValueError(f"unsupported password scheme {scheme!r}")
    return int(n), int(r), int(p), _unb64(salt), _unb64(digest)


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check. Any malformed stored value is a failed login, never a crash."""
    try:
        n, r, p, salt, expected = _parse(stored)
        candidate = _derive(password, salt, n, r, p)
    except (ValueError, TypeError, MemoryError):
        return False
    return secrets.compare_digest(candidate, expected)


def needs_rehash(stored: str) -> bool:
    """True when the stored hash was made with weaker parameters than we now use."""
    try:
        n, r, p, _, _ = _parse(stored)
    except (ValueError, TypeError):
        return True
    return (n, r, p) != (_N, _R, _P)
