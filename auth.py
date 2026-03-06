"""Shared Keycloak auth for MCP servers."""

import os
import time
from typing import Any, Callable, Coroutine

import httpx

_token_caches: dict[str, dict[str, Any]] = {}


async def get_token(
    kc_url: str, realm: str, client_id: str, username: str, password: str,
) -> str:
    """Return a valid Bearer token via Keycloak password grant."""
    cache_key = f"{kc_url}:{realm}:{username}"
    cache = _token_caches.setdefault(cache_key, {"access_token": "", "expires_at": 0})

    if cache["access_token"] and time.time() < cache["expires_at"] - 30:
        return cache["access_token"]

    token_url = f"{kc_url}/realms/{realm}/protocol/openid-connect/token"
    async with httpx.AsyncClient() as c:
        r = await c.post(token_url, data={
            "grant_type": "password",
            "client_id": client_id,
            "username": username,
            "password": password,
        }, timeout=10)
        r.raise_for_status()
        data = r.json()

    cache["access_token"] = data["access_token"]
    cache["expires_at"] = time.time() + data.get("expires_in", 300)
    return data["access_token"]


def make_auth_headers_fn(
    static_token_var: str = "BOARD_TOKEN",
) -> Callable[[], Coroutine[Any, Any, dict[str, str]]]:
    """Create an async auth_headers() function from environment variables.

    Reads: KEYCLOAK_URL, KEYCLOAK_REALM, KEYCLOAK_CLIENT_ID,
           KEYCLOAK_USERNAME, KEYCLOAK_PASSWORD
    Falls back to a static token from the env var named by static_token_var.
    """
    kc_url = os.getenv("KEYCLOAK_URL", "")
    kc_realm = os.getenv("KEYCLOAK_REALM", "opscraft")
    kc_client_id = os.getenv("KEYCLOAK_CLIENT_ID", "mr-fusion-frontend")
    kc_username = os.getenv("KEYCLOAK_USERNAME", "")
    kc_password = os.getenv("KEYCLOAK_PASSWORD", "")
    static_token = os.getenv(static_token_var, "")

    async def _auth_headers() -> dict[str, str]:
        if kc_url and kc_username:
            token = await get_token(kc_url, kc_realm, kc_client_id, kc_username, kc_password)
        else:
            token = static_token
        return {"Authorization": f"Bearer {token}"} if token else {}

    return _auth_headers
