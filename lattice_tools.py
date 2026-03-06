"""Lattice document tools for Claude — create, read, update, link, search docs."""

import json
import os
from typing import Any

import httpx

LATTICE_URL = os.getenv("LATTICE_URL", "http://localhost:8007/api/v1")


async def _lattice_get(path: str, auth_headers: dict, params: dict | None = None) -> Any:
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{LATTICE_URL}{path}", params=params, headers=auth_headers, timeout=30)
        if not r.is_success:
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:200]
            raise Exception(f"Lattice {r.status_code}: {detail}")
        return r.json()


async def _lattice_post(path: str, auth_headers: dict, body: dict | None = None) -> Any:
    headers = {"Content-Type": "application/json", **auth_headers}
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{LATTICE_URL}{path}", json=body or {}, headers=headers, timeout=30)
        if not r.is_success:
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:200]
            raise Exception(f"Lattice {r.status_code}: {detail}")
        return r.json()


async def _lattice_put(path: str, auth_headers: dict, body: dict | None = None) -> Any:
    headers = {"Content-Type": "application/json", **auth_headers}
    async with httpx.AsyncClient() as c:
        r = await c.put(f"{LATTICE_URL}{path}", json=body or {}, headers=headers, timeout=30)
        if not r.is_success:
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:200]
            raise Exception(f"Lattice {r.status_code}: {detail}")
        return r.json()


async def _lattice_delete(path: str, auth_headers: dict) -> str:
    async with httpx.AsyncClient() as c:
        r = await c.delete(f"{LATTICE_URL}{path}", headers=auth_headers, timeout=30)
        if not r.is_success:
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:200]
            raise Exception(f"Lattice {r.status_code}: {detail}")
        return "ok"


def _fmt(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)


def register_lattice_tools(mcp, auth_headers_fn):
    """Register Lattice document tools on the MCP server."""

    @mcp.tool()
    async def create_document(
        title: str,
        content: str = "",
        doc_type: str = "note",
        entity_type: str = "",
        entity_id: str = "",
        is_context: bool = False,
    ) -> str:
        """Create a Lattice document (note, spec, guide, etc). Optionally link to a Board entity.

        Args:
            title: Document title
            content: Markdown content
            doc_type: Type — note, post, spec, guide (default: note)
            entity_type: Optional — link to op, task, or project
            entity_id: Optional — UUID of the entity to link to
            is_context: If true, Claude should auto-read this doc when working on the entity
        """
        headers = await auth_headers_fn()
        body = {"title": title, "content": content, "doc_type": doc_type}
        doc = await _lattice_post("/admin/posts", headers, body)

        result = f"Created document: {doc['title']} (id: {doc['id']})"

        if entity_type and entity_id:
            link_body = {
                "document_id": doc["id"],
                "entity_type": entity_type,
                "entity_id": entity_id,
                "is_context": is_context,
            }
            await _lattice_post("/admin/links", headers, link_body)
            ctx = " (context)" if is_context else ""
            result += f"\nLinked to {entity_type}/{entity_id}{ctx}"

        return result

    @mcp.tool()
    async def read_document(document_id: str) -> str:
        """Read a Lattice document's full content.

        Args:
            document_id: UUID of the document
        """
        headers = await auth_headers_fn()
        doc = await _lattice_get(f"/admin/posts/{document_id}", headers)
        content = doc.get("content", "")
        truncated = ""
        if len(content) > 5000:
            content = content[:5000]
            truncated = "\n\n--- Content truncated at 5000 chars ---"

        return f"# {doc['title']}\n\nType: {doc['doc_type']} | Status: {doc['status']}\n\n{content}{truncated}"

    @mcp.tool()
    async def update_document(
        document_id: str,
        title: str = "",
        content: str = "",
    ) -> str:
        """Update a Lattice document's title and/or content.

        Args:
            document_id: UUID of the document
            title: New title (leave empty to keep current)
            content: New markdown content (leave empty to keep current)
        """
        headers = await auth_headers_fn()
        body: dict[str, Any] = {}
        if title:
            body["title"] = title
        if content:
            body["content"] = content
        if not body:
            return "Nothing to update — provide title or content."

        doc = await _lattice_put(f"/admin/posts/{document_id}", headers, body)
        return f"Updated document: {doc['title']} (id: {doc['id']})"

    @mcp.tool()
    async def link_document(
        document_id: str,
        entity_type: str,
        entity_id: str,
        is_context: bool = False,
    ) -> str:
        """Link a Lattice document to a Board entity (op, task, or project).

        Args:
            document_id: UUID of the document
            entity_type: op, task, or project
            entity_id: UUID of the Board entity
            is_context: If true, Claude should auto-read this doc when working on the entity
        """
        headers = await auth_headers_fn()
        body = {
            "document_id": document_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "is_context": is_context,
        }
        link = await _lattice_post("/admin/links", headers, body)
        ctx = " (context)" if is_context else ""
        return f"Linked document {document_id} to {entity_type}/{entity_id}{ctx} (link_id: {link['id']})"

    @mcp.tool()
    async def list_linked_documents(
        entity_type: str,
        entity_id: str,
    ) -> str:
        """List documents linked to a Board entity.

        Args:
            entity_type: op, task, or project
            entity_id: UUID of the entity
        """
        headers = await auth_headers_fn()
        data = await _lattice_get(f"/links/by-entity/{entity_type}/{entity_id}", headers)
        docs = data.get("documents", [])
        if not docs:
            return f"No documents linked to {entity_type}/{entity_id}"

        lines = [f"Documents linked to {entity_type}/{entity_id}:"]
        for d in docs:
            ctx = " [CONTEXT]" if d.get("is_context") else ""
            lines.append(f"  - [{d['doc_type']}] {d['title']}{ctx} (id: {d['id']})")
        return "\n".join(lines)

    @mcp.tool()
    async def search_documents(
        query: str = "",
        doc_type: str = "",
        limit: int = 20,
    ) -> str:
        """Search your Lattice documents by title.

        Args:
            query: Title search text
            doc_type: Filter by type — note, post, spec, guide
            limit: Max results (default 20)
        """
        headers = await auth_headers_fn()
        params: dict[str, Any] = {"limit": limit}
        if query:
            params["search"] = query
        if doc_type:
            params["doc_type"] = doc_type

        docs = await _lattice_get("/admin/posts", headers, params)
        if not docs:
            return "No documents found."

        lines = [f"Found {len(docs)} document(s):"]
        for d in docs:
            lines.append(f"  - [{d['doc_type']}] {d['title']} ({d['status']}) — id: {d['id']}")
        return "\n".join(lines)
