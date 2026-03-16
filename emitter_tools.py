"""Emitter (CMS) blog tools for Claude — create, manage, publish posts."""

import json
import os
from typing import Any

import httpx

CMS_URL = os.getenv("CMS_URL", "http://localhost:8009/api/v1")


async def _cms_get(path: str, auth_headers: dict, params: dict | None = None) -> Any:
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{CMS_URL}{path}", params=params, headers=auth_headers, timeout=30)
        if not r.is_success:
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:200]
            raise Exception(f"CMS {r.status_code}: {detail}")
        return r.json()


async def _cms_post(path: str, auth_headers: dict, body: dict | None = None) -> Any:
    headers = {"Content-Type": "application/json", **auth_headers}
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{CMS_URL}{path}", json=body or {}, headers=headers, timeout=30)
        if not r.is_success:
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:200]
            raise Exception(f"CMS {r.status_code}: {detail}")
        return r.json()


async def _cms_put(path: str, auth_headers: dict, body: dict | None = None) -> Any:
    headers = {"Content-Type": "application/json", **auth_headers}
    async with httpx.AsyncClient() as c:
        r = await c.put(f"{CMS_URL}{path}", json=body or {}, headers=headers, timeout=30)
        if not r.is_success:
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:200]
            raise Exception(f"CMS {r.status_code}: {detail}")
        return r.json()


async def _cms_delete(path: str, auth_headers: dict) -> str:
    async with httpx.AsyncClient() as c:
        r = await c.delete(f"{CMS_URL}{path}", headers=auth_headers, timeout=30)
        if not r.is_success:
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:200]
            raise Exception(f"CMS {r.status_code}: {detail}")
        return "ok"


def register_emitter_tools(mcp, auth_headers_fn):
    """Register Emitter blog tools on the MCP server."""

    @mcp.tool()
    async def create_blog_post(
        title: str,
        content: str = "",
        excerpt: str = "",
        featured_image: str = "",
        tags: list[str] | None = None,
        category_ids: list[str] | None = None,
    ) -> str:
        """Create a draft blog post on the Emitter CMS.

        Args:
            title: Post title
            content: Markdown content
            excerpt: Short excerpt/summary
            featured_image: URL of the featured/hero image
            tags: Optional list of tag strings
            category_ids: Optional list of category UUIDs to assign
        """
        headers = await auth_headers_fn()
        body: dict[str, Any] = {"title": title}
        if content:
            body["content"] = content
        if excerpt:
            body["excerpt"] = excerpt
        if featured_image:
            body["featured_image"] = featured_image
        if tags:
            body["tags"] = tags
        if category_ids:
            body["category_ids"] = category_ids

        post = await _cms_post("/admin/posts", headers, body)
        cats = post.get("categories", [])
        cat_str = f" in {', '.join(c['name'] for c in cats)}" if cats else ""
        return f"Created draft: {post['title']}{cat_str} (id: {post['id']})"

    @mcp.tool()
    async def list_blog_posts(status: str = "") -> str:
        """List blog posts, optionally filtered by status (draft, published).

        Args:
            status: Filter by status — draft or published (empty for all)
        """
        headers = await auth_headers_fn()
        params: dict[str, Any] = {}
        if status:
            params["status"] = status

        posts = await _cms_get("/admin/posts", headers, params)
        if not posts:
            return "No blog posts found."

        lines = [f"Found {len(posts)} post(s):"]
        for p in posts:
            cats = ", ".join(c["name"] for c in p.get("categories", []))
            cat_str = f" [{cats}]" if cats else ""
            lines.append(f"  - [{p['status']}] {p['title']}{cat_str} — id: {p['id']}")
        return "\n".join(lines)

    @mcp.tool()
    async def get_blog_post(post_id: str) -> str:
        """Read full blog post content and metadata.

        Args:
            post_id: UUID of the post
        """
        headers = await auth_headers_fn()
        post = await _cms_get(f"/admin/posts/{post_id}", headers)

        content = post.get("content", "")
        truncated = ""
        if len(content) > 5000:
            content = content[:5000]
            truncated = "\n\n--- Content truncated at 5000 chars ---"

        cats = ", ".join(c["name"] for c in post.get("categories", []))
        tags = ", ".join(post.get("tags", []))

        lines = [f"# {post['title']}"]
        lines.append(f"Status: {post['status']} | Slug: {post['slug']}")
        if cats:
            lines.append(f"Categories: {cats}")
        if tags:
            lines.append(f"Tags: {tags}")
        if post.get("excerpt"):
            lines.append(f"Excerpt: {post['excerpt']}")
        if post.get("published_at"):
            lines.append(f"Published: {post['published_at']}")
        lines.append(f"\n{content}{truncated}")
        return "\n".join(lines)

    @mcp.tool()
    async def update_blog_post(
        post_id: str,
        title: str = "",
        content: str = "",
        excerpt: str = "",
        featured_image: str = "",
        tags: list[str] | None = None,
        category_ids: list[str] | None = None,
    ) -> str:
        """Update a blog post's title, content, excerpt, tags, or categories.

        Args:
            post_id: UUID of the post
            title: New title (leave empty to keep current)
            content: New markdown content (leave empty to keep current)
            excerpt: New excerpt (leave empty to keep current)
            featured_image: URL of the featured/hero image (leave empty to keep current)
            tags: New tags list (omit to keep current)
            category_ids: New category UUIDs (omit to keep current)
        """
        headers = await auth_headers_fn()
        body: dict[str, Any] = {}
        if title:
            body["title"] = title
        if content:
            body["content"] = content
        if excerpt:
            body["excerpt"] = excerpt
        if featured_image:
            body["featured_image"] = featured_image
        if tags is not None:
            body["tags"] = tags
        if category_ids is not None:
            body["category_ids"] = category_ids
        if not body:
            return "Nothing to update — provide at least one field."

        post = await _cms_put(f"/admin/posts/{post_id}", headers, body)
        return f"Updated: {post['title']} (id: {post['id']})"

    @mcp.tool()
    async def delete_blog_post(post_id: str) -> str:
        """Delete a draft blog post. Published posts must be unpublished first.

        Args:
            post_id: UUID of the post
        """
        headers = await auth_headers_fn()
        await _cms_delete(f"/admin/posts/{post_id}", headers)
        return f"Deleted post {post_id}"

    @mcp.tool()
    async def publish_blog_post(post_id: str) -> str:
        """Publish a draft blog post — makes it live on the site.

        Args:
            post_id: UUID of the post
        """
        headers = await auth_headers_fn()
        post = await _cms_post(f"/admin/posts/{post_id}/publish", headers)
        return f"Published: {post['title']} (slug: {post['slug']})"

    @mcp.tool()
    async def unpublish_blog_post(post_id: str) -> str:
        """Unpublish a blog post — reverts it to draft status.

        Args:
            post_id: UUID of the post
        """
        headers = await auth_headers_fn()
        post = await _cms_post(f"/admin/posts/{post_id}/unpublish", headers)
        return f"Unpublished: {post['title']} — now draft"

    @mcp.tool()
    async def publish_blog_newsletter(post_id: str) -> str:
        """Publish a blog post AND push it as a newsletter to Signal (email).

        This publishes the post and sends it to all newsletter subscribers.

        Args:
            post_id: UUID of the post
        """
        headers = await auth_headers_fn()
        post = await _cms_post(f"/admin/posts/{post_id}/publish-newsletter", headers)
        return f"Published + newsletter sent: {post['title']}"

    @mcp.tool()
    async def list_blog_categories() -> str:
        """List all blog categories with post counts."""
        headers = await auth_headers_fn()
        cats = await _cms_get("/admin/categories", headers)
        if not cats:
            return "No categories found."

        lines = [f"Found {len(cats)} category/ies:"]
        for c in cats:
            count = c.get("post_count", 0)
            desc = f" — {c['description']}" if c.get("description") else ""
            lines.append(f"  - {c['name']} ({count} posts){desc} — id: {c['id']}")
        return "\n".join(lines)

    @mcp.tool()
    async def create_blog_category(
        name: str,
        description: str = "",
        position: int = 0,
    ) -> str:
        """Create a new blog category.

        Args:
            name: Category name
            description: Optional description
            position: Sort order (0 = first)
        """
        headers = await auth_headers_fn()
        body: dict[str, Any] = {"name": name, "position": position}
        if description:
            body["description"] = description

        cat = await _cms_post("/admin/categories", headers, body)
        return f"Created category: {cat['name']} (slug: {cat['slug']}, id: {cat['id']})"

    @mcp.tool()
    async def upload_post_image(
        post_id: str,
        image_url: str,
        set_featured: bool = False,
    ) -> str:
        """Download an image from a URL and attach it to a blog post.

        The image is downloaded, resized (max 1200px wide), and stored in S3/MinIO.
        Returns the new image URL served by the CMS.

        Args:
            post_id: UUID of the post to attach the image to
            image_url: URL of the image to download (e.g. from Ghost)
            set_featured: If true, also sets this as the post's featured/hero image
        """
        headers = await auth_headers_fn()
        body = {"url": image_url, "set_featured": set_featured}
        result = await _cms_post(f"/admin/posts/{post_id}/images/from-url", headers, body)
        url = result.get("url", "")
        size = result.get("size_bytes", 0)
        feat = " (set as featured)" if set_featured else ""
        return f"Uploaded: {result.get('filename', '')} -> {url} ({size} bytes){feat}"

    @mcp.tool()
    async def list_blog_subscribers(status: str = "") -> str:
        """List blog newsletter subscribers.

        Args:
            status: Filter by status — active or unsubscribed (empty for all)
        """
        headers = await auth_headers_fn()
        params: dict[str, Any] = {}
        if status:
            params["status"] = status

        subs = await _cms_get("/admin/subscribers", headers, params)
        if not subs:
            return "No subscribers found."

        lines = [f"Found {len(subs)} subscriber(s):"]
        for s in subs:
            name = f" ({s['name']})" if s.get("name") else ""
            lines.append(f"  - {s['email']}{name} [{s['status']}] — id: {s['id']}")
        return "\n".join(lines)
