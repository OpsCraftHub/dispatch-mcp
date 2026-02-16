"""Dispatch — MCP server for the OpsCraft Board service.

Exposes board CRUD as MCP tools so Claude Code / Claude Desktop
can read and manage projects, Ops, and packets.
"""

import json
import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

BOARD_URL = os.getenv("BOARD_URL", "http://localhost:8003/api/v1")
BOARD_TOKEN = os.getenv("BOARD_TOKEN", "")  # Keycloak JWT if needed

mcp = FastMCP("dispatch")


# ── HTTP helpers ─────────────────────────────────────────────


async def _get(path: str, params: dict | None = None) -> Any:
    headers = {"Authorization": f"Bearer {BOARD_TOKEN}"} if BOARD_TOKEN else {}
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{BOARD_URL}{path}", params=params, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()


async def _post(path: str, body: dict | None = None) -> Any:
    headers = {"Content-Type": "application/json"}
    if BOARD_TOKEN:
        headers["Authorization"] = f"Bearer {BOARD_TOKEN}"
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{BOARD_URL}{path}", json=body or {}, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()


async def _put(path: str, body: dict | None = None) -> Any:
    headers = {"Content-Type": "application/json"}
    if BOARD_TOKEN:
        headers["Authorization"] = f"Bearer {BOARD_TOKEN}"
    async with httpx.AsyncClient() as c:
        r = await c.put(f"{BOARD_URL}{path}", json=body or {}, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()


async def _delete(path: str) -> str:
    headers = {}
    if BOARD_TOKEN:
        headers["Authorization"] = f"Bearer {BOARD_TOKEN}"
    async with httpx.AsyncClient() as c:
        r = await c.delete(f"{BOARD_URL}{path}", headers=headers, timeout=30)
        r.raise_for_status()
        return "ok"


def _fmt(data: Any) -> str:
    """Format API response as readable JSON."""
    return json.dumps(data, indent=2, default=str)


# ── Read Tools ───────────────────────────────────────────────


@mcp.tool()
async def list_projects() -> str:
    """List all projects (Launch Pads) on the board."""
    data = await _get("/projects")
    projects = data if isinstance(data, list) else data.get("items", data)
    lines = []
    for p in projects:
        pct = p.get("progress_pct", 0)
        lines.append(f"- {p['name']} [{p['status']}] {pct}% — id: {p['id']}")
    return "\n".join(lines) if lines else "No projects found."


@mcp.tool()
async def get_board(project_id: str) -> str:
    """Get the full kanban board for a project — all columns with their packets.

    Args:
        project_id: UUID of the project
    """
    data = await _get(f"/projects/{project_id}/board")
    lines = [f"# {data['project']['name']}"]
    for col in data.get("columns", []):
        state = col["state"]
        tasks = col["tasks"]
        lines.append(f"\n## {state['name']} ({len(tasks)})")
        for t in tasks:
            flags = []
            if t.get("is_blocked"):
                flags.append("BLOCKED")
            if t.get("scope_flag") == "scope_creep":
                flags.append("SCOPE+")
            if t.get("scope_flag") == "possible_scope_creep":
                flags.append("?SCOPE")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            assignee = t.get("assignee_name", "")
            assignee_str = f" @{assignee}" if assignee else ""
            lines.append(f"  - {t['title']}{flag_str}{assignee_str} (id: {t['id']})")
    return "\n".join(lines)


@mcp.tool()
async def list_ops(project_id: str) -> str:
    """List all Ops (sub-projects/deliverables) for a project.

    Args:
        project_id: UUID of the project
    """
    data = await _get(f"/projects/{project_id}/sub-projects")
    ops = data if isinstance(data, list) else data.get("items", data)
    lines = []
    for op in ops:
        approved = " [APPROVED]" if op.get("approved_at") else ""
        scope = ""
        sc = op.get("scope_creep_count", 0)
        psc = op.get("possible_scope_creep_count", 0)
        if sc or psc:
            scope = f" (scope: {sc} creep, {psc} unresolved)"
        lines.append(
            f"- {op['name']} [{op['status']}] {op.get('progress_pct', 0)}% "
            f"({op.get('done_tasks', 0)}/{op.get('total_tasks', 0)}){approved}{scope} — id: {op['id']}"
        )
    return "\n".join(lines) if lines else "No Ops found."


@mcp.tool()
async def list_tasks(
    project_id: str,
    status: str = "",
    assignee: str = "",
    op_id: str = "",
    limit: int = 50,
) -> str:
    """List packets (tasks) in a project with optional filters.

    Args:
        project_id: UUID of the project
        status: Filter by workflow status (e.g. sprint, in_progress, triage)
        assignee: Filter by assignee user ID
        op_id: Filter by Op (sub-project) ID
        limit: Max results (default 50)
    """
    params: dict[str, Any] = {"limit": limit}
    if status:
        params["status"] = status
    if assignee:
        params["assignee"] = assignee
    if op_id:
        params["sub_project_id"] = op_id
    data = await _get(f"/projects/{project_id}/tasks", params)
    tasks = data.get("items", data) if isinstance(data, dict) else data
    lines = []
    for t in tasks:
        flags = []
        if t.get("is_blocked"):
            flags.append("BLOCKED")
        if t.get("scope_flag"):
            flags.append(t["scope_flag"])
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        lines.append(
            f"- [{t['workflow_status']}] {t['title']}{flag_str} "
            f"(priority: {t['priority']}, id: {t['id']})"
        )
    return "\n".join(lines) if lines else "No packets found."


@mcp.tool()
async def get_task(task_id: str) -> str:
    """Get full details of a packet (task) including activity log.

    Args:
        task_id: UUID of the task
    """
    data = await _get(f"/tasks/{task_id}")
    lines = [
        f"# {data['title']}",
        f"Status: {data['workflow_status']} | Priority: {data['priority']}",
    ]
    if data.get("assignee_name"):
        lines.append(f"Assignee: {data['assignee_name']}")
    if data.get("sub_project_id"):
        lines.append(f"Op: {data.get('sub_project_id')}")
    if data.get("is_blocked"):
        lines.append(f"BLOCKED: {data.get('blocked_reason', '')}")
    if data.get("scope_flag"):
        lines.append(f"Scope: {data['scope_flag']}")
    if data.get("description"):
        lines.append(f"\n{data['description']}")
    if data.get("acceptance_criteria"):
        lines.append("\nAcceptance Criteria:")
        for c in data["acceptance_criteria"]:
            if isinstance(c, dict):
                check = "x" if c.get("checked") else " "
                lines.append(f"  [{check}] {c.get('text', '')}")
            else:
                lines.append(f"  [ ] {c}")
    if data.get("tags"):
        tags = ", ".join(f"{k}:{v}" for k, v in data["tags"].items())
        lines.append(f"\nTags: {tags}")
    if data.get("events"):
        lines.append("\nActivity:")
        for evt in data["events"][-10:]:
            meta = evt.get("metadata", {})
            if evt["event_type"] == "comment":
                lines.append(f"  [{evt.get('actor_name', '')}] {meta.get('body', '')}")
            elif evt["event_type"] == "status_change":
                lines.append(f"  {evt.get('actor_name', '')} moved {meta.get('from', '')} → {meta.get('to', '')}")
            else:
                lines.append(f"  {evt.get('actor_name', '')} {evt['event_type']}")
    return "\n".join(lines)


@mcp.tool()
async def get_summary(project_id: str) -> str:
    """Get a high-level project summary — Ops, progress, health.

    Args:
        project_id: UUID of the project
    """
    project = await _get(f"/projects/{project_id}")
    ops_data = await _get(f"/projects/{project_id}/sub-projects")
    ops = ops_data if isinstance(ops_data, list) else ops_data.get("items", ops_data)
    board = await _get(f"/projects/{project_id}/board")

    # Count tasks by status
    status_counts: dict[str, int] = {}
    total = 0
    for col in board.get("columns", []):
        name = col["state"]["name"]
        count = len(col["tasks"])
        if count:
            status_counts[name] = count
            total += count

    lines = [
        f"# {project['name']}",
        f"Status: {project['status']} | Progress: {project.get('progress_pct', 0)}%",
        f"Total packets: {total}",
        "",
        "## Pipeline:",
    ]
    for name, count in status_counts.items():
        lines.append(f"  {name}: {count}")

    lines.append("\n## Ops:")
    for op in ops:
        approved = " [APPROVED]" if op.get("approved_at") else ""
        lines.append(
            f"  - {op['name']} [{op['status']}] "
            f"{op.get('done_tasks', 0)}/{op.get('total_tasks', 0)} done{approved}"
        )

    return "\n".join(lines)


@mcp.tool()
async def get_progress(project_id: str) -> str:
    """Get client-safe project progress — per-Op summary with status breakdown.

    Args:
        project_id: UUID of the project
    """
    data = await _get(f"/projects/{project_id}/progress")
    lines = [
        f"# {data['project_name']}",
        f"Progress: {data['progress_pct']}% | Ops: {data['completed_ops']}/{data['total_ops']} complete",
    ]
    for op in data["ops"]:
        health = ""
        if op.get("health"):
            health = f" — {op['health']['label']}"
        blocked = f" ({op['blocked_count']} blocked)" if op.get("blocked_count") else ""
        lines.append(
            f"\n## {op['op_name']} [{op['status']}] {op['progress_pct']}%"
            f"{health}{blocked}"
        )
        lines.append(f"  {op['done_tasks']}/{op['total_tasks']} packets done")
        if op.get("by_status"):
            for status, count in op["by_status"].items():
                lines.append(f"  {status}: {count}")
    return "\n".join(lines)


# ── Write Tools ──────────────────────────────────────────────


@mcp.tool()
async def create_task(
    project_id: str,
    title: str,
    description: str = "",
    priority: str = "medium",
    op_id: str = "",
    as_draft: bool = False,
) -> str:
    """Create a new packet (task) in a project.

    Args:
        project_id: UUID of the project
        title: Packet title
        description: Packet description
        priority: low, medium, high, critical
        op_id: Optional Op (sub-project) ID to assign to
        as_draft: If true, creates in draft state for review
    """
    body: dict[str, Any] = {
        "title": title,
        "priority": priority,
    }
    if description:
        body["description"] = description
    if as_draft:
        body["workflow_status"] = "draft"

    if op_id:
        data = await _post(f"/sub-projects/{op_id}/tasks", body)
    else:
        data = await _post(f"/projects/{project_id}/tasks", body)

    return f"Created: {data['title']} (id: {data['id']}, status: {data['workflow_status']})"


@mcp.tool()
async def move_task(task_id: str, to_status: str) -> str:
    """Move a packet to a new workflow status.

    Args:
        task_id: UUID of the task
        to_status: Target status (triage, sprint, in_progress, review, done, etc.)
    """
    data = await _put(f"/tasks/{task_id}/move", {"to_status": to_status})
    return f"Moved '{data['title']}' to {data['workflow_status']}"


@mcp.tool()
async def add_comment(task_id: str, body: str) -> str:
    """Add a comment to a packet's activity log.

    Args:
        task_id: UUID of the task
        body: Comment text
    """
    await _post(f"/tasks/{task_id}/comment", {"body": body})
    return f"Comment added to {task_id}"


@mcp.tool()
async def assign_to_op(task_id: str, op_id: str) -> str:
    """Assign a packet to an Op (sub-project).

    Args:
        task_id: UUID of the task
        op_id: UUID of the Op to assign to
    """
    data = await _put(f"/tasks/{task_id}/assign-op", {"sub_project_id": op_id})
    return f"Assigned '{data['title']}' to Op {op_id}"


@mcp.tool()
async def create_op(project_id: str, name: str, description: str = "", objective: str = "") -> str:
    """Create a new Op (deliverable) in a project.

    Args:
        project_id: UUID of the project
        name: Op name
        description: Optional description
        objective: Optional objective statement
    """
    body: dict[str, Any] = {"name": name}
    if description:
        body["description"] = description
    if objective:
        body["objective"] = objective
    data = await _post(f"/projects/{project_id}/sub-projects", body)
    return f"Created Op: {data['name']} (id: {data['id']})"


@mcp.tool()
async def approve_op(op_id: str) -> str:
    """Approve an Op — baselines scope, any new packets become possible scope creep.

    Args:
        op_id: UUID of the Op to approve
    """
    data = await _post(f"/sub-projects/{op_id}/approve")
    return (
        f"Approved '{data['name']}' — baseline: {data.get('baseline_task_count', 0)} packets"
    )


# ── Update Tools ─────────────────────────────────────────────


@mcp.tool()
async def update_task(
    task_id: str,
    title: str = "",
    description: str = "",
    priority: str = "",
) -> str:
    """Update a packet's title, description, or priority.

    Args:
        task_id: UUID of the task
        title: New title (leave empty to keep current)
        description: New description (leave empty to keep current)
        priority: New priority: low, medium, high, critical (leave empty to keep current)
    """
    body: dict[str, Any] = {}
    if title:
        body["title"] = title
    if description:
        body["description"] = description
    if priority:
        body["priority"] = priority
    if not body:
        return "Nothing to update — provide at least one field."
    data = await _put(f"/tasks/{task_id}", body)
    return f"Updated: {data['title']}"


@mcp.tool()
async def block_task(task_id: str, reason: str) -> str:
    """Block a packet with a reason.

    Args:
        task_id: UUID of the task
        reason: Why the task is blocked
    """
    data = await _put(f"/tasks/{task_id}/block", {"reason": reason})
    return f"Blocked '{data['title']}': {reason}"


@mcp.tool()
async def unblock_task(task_id: str) -> str:
    """Unblock a previously blocked packet.

    Args:
        task_id: UUID of the task
    """
    data = await _put(f"/tasks/{task_id}/unblock")
    return f"Unblocked '{data['title']}'"


@mcp.tool()
async def set_scope_flag(task_id: str, flag: str) -> str:
    """Set the scope flag on a packet (for approved Ops).

    Args:
        task_id: UUID of the task
        flag: scope_creep, not_scope_creep, or possible_scope_creep
    """
    data = await _put(f"/tasks/{task_id}/scope-flag", {"scope_flag": flag})
    return f"Set scope flag on '{data['title']}' to {flag}"


@mcp.tool()
async def merge_task(target_task_id: str, source_task_id: str) -> str:
    """Merge (eat) one packet into another — absorbs description, criteria, tags, activity.

    Args:
        target_task_id: UUID of the packet that absorbs
        source_task_id: UUID of the packet being eaten (will be closed)
    """
    data = await _post(f"/tasks/{target_task_id}/merge", {"source_task_id": source_task_id})
    return f"Merged into '{data['title']}' — source packet closed"


@mcp.tool()
async def set_tag(task_id: str, key: str, value: str) -> str:
    """Set a tag on a packet.

    Args:
        task_id: UUID of the task
        key: Tag key
        value: Tag value
    """
    data = await _put(f"/tasks/{task_id}/tags/{key}", {"value": value})
    return f"Tag set: {key}={value}"


@mcp.tool()
async def set_project_domains(project_id: str, domains: list[str]) -> str:
    """Set allowed email domains for auto-viewer access on a project.

    Args:
        project_id: UUID of the project
        domains: List of email domains (e.g. ["hertex.co.za", "client.com"])
    """
    data = await _put(f"/projects/{project_id}/domains", {"allowed_domains": domains})
    ds = data.get("allowed_domains", [])
    return f"Domains set on '{data['name']}': {', '.join(ds) if ds else '(none)'}"


@mcp.tool()
async def reject_draft(task_id: str, feedback: str) -> str:
    """Reject a draft packet with feedback — keeps it in draft for rework.

    Args:
        task_id: UUID of the task
        feedback: Feedback explaining why the draft was rejected
    """
    data = await _post(f"/tasks/{task_id}/reject-draft", {"feedback": feedback})
    return f"Rejected draft '{data['title']}' with feedback"


@mcp.tool()
async def approve_drafts(project_id: str, task_ids: list[str]) -> str:
    """Bulk approve draft packets — moves them from draft to triage.

    Args:
        project_id: UUID of the project
        task_ids: List of task UUIDs to approve
    """
    data = await _post(f"/projects/{project_id}/bulk-approve", {"task_ids": task_ids})
    return f"Approved {len(task_ids)} drafts"


@mcp.tool()
async def finalise_op(op_id: str) -> str:
    """Finalise (deliver) an Op when all packets are done.

    Args:
        op_id: UUID of the Op to finalise
    """
    data = await _put(f"/sub-projects/{op_id}/finalise")
    return f"Finalised '{data['name']}' — status: {data['status']}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
