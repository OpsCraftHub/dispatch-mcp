"""Forge workflow tools for Claude — template CRUD, instance creation, Signal webforms."""

import json
import os
import uuid
from typing import Any

import httpx

BOARD_URL = os.getenv("BOARD_URL", "https://mr-fusion.opscraft.cc/api/board")
OUTBOX_URL = os.getenv("OUTBOX_URL", "https://mr-fusion.opscraft.cc/api/outbox")


# ── Board HTTP helpers ───────────────────────────────────────


async def _board_get(path: str, auth_headers: dict, params: dict | None = None) -> Any:
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{BOARD_URL}{path}", params=params, headers=auth_headers, timeout=30)
        if not r.is_success:
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:200]
            raise Exception(f"Board {r.status_code}: {detail}")
        return r.json()


async def _board_post(path: str, auth_headers: dict, body: dict | None = None) -> Any:
    headers = {"Content-Type": "application/json", **auth_headers}
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{BOARD_URL}{path}", json=body or {}, headers=headers, timeout=30)
        if not r.is_success:
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:200]
            raise Exception(f"Board {r.status_code}: {detail}")
        return r.json()


async def _board_put(path: str, auth_headers: dict, body: dict | None = None) -> Any:
    headers = {"Content-Type": "application/json", **auth_headers}
    async with httpx.AsyncClient() as c:
        r = await c.put(f"{BOARD_URL}{path}", json=body or {}, headers=headers, timeout=30)
        if not r.is_success:
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:200]
            raise Exception(f"Board {r.status_code}: {detail}")
        return r.json()


# ── Signal (Outbox) HTTP helpers ─────────────────────────────


async def _outbox_get(path: str, auth_headers: dict, params: dict | None = None) -> Any:
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{OUTBOX_URL}{path}", params=params, headers=auth_headers, timeout=30)
        if not r.is_success:
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:200]
            raise Exception(f"Signal {r.status_code}: {detail}")
        return r.json()


async def _outbox_post(path: str, auth_headers: dict, body: dict | None = None) -> Any:
    headers = {"Content-Type": "application/json", **auth_headers}
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{OUTBOX_URL}{path}", json=body or {}, headers=headers, timeout=30)
        if not r.is_success:
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:200]
            raise Exception(f"Signal {r.status_code}: {detail}")
        return r.json()


def _fmt(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)


# ── Estimate → days mapping ─────────────────────────────────

_ESTIMATE_DAYS = {"xs": 1, "s": 2, "m": 5, "l": 10, "xl": 20}


def register_forge_tools(mcp, auth_headers_fn):
    """Register Forge workflow tools on the MCP server."""

    # ── Template CRUD ────────────────────────────────────────

    @mcp.tool()
    async def list_workflow_templates(project_id: str) -> str:
        """List workflow templates defined on a project.

        Templates are stored in project.ai_runner_settings.workflow_templates[].

        Args:
            project_id: UUID of the project
        """
        headers = await auth_headers_fn()
        project = await _board_get(f"/projects/{project_id}", headers)
        settings = project.get("ai_runner_settings") or {}
        templates = settings.get("workflow_templates") or []

        if not templates:
            return "No workflow templates defined for this project."

        lines = [f"Workflow templates ({len(templates)}):"]
        for t in templates:
            steps = len(t.get("tasks", []))
            cat = t.get("category", "")
            src = t.get("source", "custom")
            sched = ""
            if t.get("schedule"):
                sched = f" [{t['schedule'].get('frequency', '')}]"
            lines.append(
                f"  - {t['name']} ({steps} steps, {cat}, {src}){sched}"
                f" — id: {t['id']}"
            )
            if t.get("description"):
                lines.append(f"    {t['description'][:100]}")
        return "\n".join(lines)

    @mcp.tool()
    async def create_workflow_template(
        project_id: str,
        name: str,
        description: str,
        category: str,
        tasks: str,
        objective: str = "",
        fields: str = "",
        default_estimate: str = "m",
        default_duration_hours: int = 0,
        default_assignee: str = "",
    ) -> str:
        """Create a workflow template on a project.

        Templates define reusable processes (e.g. sales pipeline, onboarding).

        Args:
            project_id: UUID of the project
            name: Template name (e.g. "Sales Pipeline")
            description: What this workflow does
            category: financial, maintenance, client, or operations
            tasks: JSON array of task steps. Each: {title, source_type, description, order, outcome_prompt, module_link}. source_type: manual|webform|approval|automated|notification|checkpoint
            objective: Optional objective statement
            fields: Optional JSON array of intake fields: [{key, label, type, required}]. type: text|phone|email|url|number
            default_estimate: T-shirt size for workflow duration (xs/s/m/l/xl)
            default_duration_hours: Duration in hours (overrides estimate for same-day scheduling)
            default_assignee: Optional user ID to auto-assign all steps
        """
        headers = await auth_headers_fn()

        # Parse tasks JSON
        try:
            task_list = json.loads(tasks)
        except json.JSONDecodeError as e:
            return f"Invalid tasks JSON: {e}"

        # Parse fields JSON if provided
        field_list = []
        if fields:
            try:
                field_list = json.loads(fields)
            except json.JSONDecodeError as e:
                return f"Invalid fields JSON: {e}"

        template = {
            "id": str(uuid.uuid4()),
            "name": name,
            "description": description,
            "category": category,
            "source": "custom",
            "objective": objective or description,
            "tasks": task_list,
            "fields": field_list,
            "default_estimate": default_estimate,
        }
        if default_duration_hours:
            template["default_duration_hours"] = default_duration_hours
        if default_assignee:
            template["default_assignee"] = default_assignee

        # Fetch fresh settings → append → save
        project = await _board_get(f"/projects/{project_id}", headers)
        settings = project.get("ai_runner_settings") or {}
        templates = settings.get("workflow_templates") or []
        templates.append(template)
        settings["workflow_templates"] = templates

        await _board_put(f"/projects/{project_id}", headers, {"ai_runner_settings": settings})

        return (
            f"Created template: {name} ({len(task_list)} steps, {category})"
            f"\nTemplate ID: {template['id']}"
        )

    @mcp.tool()
    async def update_workflow_template(
        project_id: str,
        template_id: str,
        name: str = "",
        description: str = "",
        category: str = "",
        objective: str = "",
        tasks: str = "",
        fields: str = "",
        default_estimate: str = "",
        default_assignee: str = "",
    ) -> str:
        """Update an existing workflow template.

        Only provided fields are changed. Omit fields to keep current values.

        Args:
            project_id: UUID of the project
            template_id: UUID of the template to update
            name: New name
            description: New description
            category: New category
            objective: New objective
            tasks: New tasks JSON array (replaces all steps)
            fields: New fields JSON array (replaces all intake fields)
            default_estimate: New t-shirt estimate
            default_assignee: New default assignee user ID
        """
        headers = await auth_headers_fn()
        project = await _board_get(f"/projects/{project_id}", headers)
        settings = project.get("ai_runner_settings") or {}
        templates = settings.get("workflow_templates") or []

        idx = next((i for i, t in enumerate(templates) if t["id"] == template_id), -1)
        if idx == -1:
            return f"Template {template_id} not found."

        t = templates[idx]
        if name:
            t["name"] = name
        if description:
            t["description"] = description
        if category:
            t["category"] = category
        if objective:
            t["objective"] = objective
        if default_estimate:
            t["default_estimate"] = default_estimate
        if default_assignee:
            t["default_assignee"] = default_assignee
        if tasks:
            try:
                t["tasks"] = json.loads(tasks)
            except json.JSONDecodeError as e:
                return f"Invalid tasks JSON: {e}"
        if fields:
            try:
                t["fields"] = json.loads(fields)
            except json.JSONDecodeError as e:
                return f"Invalid fields JSON: {e}"

        templates[idx] = t
        settings["workflow_templates"] = templates
        await _board_put(f"/projects/{project_id}", headers, {"ai_runner_settings": settings})

        return f"Updated template: {t['name']} (id: {template_id})"

    @mcp.tool()
    async def delete_workflow_template(project_id: str, template_id: str) -> str:
        """Delete a workflow template from a project.

        Args:
            project_id: UUID of the project
            template_id: UUID of the template to delete
        """
        headers = await auth_headers_fn()
        project = await _board_get(f"/projects/{project_id}", headers)
        settings = project.get("ai_runner_settings") or {}
        templates = settings.get("workflow_templates") or []

        before = len(templates)
        templates = [t for t in templates if t["id"] != template_id]
        if len(templates) == before:
            return f"Template {template_id} not found."

        settings["workflow_templates"] = templates
        await _board_put(f"/projects/{project_id}", headers, {"ai_runner_settings": settings})

        return f"Deleted template {template_id}. {len(templates)} template(s) remain."

    # ── Instance creation ────────────────────────────────────

    @mcp.tool()
    async def create_workflow_instance(
        project_id: str,
        template_id: str,
        name: str = "",
        instance_data: str = "",
    ) -> str:
        """Create a live workflow instance from a template.

        Creates an Op + tasks from the template steps. For webform steps,
        automatically creates a Signal webform and links it to the task.

        Args:
            project_id: UUID of the project
            template_id: UUID of the template to instantiate
            name: Instance name (defaults to template name)
            instance_data: JSON object of intake field values (e.g. {"client_name": "Acme"})
        """
        headers = await auth_headers_fn()

        # Load template
        project = await _board_get(f"/projects/{project_id}", headers)
        settings = project.get("ai_runner_settings") or {}
        templates = settings.get("workflow_templates") or []
        template = next((t for t in templates if t["id"] == template_id), None)
        if not template:
            return f"Template {template_id} not found."

        data = {}
        if instance_data:
            try:
                data = json.loads(instance_data)
            except json.JSONDecodeError as e:
                return f"Invalid instance_data JSON: {e}"

        op_name = name or template["name"]

        # 1. Create Op
        op = await _board_post(f"/projects/{project_id}/sub-projects", headers, {
            "name": op_name,
            "objective": template.get("objective") or template.get("description", ""),
            "technical_notes": json.dumps({
                "workflow_mode": True,
                "template_id": template_id,
                "instance_data": data,
            }),
        })
        op_id = op["id"]

        # 2. Set planned dates based on estimate
        estimate = template.get("default_estimate", "m")
        days = _ESTIMATE_DAYS.get(estimate, 5)
        from datetime import date, timedelta
        today = date.today()
        await _board_put(f"/sub-projects/{op_id}", headers, {
            "planned_start": today.isoformat(),
            "planned_end": (today + timedelta(days=days)).isoformat(),
        })

        # 3. Create tasks from template steps
        created_tasks = []
        webform_results = []
        for task_def in template.get("tasks", []):
            tags = {
                "forge": "true",
                "source_type": task_def.get("source_type", "manual"),
                "template_order": str(task_def.get("order", 0)),
            }
            if task_def.get("outcome_prompt"):
                tags["outcome_prompt"] = task_def["outcome_prompt"]
            if task_def.get("module_link"):
                tags["module_link"] = task_def["module_link"]

            task_body: dict[str, Any] = {
                "title": task_def["title"],
                "description": task_def.get("description", ""),
                "priority": "medium",
                "tags": tags,
            }
            if template.get("default_assignee"):
                task_body["assignee"] = template["default_assignee"]

            task = await _board_post(f"/sub-projects/{op_id}/tasks", headers, task_body)
            created_tasks.append(task)

            # Create Signal webform for webform steps
            if task_def.get("source_type") == "webform" and task_def.get("webform_schema"):
                try:
                    wf = await _outbox_post("/webforms", headers, {
                        "name": task_def["title"],
                        "schema_": {"fields": task_def["webform_schema"]},
                        "source_service": "forge",
                        "source_entity_type": "task",
                        "source_entity_id": task["id"],
                        "is_reusable": False,
                        "max_submissions": 1,
                        "require_auth": task_def.get("webform_require_auth", False),
                    })
                    # Link webform to task via tags
                    await _board_put(f"/tasks/{task['id']}", headers, {
                        "tags": {**tags, "webform_id": wf["id"], "webform_token": wf["token"]},
                    })
                    webform_results.append(f"  Webform: {task_def['title']} (token: {wf['token']})")
                except Exception as e:
                    webform_results.append(f"  Webform failed for {task_def['title']}: {e}")

        lines = [
            f"Created workflow instance: {op_name}",
            f"Op ID: {op_id}",
            f"Steps: {len(created_tasks)} tasks created",
            f"Duration: {days} days ({estimate})",
        ]
        if webform_results:
            lines.append("Webforms:")
            lines.extend(webform_results)

        return "\n".join(lines)

    # ── Signal webform tools ─────────────────────────────────

    @mcp.tool()
    async def create_webform(
        name: str,
        fields: str,
        source_service: str = "forge",
        source_entity_type: str = "",
        source_entity_id: str = "",
        is_reusable: bool = True,
        max_submissions: int = 0,
        require_auth: bool = False,
    ) -> str:
        """Create a Signal webform for collecting data.

        Args:
            name: Form name
            fields: JSON array of field definitions. Each: {name, type, label, required, options, placeholder}. type: text|textarea|email|number|date|select|checkbox
            source_service: Which service owns this form (forge, board, etc.)
            source_entity_type: Entity type this form belongs to (task, op, project)
            source_entity_id: UUID of the owning entity
            is_reusable: Whether the form can accept multiple submissions
            max_submissions: Max submissions (0 = unlimited)
            require_auth: Whether Keycloak auth is needed to submit
        """
        headers = await auth_headers_fn()

        try:
            field_list = json.loads(fields)
        except json.JSONDecodeError as e:
            return f"Invalid fields JSON: {e}"

        body: dict[str, Any] = {
            "name": name,
            "schema_": {"fields": field_list},
            "source_service": source_service,
            "is_reusable": is_reusable,
            "require_auth": require_auth,
        }
        if source_entity_type:
            body["source_entity_type"] = source_entity_type
        if source_entity_id:
            body["source_entity_id"] = source_entity_id
        if max_submissions:
            body["max_submissions"] = max_submissions

        wf = await _outbox_post("/webforms", headers, body)
        return (
            f"Created webform: {wf['name']}"
            f"\nID: {wf['id']}"
            f"\nToken: {wf['token']}"
            f"\nPublic URL: /forms/{wf['token']}"
        )

    @mcp.tool()
    async def list_webforms() -> str:
        """List all Signal webforms with status and submission counts."""
        headers = await auth_headers_fn()
        forms = await _outbox_get("/webforms", headers)

        if not forms:
            return "No webforms found."

        lines = [f"Webforms ({len(forms)}):"]
        for f in forms:
            status = f.get("status", "active")
            subs = f.get("submission_count", 0)
            source = f.get("source_service", "")
            src_str = f" [{source}]" if source else ""
            lines.append(
                f"  - {f['name']} ({status}, {subs} submissions){src_str}"
                f" — id: {f['id']}, token: {f['token']}"
            )
        return "\n".join(lines)
