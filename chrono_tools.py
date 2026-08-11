"""Chrono (Clock) tools for Claude — budgets, allocations, burndown analysis.

Chrono owns per-project + per-Op + per-member hours budgets, plus the time
entries logged against them. These tools let Claude report on budget usage,
project a burn rate, and surface pending overage approvals.

Read-only by design — mutation of budgets happens through the UI so an
accidental Claude call can't shift a client's contracted hours.
"""

import os
from datetime import date, timedelta
from typing import Any

import httpx

CLOCK_URL = os.getenv("CLOCK_URL", "http://localhost:8004/api/v1")


async def _get(path: str, auth_headers: dict, params: dict | None = None) -> Any:
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{CLOCK_URL}{path}", params=params, headers=auth_headers, timeout=30)
        if not r.is_success:
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:200]
            raise Exception(f"Chrono {r.status_code} GET {path}: {detail}")
        return r.json()


def _current_period_bounds(period: str, today: date | None = None) -> tuple[date, date]:
    """Return (start, end) of the current budget period based on period type.

    monthly   → first-of-month … last-of-month
    quarterly → first-of-quarter … last-of-quarter
    fallback  → last 30 days (used for unknown period strings)
    """
    today = today or date.today()
    if period == "monthly":
        start = today.replace(day=1)
        # Roll to next month, then back one day
        if start.month == 12:
            next_month = start.replace(year=start.year + 1, month=1)
        else:
            next_month = start.replace(month=start.month + 1)
        end = next_month - timedelta(days=1)
        return start, end
    if period == "quarterly":
        q = (today.month - 1) // 3
        start_month = q * 3 + 1
        start = today.replace(month=start_month, day=1)
        end_month = start_month + 2
        if end_month == 12:
            end = today.replace(month=12, day=31)
        else:
            # First of month after quarter end, minus 1 day
            end = today.replace(month=end_month + 1, day=1) - timedelta(days=1)
        return start, end
    # Fallback: rolling 30 days
    return today - timedelta(days=30), today


async def _sum_hours(auth_headers: dict, project_ref: str, start: date, end: date) -> float:
    """Sum duration_minutes over TimeEntry rows for a project in the window."""
    # /entries paginates at 500; iterate until we drain (usually one page).
    total_min = 0
    offset = 0
    while True:
        params = {
            "project": project_ref,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "limit": 500,
            "offset": offset,
        }
        rows = await _get("/entries", auth_headers, params)
        if not rows:
            break
        total_min += sum(r.get("duration_minutes", 0) for r in rows)
        if len(rows) < 500:
            break
        offset += 500
    return round(total_min / 60.0, 2)


def register_chrono_tools(mcp, auth_headers_fn):
    """Register Chrono (Clock) tools on the MCP server."""

    @mcp.tool()
    async def list_project_budgets() -> str:
        """List all project budgets on Chrono (hours-per-period contracts)."""
        headers = await auth_headers_fn()
        rows = await _get("/budgets", headers)
        if not rows:
            return "No project budgets configured. Create one in Chrono → Budgets."
        lines = [f"{len(rows)} project budget(s):"]
        for b in rows:
            name = b.get("project_name") or "?"
            hrs = b.get("hours_budget")
            hrs_s = f"{hrs}h" if hrs is not None else "—"
            sla = f" +{b.get('sla_included_hours')}h SLA" if b.get("sla_included_hours") else ""
            lines.append(
                f"  {b.get('project_ref','?'):40} {name:30} "
                f"{b.get('budget_mode','?'):8} {b.get('period','?'):9} "
                f"{hrs_s}{sla}  alert@{b.get('alert_threshold_pct','?')}%"
            )
        return "\n".join(lines)

    @mcp.tool()
    async def get_project_budget(project_ref: str) -> str:
        """Fetch one project budget with current-period consumption.

        Calculates actual hours logged in the current period (from
        `/entries`) and compares against the contracted budget so you
        can see % used, remaining hours, and a simple pace check.

        Args:
            project_ref: Project reference (usually the Board project ID)
        """
        headers = await auth_headers_fn()
        b = await _get(f"/budgets/{project_ref}", headers)
        start, end = _current_period_bounds(b.get("period", "monthly"))
        actual_h = await _sum_hours(headers, project_ref, start, end)

        budget_h = b.get("hours_budget")
        sla_h = b.get("sla_included_hours") or 0
        total_h = (budget_h or 0) + sla_h

        lines = [
            f"Budget: {b.get('project_name') or project_ref}  ({b.get('budget_mode')}, {b.get('period')})",
            f"  Period:     {start} → {end}",
            f"  Contract:   {budget_h}h" + (f" + {sla_h}h SLA = {total_h}h total" if sla_h else ""),
            f"  Logged:     {actual_h}h",
        ]
        if total_h > 0:
            pct = round(actual_h / total_h * 100)
            remaining = round(total_h - actual_h, 2)
            lines.append(f"  Used:       {pct}%  (remaining: {remaining}h)")
            # Simple pace check — where should we be by now if usage is linear?
            days_total = (end - start).days + 1
            days_done = min(days_total, (date.today() - start).days + 1)
            if days_done > 0:
                expected_pct = round(days_done / days_total * 100)
                lines.append(f"  Pace:       {days_done}/{days_total} days elapsed → expected ~{expected_pct}%")
                if pct >= (b.get("alert_threshold_pct") or 80) and pct > expected_pct + 10:
                    lines.append("  ⚠ Burning faster than expected — over threshold + ahead of pace")
                elif pct >= 100:
                    lines.append("  ⚠ Over budget — consider request-overage")
                elif pct < expected_pct - 20:
                    lines.append("  ✓ Well under pace")
                else:
                    lines.append("  ✓ On pace")
        return "\n".join(lines)

    @mcp.tool()
    async def analyse_budget_burndown(project_ref: str) -> str:
        """Day-by-day burn for the current period — logged hours, cumulative, projection.

        Useful for spotting "we blew the budget in week 1" patterns before
        month end.

        Args:
            project_ref: Project reference
        """
        headers = await auth_headers_fn()
        b = await _get(f"/budgets/{project_ref}", headers)
        start, end = _current_period_bounds(b.get("period", "monthly"))
        params = {"project": project_ref, "from": start.isoformat(),
                  "to": end.isoformat(), "limit": 500}
        rows = await _get("/entries", headers, params)

        # Bucket by day
        by_day: dict[str, float] = {}
        for r in rows:
            d = r.get("entry_date")
            if not d:
                continue
            by_day[d] = by_day.get(d, 0) + r.get("duration_minutes", 0) / 60.0

        budget_h = (b.get("hours_budget") or 0) + (b.get("sla_included_hours") or 0)
        days_total = (end - start).days + 1
        today = date.today()
        days_done = min(days_total, max(1, (today - start).days + 1))

        cum = 0.0
        lines = [f"Burndown — {b.get('project_name') or project_ref}  ({start} → {end}, budget {budget_h}h)"]
        lines.append(f"  {'Date':12} {'Logged':>8} {'Cumulative':>12} {'% of budget':>13}")
        cursor = start
        while cursor <= min(end, today):
            k = cursor.isoformat()
            logged = round(by_day.get(k, 0), 2)
            cum = round(cum + logged, 2)
            pct = round(cum / budget_h * 100) if budget_h else 0
            marker = ""
            if budget_h and pct >= 100 and not lines[-1].endswith("← over budget"):
                marker = "  ← over budget"
            lines.append(f"  {k:12} {logged:>8.2f} {cum:>12.2f} {pct:>12}%{marker}")
            cursor += timedelta(days=1)

        # Simple linear projection
        if days_done > 0 and budget_h:
            daily_rate = cum / days_done
            projected = round(daily_rate * days_total, 2)
            proj_pct = round(projected / budget_h * 100)
            lines.append("")
            lines.append(f"  Avg daily burn:  {round(daily_rate, 2)}h/day")
            lines.append(f"  Projected total: {projected}h ({proj_pct}% of {budget_h}h budget)")
        return "\n".join(lines)

    @mcp.tool()
    async def list_op_budgets(project_ref: str) -> str:
        """Per-Op budget breakdown for a project.

        Args:
            project_ref: Project reference (usually the Board project ID)
        """
        headers = await auth_headers_fn()
        rows = await _get(f"/budgets/{project_ref}/ops", headers)
        if not rows:
            return f"No Op-level budgets set for project {project_ref}."
        lines = [f"{len(rows)} Op budget(s):"]
        for o in rows:
            hrs = o.get("hours_budget")
            hrs_s = f"{hrs}h" if hrs is not None else "—"
            sla = f" +{o.get('sla_included_hours')}h SLA" if o.get("sla_included_hours") else ""
            lines.append(
                f"  op_ref={o.get('op_ref','?')} name={o.get('op_name') or '?'} "
                f"mode={o.get('budget_mode','?')} budget={hrs_s}{sla}"
            )
        return "\n".join(lines)

    @mcp.tool()
    async def list_member_budgets(project_ref: str, op_ref: str = "") -> str:
        """Per-member hours allocations. Scoped project-wide or to a single Op.

        Args:
            project_ref: Project reference
            op_ref: Optional Op reference to scope down to one Op's allocations
        """
        headers = await auth_headers_fn()
        path = f"/budgets/{project_ref}/ops/{op_ref}/members" if op_ref else f"/budgets/{project_ref}/members"
        rows = await _get(path, headers)
        if not rows:
            scope = f" for Op {op_ref}" if op_ref else ""
            return f"No member allocations{scope} on project {project_ref}."
        lines = [f"{len(rows)} member allocation(s):"]
        for m in rows:
            hrs = m.get("hours_budget")
            hrs_s = f"{hrs}h" if hrs is not None else "—"
            lines.append(f"  user={m.get('user_id','?')}  {hrs_s}")
        return "\n".join(lines)

    @mcp.tool()
    async def list_budget_approvals(project_ref: str) -> str:
        """Overage-approval requests for a project (pending + historical).

        Args:
            project_ref: Project reference
        """
        headers = await auth_headers_fn()
        rows = await _get(f"/budgets/{project_ref}/approvals", headers)
        if not rows:
            return f"No overage approvals on project {project_ref}."
        lines = [f"{len(rows)} approval(s):"]
        for a in rows:
            reason = a.get("reason") or ""
            reason_s = f" — {reason[:60]}" if reason else ""
            lines.append(
                f"  [{a.get('status','?'):9}] +{a.get('extra_hours','?')}h "
                f"({a.get('period_start')} → {a.get('period_end')}) "
                f"requested_by={a.get('requested_by','?')}{reason_s}"
            )
        return "\n".join(lines)
