# Dispatch MCP

MCP server for the OpsCraft Board service. Gives Claude Code and Claude Desktop full control of your projects, Ops, and packets via natural language.

## Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package runner)
- A running Board service (local or remote)

### Claude Code

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "dispatch": {
      "command": "uv",
      "args": [
        "run",
        "--with", "mcp[cli]>=1.26.0",
        "--with", "httpx>=0.27.0",
        "python", "/path/to/dispatch-mcp/server.py"
      ],
      "env": {
        "BOARD_URL": "http://localhost:8003/api/v1"
      }
    }
  }
}
```

Or clone and run directly:

```bash
git clone git@github.com:OpsCraftHub/dispatch-mcp.git
```

Then point `server.py` path at your clone.

### Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "dispatch": {
      "command": "uv",
      "args": [
        "run",
        "--with", "mcp[cli]>=1.26.0",
        "--with", "httpx>=0.27.0",
        "python", "/path/to/dispatch-mcp/server.py"
      ],
      "env": {
        "BOARD_URL": "http://localhost:8003/api/v1"
      }
    }
  }
}
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BOARD_URL` | Yes | `http://localhost:8003/api/v1` | Board service API base URL |
| `BOARD_TOKEN` | No | — | Keycloak JWT for authenticated access. Not needed if Board runs in dev mode |

### Authentication

In **dev mode** (`DEV_MODE=true` on the Board service), no token is needed — requests are accepted without auth.

In **production**, set `BOARD_TOKEN` to a valid Keycloak JWT. You can get one via:

```bash
curl -s -X POST "https://auth.opscraft.cc/realms/opscraft/protocol/openid-connect/token" \
  -d "client_id=mr-fusion-frontend" \
  -d "grant_type=password" \
  -d "username=YOUR_USER" \
  -d "password=YOUR_PASS" | jq -r '.access_token'
```

Note: JWTs expire (default 5 minutes). For long-running use, consider a Keycloak service account with client credentials grant.

## Available Tools

### Read

| Tool | Description |
|------|-------------|
| `list_projects` | List all projects |
| `get_board` | Full kanban board — all columns and packets |
| `list_ops` | List Ops (deliverables) in a project |
| `list_tasks` | List packets with filters (status, assignee, Op) |
| `get_task` | Full packet detail with activity log |
| `get_summary` | High-level project summary |
| `get_progress` | Client-safe progress report per Op |

### Write

| Tool | Description |
|------|-------------|
| `create_task` | Create a packet (optionally as draft) |
| `move_task` | Move packet to new workflow status |
| `add_comment` | Comment on a packet |
| `assign_to_op` | Assign packet to an Op |
| `create_op` | Create a new Op |
| `approve_op` | Approve Op — baselines scope |
| `finalise_op` | Deliver an Op when all packets done |

### Update

| Tool | Description |
|------|-------------|
| `update_task` | Update title, description, priority |
| `block_task` | Block a packet with reason |
| `unblock_task` | Unblock a packet |
| `set_scope_flag` | Flag scope creep on a packet |
| `merge_task` | Absorb one packet into another |
| `set_tag` | Set key:value tag on a packet |
| `set_project_domains` | Set auto-viewer email domains |
| `reject_draft` | Reject draft with feedback |
| `approve_drafts` | Bulk approve drafts to triage |

## Example Usage

Once connected, just talk to Claude:

> "Show me the board for FuelCo"

> "Create a high priority ticket in the Infrastructure Op: Set up monitoring with Prometheus and Grafana"

> "Move BOARD-080 to done"

> "What's blocked right now?"

> "Give me a progress report for the client"
