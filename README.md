# Mr. Fusion MCP

MCP servers for the OpsCraft platform. One repo, multiple servers — each module gets its own MCP entry point so users only load the tools they need.

| Server | File | Purpose |
|--------|------|---------|
| **Dispatch** | `dispatch_server.py` | Board/project management — projects, Ops, tasks, AI Runner |
| **Emitter** | `emitter_server.py` | Blog/CMS — posts, categories, subscribers, publishing |
| **Forge** | `forge_server.py` | Workflow engine — templates, instances, Signal webforms |

Shared Keycloak auth lives in `auth.py` — all servers import from it.

## Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package runner)
- Running OpsCraft services (local or remote)

### Clone

```bash
git clone git@github.com:OpsCraftHub/mr-fusion-mcp.git
```

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
        "--python", "3.12",
        "/path/to/mr-fusion-mcp/dispatch_server.py"
      ],
      "env": {
        "BOARD_URL": "https://mr-fusion.opscraft.cc/api/v1"
      }
    },
    "emitter": {
      "command": "uv",
      "args": [
        "run",
        "--with", "mcp[cli]>=1.26.0",
        "--with", "httpx>=0.27.0",
        "--python", "3.12",
        "/path/to/mr-fusion-mcp/emitter_server.py"
      ],
      "env": {
        "CMS_URL": "https://mr-fusion.opscraft.cc/api/cms"
      }
    },
    "forge": {
      "command": "uv",
      "args": [
        "run",
        "--with", "mcp[cli]>=1.26.0",
        "--with", "httpx>=0.27.0",
        "--python", "3.12",
        "/path/to/mr-fusion-mcp/forge_server.py"
      ],
      "env": {
        "BOARD_URL": "https://mr-fusion.opscraft.cc/api/board",
        "OUTBOX_URL": "https://mr-fusion.opscraft.cc/api/outbox"
      }
    }
  }
}
```

### Claude Desktop

Same config in `~/Library/Application Support/Claude/claude_desktop_config.json`.

## Environment Variables

### Dispatch (`dispatch_server.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `BOARD_URL` | `https://mr-fusion.opscraft.cc/api/v1` | Board API base URL |
| `BOARD_TOKEN` | — | Static JWT fallback (not needed with Keycloak) |

### Emitter (`emitter_server.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `CMS_URL` | `http://localhost:8009/api/v1` | CMS API base URL |

### Forge (`forge_server.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `BOARD_URL` | `https://mr-fusion.opscraft.cc/api/board` | Board API base URL |
| `OUTBOX_URL` | `https://mr-fusion.opscraft.cc/api/outbox` | Signal (Outbox) API base URL |
| `BOARD_TOKEN` | — | Static JWT fallback (not needed with Keycloak) |

### Keycloak (all servers)

| Variable | Default | Description |
|----------|---------|-------------|
| `KEYCLOAK_URL` | — | Keycloak base URL (enables auto-auth) |
| `KEYCLOAK_REALM` | `opscraft` | Realm name |
| `KEYCLOAK_CLIENT_ID` | `mr-fusion-frontend` | OIDC client ID |
| `KEYCLOAK_USERNAME` | — | Service account username |
| `KEYCLOAK_PASSWORD` | — | Service account password |

In **dev mode** no auth is needed. In **production**, set the Keycloak variables — the servers handle token refresh automatically.

## Dispatch Tools

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
| `list_members` | List project team members with IDs and roles |

### Write

| Tool | Description |
|------|-------------|
| `create_project` | Create a new project |
| `create_task` | Create a packet (optionally as draft) |
| `move_task` | Move packet to new workflow status |
| `groom_and_ready` | Groom and advance to ready |
| `submit_for_review` | Move to review with summary |
| `complete_task` | Complete with closing comment |
| `add_comment` | Comment on a packet |
| `assign_task` | Assign a team member to a packet |
| `assign_to_op` | Assign packet to an Op |
| `create_op` | Create a new Op |
| `approve_op` | Approve Op — baselines scope |
| `finalise_op` | Deliver an Op when all packets done |

### Update

| Tool | Description |
|------|-------------|
| `update_task` | Update title, description, priority, estimate |
| `update_op` | Update Op details and dates |
| `update_project` | Update project name, description, repos |
| `block_task` / `unblock_task` | Block/unblock a packet |
| `set_scope_flag` | Flag scope creep on a packet |
| `merge_task` | Absorb one packet into another |
| `set_tag` | Set key:value tag on a packet |
| `reject_draft` / `approve_drafts` | Draft review workflow |
| `delegate_criterion` | Delegate acceptance criterion to a user |

### AI Runner

| Tool | Description |
|------|-------------|
| `get_ai_settings` / `update_ai_settings` | Manage AI Runner config |
| `get_ai_usage` | View token spend and budget |
| `accept_suggestion` / `dismiss_suggestion` | Handle AI assignee suggestions |
| `trigger_reclassify` / `trigger_regroom` | Re-run AI pipelines |
| `trigger_reenrich` / `trigger_rebootstrap` | Re-scan repos and context |

### Workflow Intelligence

| Tool | Description |
|------|-------------|
| `pick_next_task` | Recommend best task to work on next |
| `standup_summary` | Generate daily standup report |
| `sprint_review` | Full sprint review with metrics |
| `scan_local_workspace` | Scan local repos for tech stack |
| `sync_workspace_to_lp` | Upload repo info to Dispatch |

### Lattice (Documents)

| Tool | Description |
|------|-------------|
| `create_document` / `read_document` / `update_document` | Doc CRUD |
| `link_document` / `list_linked_documents` | Link docs to Board entities |
| `search_documents` | Search documents by title |

## Emitter Tools

| Tool | Description |
|------|-------------|
| `create_blog_post` | Create a draft blog post |
| `list_blog_posts` | List posts with optional status filter |
| `get_blog_post` | Read full post content |
| `update_blog_post` | Update post fields |
| `delete_blog_post` | Delete a draft post |
| `publish_blog_post` | Publish a draft |
| `unpublish_blog_post` | Revert to draft |
| `publish_blog_newsletter` | Publish + push to Signal (email) |
| `list_blog_categories` | List all categories |
| `create_blog_category` | Create a category |
| `upload_post_image` | Upload image from URL to a post |
| `list_blog_subscribers` | List newsletter subscribers |

## Forge Tools

### Templates

| Tool | Description |
|------|-------------|
| `list_workflow_templates` | List templates on a project |
| `create_workflow_template` | Create a reusable workflow template with steps |
| `update_workflow_template` | Update template name, steps, fields |
| `delete_workflow_template` | Remove a template |

### Instances

| Tool | Description |
|------|-------------|
| `create_workflow_instance` | Spin up a live workflow from a template (Op + tasks + webforms) |

### Signal Webforms

| Tool | Description |
|------|-------------|
| `create_webform` | Create a data collection form |
| `list_webforms` | List forms with submission counts |

## Example Usage

Once connected, just talk to Claude:

> "Show me the board"

> "Create a high priority ticket: Set up monitoring with Prometheus"

> "What's blocked right now?"

> "Give me a progress report"

> "Write a blog post about our new feature"

> "Publish the draft and send it as a newsletter"

> "Create a sales process template with 5 steps"

> "Spin up a new client onboarding workflow for Acme Corp"

> "List my workflow templates"
