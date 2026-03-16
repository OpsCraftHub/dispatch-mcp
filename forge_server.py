"""Forge — MCP server for the OpsCraft workflow engine.

Exposes workflow template CRUD, instance creation, and Signal webform tools.
"""

from mcp.server.fastmcp import FastMCP

from auth import make_auth_headers_fn
from forge_tools import register_forge_tools

mcp = FastMCP("forge")

_auth_headers = make_auth_headers_fn(static_token_var="BOARD_TOKEN")

register_forge_tools(mcp, _auth_headers)

if __name__ == "__main__":
    mcp.run(transport="stdio")
