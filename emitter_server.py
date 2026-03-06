"""Emitter — MCP server for the OpsCraft CMS (blog management).

Exposes blog post CRUD, publishing, categories, and subscriber tools.
"""

from mcp.server.fastmcp import FastMCP

from auth import make_auth_headers_fn
from emitter_tools import register_emitter_tools

mcp = FastMCP("emitter")

_auth_headers = make_auth_headers_fn(static_token_var="CMS_TOKEN")

register_emitter_tools(mcp, _auth_headers)

if __name__ == "__main__":
    mcp.run(transport="stdio")
