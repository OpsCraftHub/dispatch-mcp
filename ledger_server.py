"""Vault — MCP server for the OpsCraft Ledger (bookkeeping) service.

Exposes chart of accounts, contacts, cashbook (CSV import + categorise +
learned rules), invoices, credit notes, bills (+AI draft flow), payments,
journal, reports, periods, and members.
"""

from mcp.server.fastmcp import FastMCP

from auth import make_auth_headers_fn
from ledger_tools import register_ledger_tools

mcp = FastMCP("ledger")

_auth_headers = make_auth_headers_fn(static_token_var="LEDGER_TOKEN")

register_ledger_tools(mcp, _auth_headers)

if __name__ == "__main__":
    mcp.run(transport="stdio")
