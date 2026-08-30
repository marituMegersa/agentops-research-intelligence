import json

import pytest

from mcp.shared.memory import (
    create_connected_server_and_client_session,
)

from app.mcp.research_server import mcp


def extract_tool_result(result) -> dict:
    """Extract the JSON payload returned by an MCP tool."""
    assert not result.isError
    assert result.content

    text = result.content[0].text
    return json.loads(text)


@pytest.mark.anyio
async def test_search_sources():
    async with create_connected_server_and_client_session(
        mcp._mcp_server
    ) as session:
        await session.initialize()

        result = await session.call_tool(
            "search_sources",
            {"query": "agentic AI"},
        )

        data = extract_tool_result(result)

        assert data["query"] == "agentic AI"
        assert data["results"]
        assert data["results"][0]["id"] == "doc-002"


@pytest.mark.anyio
async def test_get_source():
    async with create_connected_server_and_client_session(
        mcp._mcp_server
    ) as session:
        await session.initialize()

        result = await session.call_tool(
            "get_source",
            {"source_id": "doc-002"},
        )

        data = extract_tool_result(result)

        assert data["id"] == "doc-002"
        assert data["title"] == "Agentic AI Systems"
        assert "agentic ai" in data["content"].lower()


@pytest.mark.anyio
async def test_missing_source():
    async with create_connected_server_and_client_session(
        mcp._mcp_server
    ) as session:
        await session.initialize()

        result = await session.call_tool(
            "get_source",
            {"source_id": "does-not-exist"},
        )

        data = extract_tool_result(result)

        assert "error" in data
        assert "not found" in data["error"].lower()