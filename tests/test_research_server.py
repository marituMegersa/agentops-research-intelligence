import json
import pytest

from app.mcp.research_server import mcp


def extract_tool_result(result) -> dict:
    """Extract the JSON payload returned by an MCP tool."""
    assert not result.is_error
    assert result.content

    text = result.content[0].text
    return json.loads(text)


@pytest.mark.asyncio
async def test_search_sources():
    result = await mcp.call_tool(
        "search_sources",
        {"query": "agentic AI"},
    )

    data = extract_tool_result(result)

    assert data["query"] == "agentic AI"
    assert data["results"]
    assert data["results"][0]["id"] == "doc-002"


@pytest.mark.asyncio
async def test_get_source():
    result = await mcp.call_tool(
        "get_source",
        {"source_id": "doc-002"},
    )

    data = extract_tool_result(result)

    assert data["id"] == "doc-002"
    assert data["title"] == "Agentic AI Systems"
    assert "agentic ai" in data["content"].lower()


@pytest.mark.asyncio
async def test_missing_source():
    result = await mcp.call_tool(
        "get_source",
        {"source_id": "does-not-exist"},
    )

    data = extract_tool_result(result)

    assert "error" in data
    assert "not found" in data["error"].lower()


@pytest.mark.asyncio
async def test_extract_claims():
    result = await mcp.call_tool(
        "extract_claims",
        {"text": "RAG enhances LLM outputs. Agentic systems plan and act autonomously!"},
    )

    data = extract_tool_result(result)

    assert data["count"] == 2
    assert len(data["claims"]) == 2
    assert "RAG enhances LLM outputs" in data["claims"][0]