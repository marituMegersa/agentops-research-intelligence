"""Tests for FastMCP research server tools."""

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


@pytest.mark.asyncio
async def test_ingest_and_record_evidence():
    # 1. Ingest a new document
    ingest_res = await mcp.call_tool(
        "ingest_source",
        {
            "title": "Autonomous Decision Making",
            "content": "Autonomous decision making requires rigorous evaluation benchmarks. Tracing tool steps ensures observability.",
            "metadata": {"author": "Habtamu"},
        },
    )
    ingest_data = extract_tool_result(ingest_res)
    assert ingest_data["status"] == "indexed"
    assert ingest_data["chunks_indexed"] >= 1
    new_doc_id = ingest_data["id"]

    # 2. Search for the ingested document
    search_res = await mcp.call_tool(
        "search_sources",
        {"query": "tracing tool steps observability"},
    )
    search_data = extract_tool_result(search_res)
    assert search_data["results"]
    top_match = search_data["results"][0]
    assert top_match["id"] == new_doc_id

    # 3. Extract claims
    extract_res = await mcp.call_tool(
        "extract_claims",
        {"text": "Tracing tool steps ensures observability."},
    )
    extract_data = extract_tool_result(extract_res)
    claim_id = extract_data["claim_objects"][0]["id"]

    # 4. Record evidence
    ev_res = await mcp.call_tool(
        "record_evidence",
        {
            "claim_id": claim_id,
            "source_id": new_doc_id,
            "chunk_id": top_match["chunk_id"],
            "quote": "Tracing tool steps ensures observability.",
            "status": "verified",
        },
    )
    ev_data = extract_tool_result(ev_res)
    assert ev_data["claim_id"] == claim_id
    assert ev_data["status"] == "verified"