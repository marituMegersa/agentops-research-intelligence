from mcp.server.mcpserver import MCPServer


mcp = MCPServer(
    "research-server",
)


DOCUMENTS = [
    {
        "id": "doc-001",
        "title": "Introduction to Retrieval-Augmented Generation",
        "content": (
            "RAG combines retrieval with language generation "
            "to provide models with external context."
        ),
    },
    {
        "id": "doc-002",
        "title": "Agentic AI Systems",
        "content": (
            "Agentic AI systems use models to plan tasks, "
            "select tools, execute actions, and adapt to results."
        ),
    },
    {
        "id": "doc-003",
        "title": "AI Evaluation",
        "content": (
            "Evaluation measures the quality, reliability, "
            "and performance of AI systems."
        ),
    },
]


@mcp.tool()
def search_sources(query: str, limit: int = 5) -> dict:
    """Search the research knowledge source."""

    query_terms = query.lower().split()
    results = []

    for document in DOCUMENTS:
        text = (
            f"{document['title']} {document['content']}"
        ).lower()

        score = sum(term in text for term in query_terms)

        if score > 0:
            results.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "score": score,
                }
            )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return {
        "query": query,
        "results": results[:limit],
    }


@mcp.tool()
def get_source(source_id: str) -> dict:
    """Retrieve a research source by its identifier."""

    for document in DOCUMENTS:
        if document["id"] == source_id:
            return document

    return {
        "error": f"Source '{source_id}' was not found."
    }


@mcp.tool()
def extract_claims(text: str) -> dict:
    """Extract candidate claims from research text."""

    sentences = [
        sentence.strip()
        for sentence in text.replace("!", ".").split(".")
        if sentence.strip()
    ]

    return {
        "claims": sentences,
        "count": len(sentences),
    }


if __name__ == "__main__":
    mcp.run()