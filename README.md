# Agentic Research Intelligence Platform

> A production-oriented multi-agent AI platform for autonomous research, evidence retrieval, source verification, and decision support.

## 🚧 Project Status

**Early development — Milestone 1**

The project is being developed incrementally from a working MCP research tool layer toward a complete agentic research and decision intelligence platform.

## Overview

The Agentic Research Intelligence Platform is designed to help AI agents solve complex research tasks by combining:

* Multi-agent orchestration
* Model Context Protocol (MCP)
* Retrieval-Augmented Generation (RAG)
* Tool calling
* Source verification
* Structured evidence collection
* Human-in-the-loop workflows
* Agent evaluation
* Observability
* FastAPI
* Docker
* Automated testing and CI/CD

The long-term goal is to build an AI system that can:

```text
User Question
      │
      ▼
   Planner
      │
      ▼
Task Decomposition
      │
      ├───────────────┐
      ▼               ▼
 Research Agent   Retrieval Agent
      │               │
      └───────┬───────┘
              ▼
       Evidence Collection
              │
              ▼
       Verification Agent
              │
              ▼
       Synthesis Agent
              │
              ▼
        Evaluation Agent
              │
              ▼
         Final Answer
```

## Current Milestone

### MCP Research Server

The first milestone implements an MCP-based research server exposing tools for:

* Searching research sources
* Retrieving individual sources
* Extracting candidate claims

Current tools:

```text
search_sources()
get_source()
extract_claims()
```

The research data is currently an in-memory dataset. A persistent document and vector retrieval layer will be introduced in later milestones.

## Architecture

The planned architecture is:

```text
                         ┌──────────────┐
                         │     User     │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │   FastAPI    │
                         │   Gateway    │
                         └──────┬───────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Orchestrator  │
                       │    (LangGraph)  │
                       └────────┬────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
        Research Agent    Retrieval Agent    Verification
              │                 │                 │
              └─────────────────┼─────────────────┘
                                │
                                ▼
                         Synthesis Agent
                                │
                                ▼
                         Evaluation Agent
                                │
                                ▼
                          Final Response
```

### MCP Tool Layer

The platform will use MCP to standardize access to external tools and data sources.

Planned MCP servers:

```text
mcp/
├── research_server.py
├── knowledge_server.py
├── database_server.py
└── verification_server.py
```

## Technology Stack

### AI / Agentic AI

* Python
* LangGraph
* LangChain
* Large Language Models
* Model Context Protocol (MCP)
* Retrieval-Augmented Generation

### Backend

* FastAPI
* REST APIs
* PostgreSQL
* Vector search

### Infrastructure

* Docker
* GitHub Actions
* Automated testing
* CI/CD

### Evaluation

* Retrieval evaluation
* Answer faithfulness
* Citation accuracy
* Tool-selection accuracy
* Task completion rate
* Latency and execution metrics

## Repository Structure

```text
agentic-research-intelligence/
│
├── app/
│   ├── api/
│   ├── agents/
│   ├── graph/
│   ├── mcp/
│   ├── retrieval/
│   ├── evaluation/
│   └── main.py
│
├── tests/
│
├── data/
│
├── eval/
│
├── .github/
│   └── workflows/
│
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Development Roadmap

### Phase 1 — MCP Foundation

* [x] MCP research server
* [x] Research search tool
* [x] Source retrieval tool
* [x] Claim extraction tool
* [x] In-memory MCP tests

### Phase 2 — Retrieval

* [x] Document ingestion
* [x] Embeddings
* [x] Vector database
* [x] Semantic search
* [x] Reranking (Reciprocal Rank Fusion hybrid search)
* [x] Evidence management

### Phase 3 — Agent Orchestration

* [x] LangGraph state model
* [x] Planning agent
* [x] Research agent
* [x] Retrieval agent
* [x] Verification agent
* [x] Synthesis agent

### Phase 4 — Agent Intelligence

* [ ] Tool selection
* [ ] Agent handoffs
* [ ] Short-term memory
* [ ] Long-term memory
* [ ] Failure recovery
* [ ] Human-in-the-loop approval

### Phase 5 — Evaluation & Observability

* [ ] Evaluation dataset
* [ ] Retrieval metrics
* [ ] Generation metrics
* [ ] Agent metrics
* [ ] Execution traces
* [ ] Cost and latency tracking

### Phase 6 — Production

* [ ] FastAPI service
* [ ] Docker
* [ ] CI/CD
* [ ] Integration tests
* [ ] API documentation
* [ ] Production deployment

## Example Future Workflow

A user could eventually ask:

> "Research the latest approaches to enterprise RAG and recommend an architecture for a large organization."

The system will:

1. Decompose the research question.
2. Identify required evidence.
3. Search available sources.
4. Retrieve relevant documents.
5. Invoke specialized MCP tools.
6. Compare conflicting evidence.
7. Verify important claims.
8. Generate an evidence-grounded recommendation.
9. Evaluate the generated response.
10. Return the final answer with supporting sources.

## Engineering Goals

This project is intentionally focused on engineering reliable agentic AI systems rather than building a simple chatbot.

Key goals:

* Reliability
* Observability
* Reproducibility
* Evaluation
* Modular tool integration
* Safe agent execution
* Testability
* Production-oriented architecture

## License

MIT License
