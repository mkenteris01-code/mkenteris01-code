---
title: ScholarGraph
---

# ScholarGraph - Research Knowledge Graph with GraphRAG

A GraphRAG system that gives Claude Code persistent memory of research literature using Neo4j and the Model Context Protocol (MCP).

## Features

- **51-paper scoping review corpus** indexed with semantic search
- **Neo4j knowledge graph** with vector embeddings
- **MCP server** for native Claude Code integration
- **Local-first, privacy-preserving** architecture

## Quick Start

### Installation

```bash
cd ScholarGraph
pip install -r requirements.txt
```

### Configure Claude Code

Add to `~/.claude/config.json`:

```json
{
  "mcpServers": {
    "scholargraph": {
      "command": "python",
      "args": ["C:\\projects\\mkenteris01-code\\ScholarGraph\\mcp_server\\server.py"],
      "env": {"PYTHONPATH": "C:\\projects\\mkenteris01-code\\ScholarGraph"}
    }
  }
}
```

### Available Tools

| Tool | Description |
|------|-------------|
| `search_papers` | Semantic/keyword/hybrid search |
| `get_paper_details` | Full paper content retrieval |
| `list_corpus_papers` | List all 51 corpus papers |
| `compare_to_corpus_gaps` | Evaluate against research gaps |
| `get_database_stats` | Database statistics |

## Repository

Full source code: [github.com/mkenteris01-code/ScholarGraph](https://github.com/mkenteris01-code/ScholarGraph)

## Research Context

This system supports my scoping review on the convergence of **Federated Learning**, **Knowledge Graphs**, and **Large Language Models** in language education.

### Key Gaps Identified

| Gap | Coverage | Severity |
|-----|----------|----------|
| FL+KG+LLM Convergence | 0% (0/51) | CRITICAL |
| Dimension 2 Grounding | 0% (0/51) | CRITICAL |
| Validation Metrics | 9.8% (5/51) | HIGH |
| CEFR Alignment | 11.8% (6/51) | HIGH |

## Links

- [Blog: Giving AI Memory](/blog/2025/12/25/giving-ai-memory/)
- [LinkedIn Article](https://www.linkedin.com/pulse/researchers-christmas-miracle-gift-graphrag-memory-michael-kenteris-ped7f/)
