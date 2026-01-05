# ScholarGraph Temporal Versioning & Supersession Design

**Status:** Future Implementation
**Created:** January 3, 2026
**Priority:** High - Addresses outdated information retrieval

---

## Problem Statement

When querying ScholarGraph via MCP for research information, the system currently:
1. Returns all matching documents regardless of recency
2. Has no concept of "superseded" or "outdated" information
3. Relies only on `ingestion_date` for temporal sorting (hard cutoff filtering)

**Example Issue:**
- Query: "Phase 2A LoRA training results"
- Returns: Old "Phase 2A success" document AND new "Phase 2A failed" document
- Problem: Claude cannot determine which is current truth

---

## Proposed Solution: Document Versioning & Supersession

### Schema Changes

#### New Document Properties

```cypher
// Document node - new properties
(d:Document {
    version: 1,              // Integer version number
    is_latest: true,         // Boolean - true for most recent version
    superseded_by: "doc_id", // Reference to newer version (or null)
    superseded_at: null      // ISO timestamp when superseded
})
```

#### New Relationship Type

```cypher
// SUPERSEDES relationship
(d1:Document)-[:SUPERSEDES {reason: string, timestamp: datetime}]->(d2:Document)
```

### Example State

```
Document A (v1, is_latest=false, superseded_by=B)
  "Phase 2A LoRA training successful"
  ↑
  │ SUPERSEDES {reason: "Findings updated after testing", timestamp: "2026-01-03"}
  │
Document B (v2, is_latest=true, superseded_by=null)
  "Phase 2A Local LoRA Post-Mortem: Failed due to hardware constraints"
```

---

## Supersession Detection Logic

### Automatic Detection on Ingestion

When ingesting a new document:

```python
def detect_supersession(new_doc, existing_docs):
    """
    Determine if new document supersedes existing ones.

    Triggers:
    1. Similar title (fuzzy match > 0.8)
    2. Same topics + newer ingestion_date
    3. Session docs with date-based naming (YYYY-MM-DD pattern)
    """
    candidates = []

    for existing in existing_docs:
        # Check title similarity
        if title_similarity(new_doc.title, existing.title) > 0.8:
            if new_doc.ingestion_date > existing.ingestion_date:
                candidates.append(existing)

        # Check topic overlap
        if topic_overlap(new_doc, existing) > 0.7:
            if new_doc.ingestion_date > existing.ingestion_date:
                candidates.append(existing)

        # Check session naming pattern
        if is_session_doc(new_doc) and is_session_doc(existing):
            if new_doc.date > existing.date:
                candidates.append(existing)

    return candidates
```

### Manual Supersession (CLI)

```bash
# Manually mark a document as superseded
rkg supersede <old_doc_id> --by <new_doc_id> --reason "Updated with new findings"

# View supersession chain
rkg chain <document_id>
```

---

## Search Behavior Changes

### Default Search (only_latest=true)

```cypher
// Vector search with latest-only filter
CALL db.index.vector.queryNodes('chunk_embeddings', $k, $embedding)
YIELD node AS c, score
MATCH (d:Document)-[:CONTAINS]->(c)
WHERE d.is_latest = true  // NEW: Only return latest versions
RETURN c.chunk_id, c.content, d.title, score
ORDER BY score DESC, d.ingestion_date DESC
```

### All Versions Search (only_latest=false)

```cypher
// Return all versions with metadata
CALL db.index.vector.queryNodes('chunk_embeddings', $k, $embedding)
YIELD node AS c, score
MATCH (d:Document)-[:CONTAINS]->(c)
OPTIONAL MATCH (d)-[s:SUPERSEDES]->(old:Document)
RETURN c.chunk_id,
       c.content,
       d.title,
       d.is_latest,
       d.superseded_by,
       score
ORDER BY score DESC, d.ingestion_date DESC
```

---

## MCP API Changes

### Updated search_papers()

```python
async def search_papers(
    self,
    query: str,
    mode: str = "hybrid",
    k: int = 5,
    filter_corpus: Optional[bool] = None,
    days_ago: Optional[int] = None,
    only_latest: bool = True,  # NEW: Default to latest only
    include_superseded: bool = False  # NEW: Include superseded metadata
) -> Dict[str, Any]:
    """
    Search with temporal awareness.

    Args:
        only_latest: If True (default), exclude superseded documents
        include_superseded: If True, include superseded_by metadata in results
    """
```

### Response Format (with superseded metadata)

```json
{
  "success": true,
  "results": [
    {
      "chunk_id": "abc123",
      "content": "Phase 2A failed due to 4GB VRAM limitation",
      "document_title": "Phase 2A Local LoRA Post-Mortem",
      "score": 0.92,
      "is_latest": true,
      "supersedes": [
        {
          "document_id": "old_doc_123",
          "title": "Phase 2A Initial Results",
          "reason": "Hardware testing revealed limitations"
        }
      ]
    }
  ]
}
```

---

## Implementation Plan

| Component | Changes | Estimated Time |
|-----------|---------|----------------|
| **Schema** | Add `version`, `is_latest`, `superseded_by` to Document; add `SUPERSEDES` relationship | 15 min |
| **graph/nodes.py** | Update `create_document_node()` to accept version parameters | 20 min |
| **graph/relationships.py** | Add `create_supersedes_relationship()` method | 10 min |
| **ingestion/batch_ingester.py** | Add supersession detection logic | 30 min |
| **search/vector_index.py** | Add `WHERE d.is_latest = true` filter | 20 min |
| **search/keyword_search.py** | Add `WHERE d.is_latest = true` filter | 15 min |
| **search/hybrid_search.py** | Add `WHERE d.is_latest = true` filter | 10 min |
| **mcp_server/tools.py** | Add `only_latest` parameter | 15 min |
| **rkg.py CLI** | Add `--all-versions` flag; add `supersede` command | 10 min |
| **Migration script** | Backfill existing docs with `is_latest=true` | 15 min |
| **Testing** | Test with session docs; verify supersession chains | 20 min |

| **Total** | | **~3 hours** |

---

## Migration Strategy

### Phase 1: Schema Update (Non-breaking)

```cypher
// Add new properties with defaults
MATCH (d:Document)
SET d.version = 1,
    d.is_latest = true,
    d.superseded_by = null,
    d.superseded_at = null
```

### Phase 2: Retroactive Supersession Detection

```python
# One-time script to detect existing supersessions
def retroactively_detect_supersessions():
    # Find session docs with date patterns
    session_docs = get_session_documents()

    # Group by similar titles
    groups = group_by_title_similarity(session_docs)

    # Mark older as superseded by newer
    for group in groups:
        sorted_docs = sorted(group, key=lambda d: d.ingestion_date)
        for i in range(len(sorted_docs) - 1):
            mark_superseded(sorted_docs[i], sorted_docs[i+1])
```

### Phase 3: Gradual Rollout

1. **Week 1:** Deploy schema changes (default `is_latest=true`)
2. **Week 2:** Enable automatic supersession detection
3. **Week 3:** Add MCP `only_latest` parameter (default `true`)
4. **Week 4:** Monitor and tune detection thresholds

---

## Alternative: Simpler "Ingestion Date Indexing"

If versioning is too complex, a simpler approach:

```python
# Just add an index on ingestion_date for faster recent-first queries
CREATE INDEX document_ingestion_date_index IF NOT EXISTS
FOR (d:Document) ON (d.ingestion_date)

# And always ORDER BY ingestion_date DESC
# Combined with a "recent_first" flag in search
```

**Trade-off:**
- ✅ Simpler (1 hour implementation)
- ❌ Doesn't handle true supersession (just temporal preference)
- ❌ Still returns contradictory old information

---

## Open Questions

1. **Supersession thresholds:** What title similarity / topic overlap percentage triggers auto-detection?
2. **Manual override:** How to easily undo incorrect supersession?
3. **Cross-doc validation:** Should chunks from superseded docs still be searchable?
4. **Session docs special handling:** Treat `YYYY-MM-DD-*-*.md` files as auto-versioning series?

---

## References

- Current ScholarGraph schema: `graph/schema.py`
- MCP tools: `mcp_server/tools.py`
- Search implementation: `search/vector_index.py`, `search/keyword_search.py`
- Ingestion pipeline: `ingestion/batch_ingester.py`

---

**Next Steps:** Prioritize and schedule implementation. The simpler ingestion-date-indexing approach could be a quick win while full versioning is developed.
