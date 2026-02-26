# RAAG: Relationship-Aware Augmented Generation

**Structure-derived document relationship intelligence for RAG pipelines.**

RAAG is a Python SDK that makes retrieval relationship-aware without LLM extraction. It ingests documents, derives cross-document relationships from structure and explicit patterns, and guides retrieval using a document relationship graph.

RAAG is not a RAG framework. Not a vector database. Not an embedding model. It is the intelligence layer between documents and retrieval.

---

## Installation

```bash
# Core library (no vector DB dependency)
pip install raag
```

### Vector Store Adapters

RAAG ships optional adapters for popular vector databases. Install the one you use:

```bash
pip install raag[qdrant]       # Qdrant
```

More adapters coming: `chromadb`, `pinecone`, `pgvector`, `weaviate`.

### Bring Your Own

RAAG defines a `VectorStore` Protocol with three methods: `upsert`, `search`, `delete`. If your vector DB isn't listed above, wrap it yourself:

```python
from raag.protocols import VectorStore, VectorItem, VectorSearchResult

class MyVectorStore:
    async def upsert(self, items: list[VectorItem]) -> None: ...
    async def search(self, vector: list[float], k: int, tenant_id: str) -> list[VectorSearchResult]: ...
    async def delete(self, ids: list[str], tenant_id: str) -> None: ...
```

No base class needed. If it matches the protocol, it works.

---

## How It Works

### Architecture

```mermaid
graph LR
    subgraph Consumer
        A[Documents] --> B[Parser]
        F[Vector/BM25 Search] --> G[Candidate Results]
        I[LLM] --> J[Answer + Provenance]
    end

    subgraph RAAG
        B --> C[Ingestion Pipeline]
        C --> D[Document Relationship Graph]
        G --> H[Graph-Guided Retrieval]
        D --> H
        H --> I
    end

    subgraph Storage["Consumer's Storage"]
        C --> E1[Relational Store]
        C --> E2[Vector Store]
        D --> E1
    end
```

RAAG sits between parsing and retrieval. The consumer owns the parser, the embedding model, the vector store, and the LLM. RAAG owns the relationship graph and the retrieval enrichment logic.

---

### Ingestion Pipeline

When a document is uploaded via `add_document()`:

```mermaid
flowchart TD
    A[Document Upload] --> B[Step 1: Parse]
    B --> C[Step 2: Normalize]
    C --> D[Step 3: Extract Topics and Dates]
    D --> E[Step 4: Extract Relationships]
    E --> F[Step 5: Propagate Terms]
    F --> G[Step 6: Link Topics]
    G --> H[Step 7: Embed]
    H --> I[Step 8: Resolve References]
    I --> J[Step 9: Persist Graph]
    J --> K[Step 10: Index Metadata]
    K --> L[Step 11: Versioning]

    B -.- B1["Docling or consumer's parser
    Output: structure tree, headings, spans"]

    C -.- C1["Section hierarchy, stable IDs
    Fuzzy correction on save"]

    D -.- D1["spaCy noun phrases + term frequency
    PDF metadata, date patterns"]

    E -.- E1["Layer 1: Regex, patterns
    Layer 2: spaCy NLP
    12 relationship types"]

    F -.- F1["Scan sections for defined terms
    Create uses_term edges"]

    G -.- G1["Jaccard on topic arrays
    Create topically_related edges"]

    H -.- H1["Consumer's embedding model
    Store in consumer's vector DB"]

    I -.- I1["Layer 3: Semantic matching
    Forward + backward resolution
    Unresolved → parked for later"]

    style B1 fill:none,stroke:none
    style C1 fill:none,stroke:none
    style D1 fill:none,stroke:none
    style E1 fill:none,stroke:none
    style F1 fill:none,stroke:none
    style G1 fill:none,stroke:none
    style H1 fill:none,stroke:none
    style I1 fill:none,stroke:none
```

---

### Document Relationship Graph

```mermaid
graph TD
    subgraph Nodes
        DOC[Document]
        VER[Document Version]
        SEC[Section / Clause]
    end

    subgraph Edges
        SEC -->|references| SEC
        SEC -->|overrides| SEC
        VER -->|supersedes| VER
        DOC -->|amends| DOC
        SEC -->|defines| TERM[Term]
        SEC -->|uses_term| SEC
        SEC -->|depends_on| SEC
        SEC -->|precedes| SEC
        SEC -->|effective_on| DATE[DateRange]
        SEC -->|applies_to| DATE
        SEC -->|applies_to_scope| SCOPE[Scope]
        VER -->|topically_related| VER
    end
```

Every edge carries provenance: the matched text that triggered extraction, normalized target, confidence score, and optional condition/scope fields.

---

### Extraction Layers

```mermaid
flowchart TD
    INPUT[Document Text] --> L1

    subgraph L1["Layer 1: Deterministic"]
        R1[Regex and parser rules]
        R2[Metadata and filename detection]
        R3[Phrase pattern matching]
        R4[Date pattern extraction]
        R5[Fuzzy string matching]
        R6[Topic/keyword extraction]
        R7[Term-to-usage scan]
        R8[Topic overlap detection]
        R9[Process/sequence detection]
        R10[Scope and conditional detection]
    end

    L1 -->|Unresolved| L2

    subgraph L2["Layer 2: Structural NLP"]
        S1[POS tagging]
        S2[Dependency parsing]
        S3[Parse tree pattern matching]
        S4[Conditional clause parsing]
        S5[Scope phrase extraction]
    end

    L2 -->|Still unresolved| L3

    subgraph L3["Layer 3: Semantic Similarity"]
        E1["Heading-to-reference matching
        (consumer's embedding model)"]
        E2["Cross-language resolution"]
        E3["Candidate ranking by similarity"]
    end

    L3 -->|Still unresolved| PARK[Parked in unresolved_references]
    PARK -->|"New document uploaded"| L1

    style L1 fill:#e8f5e9
    style L2 fill:#fff3e0
    style L3 fill:#e3f2fd
```

Layer 1 is fast, deterministic, and free. Layer 2 uses spaCy (no LLM). Layer 3 uses the consumer's embedding model. Cost and latency increase with each layer. Most references resolve at Layer 1.

---

### Query-Time Flow

When `search()` is called with candidate results from the consumer's pipeline:

```mermaid
flowchart LR
    A["Consumer's
    Vector/BM25
    Search"] --> B[Candidate Results]

    B --> C["RAAG search()"]

    subgraph RAAG["RAAG Query Pipeline"]
        direction TB
        C --> D[Graph Expansion]
        D --> E[Graph-Aware Scoring]
        E --> F[Conflict Detection]
        F --> G[Subgraph Assembly]
    end

    G --> H[EnrichedResults]

    subgraph Output["EnrichedResults"]
        direction TB
        H1["Per-result: text, source,
        retrieval_reason, expansion_depth,
        related_results, scope, topics"]
        H2["result_graph: edges between
        all returned results"]
        H3["retrieval_summary: primary hits,
        expanded, documents involved,
        conflicts detected"]
    end

    H --> H1
    H --> H2
    H --> H3
```

```mermaid
flowchart TD
    subgraph Expansion["Graph Expansion (1-2 hops)"]
        direction LR
        HARD["Hard edges first:
        references, overrides,
        depends_on, uses_term,
        precedes"] --> SOFT["Soft edges:
        topically_related
        (1 hop only)"]
    end

    subgraph Scoring["Graph-Aware Scoring"]
        direction LR
        BOOST["Boost:
        referenced by high scorer,
        overrides candidate,
        date overlap,
        higher authority,
        scope match,
        uses defined term"] --> PENALIZE["Penalize:
        superseded,
        scope mismatch"]
    end

    subgraph Conflict["Conflict Resolution Order"]
        direction TB
        CR1["1. Exact date/event match"]
        CR2["2. Non-superseded over superseded"]
        CR3["3. Higher authority_level"]
        CR4["4. Explicit override edge"]
        CR5["5. Scope match"]
        CR6["6. Higher retrieval score"]
        CR1 --> CR2 --> CR3 --> CR4 --> CR5 --> CR6
    end
```

---

### Example: Multi-Document Answer

```mermaid
graph TD
    Q["Query: What is the leave policy?"] --> S["Consumer's vector search"]
    S --> R0["Result 0: Leave Policy Section 2.1
    (direct hit)"]

    R0 -->|"references: see Section 4"| R1["Result 1: Leave Policy Section 4.1
    Maternity Leave
    (graph expanded, depth 1)"]

    R0 -->|"topically_related (0.78)"| R2["Result 2: WFH Policy Section 3.2
    Leave Extension
    (graph expanded, depth 1)"]

    R0 -->|"references: per HR Handbook"| R3["Result 3: HR Processes Section 5.1
    Approval Process
    (graph expanded, depth 1)"]

    R1 -->|"depends_on"| R3

    style R0 fill:#c8e6c9
    style R1 fill:#bbdefb
    style R2 fill:#bbdefb
    style R3 fill:#bbdefb
```

The consumer's LLM receives this subgraph and can synthesize: "Employees get 30 days annual leave (Leave Policy Section 2.1), with 90 days for maternity (Section 4.1). Leave can be extended using WFH policy (WFH Policy Section 3.2) but requires written permission from the immediate manager (HR Processes Section 5.1)."

Every claim traced to a specific source section.

---

## Quick Start

```python
from raag import RAAG

raag = RAAG(
    db=postgres_connection,          # graph and metadata storage
    embed=my_embedding_function,     # your embedding model
    vector_store=my_vector_client,   # your vector DB
    parser=docling_parser,           # optional, defaults to Docling
)

# Ingest a document
result = raag.add_document(file="leave_policy.pdf", metadata={
    "doc_type": "policy",
    "authority_level": 50
})

# Enrich retrieval results
candidates = my_vector_search("leave policy", top_k=5)
enriched = raag.search(
    query_results=candidates,
    as_of_date="2026-01-15",
    max_expansion_hops=1,
    scope_context="full-time"
)

# enriched.results       -> per-result data with provenance
# enriched.result_graph  -> edges between results
# enriched.retrieval_summary -> documents involved, conflicts detected
```

---

## Key Design Decisions

| # | Decision |
|---|----------|
| 1 | No LLM dependency for core extraction. Deterministic, structure-derived. |
| 2 | Consumer provides embedding model. RAAG uses it for embedding AND semantic similarity. |
| 3 | Storage agnostic. Abstract interface with adapters. PostgreSQL reference adapter included. |
| 4 | Embeddings stored in consumer's vector store, not in RAAG metadata. |
| 5 | Search returns a result subgraph (not a flat list). Includes relationship map and retrieval summary. |
| 6 | Graph expansion: default 1 hop, max 2 hops. Configurable per query. |
| 7 | Multi-lingual support delegated to consumer's embedding model capability. |
| 8 | spaCy for structural NLP only (POS tagging, dependency parsing). Not for similarity. |
| 9 | Fuzzy string matching (Levenshtein) for OCR correction. Normalized on save, raw preserved. |

---

## Relationship Types

| Edge | Detection | Example |
|------|-----------|---------|
| `references` | Regex, NLP, semantic | "as defined in Section 4.2" |
| `overrides` | Phrase patterns | "notwithstanding Section 3" |
| `supersedes` | Metadata, filenames | Rev.B supersedes Rev.A |
| `amends` | Metadata, patterns | Amendment 3 amends the Master Agreement |
| `defines` | Definition section patterns | Section 1.1 defines "Force Majeure" |
| `uses_term` | Regex on known term list | Section 8.3 uses "Force Majeure" |
| `depends_on` | Cross-reference patterns | Clause requires another clause |
| `precedes` | Sequential step patterns | Step 1 before Step 2 |
| `applies_to` | Date patterns | "purchases between 2000-2005" |
| `effective_on` | Date patterns | "effective from January 1, 2024" |
| `topically_related` | Jaccard on topic arrays | Leave Policy and WFH Policy share topics |
| `applies_to_scope` | Scope phrase patterns | "applies to full-time employees" |

---

*RAAG: Relationship-Aware Augmented Generation. By [AALA AI](https://aala.ai).*
