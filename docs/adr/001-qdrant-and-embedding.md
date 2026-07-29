# ADR-001: Keep Qdrant and benchmark embedding independently

## Status
Accepted

## Date
2026-07-29

## Context
LitAgent already uses Qdrant named dense vectors, BM25 sparse vectors and
server-side RRF. The current paper corpus and retrieval benchmark are not yet
large enough to justify a vector-store migration. Embedding quality and vector
storage are independent decisions.

## Decision drivers
- Preserve the working hybrid retrieval path while Phase 14 builds reproducible
  corpus and benchmark evidence.
- Avoid migration cost without measured retrieval or operational benefit.
- Keep embedding experiments isolated through versioned collection identities.

## Decision
- Keep Qdrant as the Paper and Claims serving store.
- Persist Qdrant through a named Docker volume.
- Treat indexes as rebuildable from manifest, raw assets and CorpusState.
- Keep the current all-MiniLM baseline until Phase 14.3 compares it with
  SPECTER2 on versioned collections.
- Never mix vectors produced by incompatible embedding models.

## Consequences
- Phase 14 invests in corpus, provenance, benchmark and reproducibility instead
  of migrating to pgvector or FAISS.
- Every benchmark records corpus/schema/parser/chunking/model versions.
- A model switch creates a new collection identity.

## Rejected alternatives
- pgvector migration before benchmark evidence.
- FAISS as an additional local index.
- Selecting SPECTER2 only because it is paper-domain-specific.

## Reconsider when
- A versioned benchmark shows a stable quality/cost gain from another embedding.
- Qdrant no longer satisfies measured filtering, latency, persistence or
  operational requirements.
- Consolidating into PostgreSQL produces evidence-backed operational savings
  that outweigh migration and retrieval trade-offs.

An embedding change does not supersede this ADR. A vector-store migration must
be recorded in a new ADR that marks this decision as superseded.
