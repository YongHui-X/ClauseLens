# ClauseLens Project Plan

## Phase 1: Dataset Understanding

Goal: understand CUAD before building anything.

Questions to answer:

- What columns exist in `master_clauses.csv`?
- Which columns are clause labels?
- Which columns are normalized answers?
- How do filenames in the CSV map to text files?
- Which 20-50 contracts should be used for the first prototype?

Deliverable:

```text
small selected contract set
chosen metadata fields
chosen clause types
```

Current implementation:

```text
scripts/prepare_cuad_subset.py
```

This script prepares clean starter clause evidence records from CUAD.

## Phase 2: Chunking Design

Goal: decide how contract text becomes retrievable chunks.

Each chunk should eventually have:

```text
document_id
source_file
chunk_id
chunk_text
clause_type
start_char
end_char
```

If using PDFs later, add:

```text
page_number
```

Deliverable:

```text
chunking rules
sample chunks inspected manually
```

## Phase 3: Qdrant Design

Goal: design the first collection.

Collection:

```text
contracts_chunks
```

Each Qdrant point should represent one chunk.

Payload fields:

```text
document_id
source_file
contract_type
vendor
clause_type
chunk_text
start_char
end_char
```

Later payload fields:

```text
effective_date
renewal_date
value
product
page_number
risk_flags
```

Deliverable:

```text
one collection design
payload schema
```

Implementation note:

```text
scripts/index_qdrant.py
scripts/search_qdrant.py
```

These files are placeholders for your own Qdrant indexing and search implementation.

## Phase 4: Retrieval

Goal: prove search works before answer generation.

Retrieval types:

- vector search
- metadata filter search
- vector search plus metadata filters

Example tests:

```text
termination notice
auto renewal
governing law
cap on liability
audit rights
```

Deliverable:

```text
manual retrieval checks with correct source chunks
```

## Phase 5: Grounded Answering

Goal: answer only from retrieved evidence.

Answer rules:

- cite source file
- cite chunk or page where available
- say when evidence is missing
- separate facts from interpretation

Deliverable:

```text
answers with citations
```

## Phase 6: Evaluation

Goal: make the project portfolio-grade.

Track:

- retrieval accuracy
- citation correctness
- answer faithfulness
- filter correctness

Deliverable:

```text
small evaluation set
repeatable score report
```
