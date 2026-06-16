# ClauseLens

ClauseLens is a contract-intelligence RAG prototype for finding and citing clause-level evidence in legal agreements. It turns CUAD contract labels into searchable evidence records, embeds them with Sentence Transformers, stores them in Qdrant, and exposes the retrieval layer through a CLI, FastAPI service, Streamlit demo, and repeatable evaluation script.

This is a portfolio-grade retrieval foundation. It does not provide legal advice and does not generate LLM answers yet; every result is grounded in retrieved source evidence.

## Portfolio Pitch

- Built a contract-intelligence RAG prototype over CUAD using Sentence Transformers and Qdrant.
- Implemented metadata-filtered semantic search with source-grounded clause citations.
- Added FastAPI and Streamlit demo surfaces plus retrieval evaluation metrics.

## Architecture

```text
CUAD CSV + TXT contracts
        |
        v
scripts/prepare_cuad_subset.py
        |
        v
data/processed/starter_clause_evidence.jsonl
        |
        v
scripts/index_qdrant.py --> SentenceTransformer embeddings --> Qdrant
        |
        v
app/rag.py shared retrieval helpers
        |
        +--> scripts/search_qdrant.py
        +--> app/api.py FastAPI service
        +--> app/streamlit_app.py Streamlit demo
        +--> evaluation/eval.py retrieval metrics
```

## What It Does

- Reads CUAD `master_clauses.csv` and matching contract TXT files.
- Extracts starter clause evidence for selected legal clause types.
- Embeds evidence text with `BAAI/bge-small-en-v1.5`.
- Stores vectors and citation metadata in embedded local Qdrant by default.
- Searches by natural-language query with optional clause-type filtering.
- Returns source PDF, TXT path, document ID, answer label, score, and evidence text.
- Reports retrieval metrics: pass count, clause-type MRR, nDCG, top-k hit rate, and keyword coverage.

Starter clause types:

```text
Anti-Assignment
Audit Rights
Cap On Liability
License Grant
Termination For Convenience
```

## Project Structure

```text
app/
  api.py                  FastAPI retrieval service
  streamlit_app.py        local demo UI
  rag.py                  shared Qdrant and embedding helpers
  cuad.py                 CUAD data preparation helpers

scripts/
  prepare_cuad_subset.py  creates starter JSONL evidence records
  index_qdrant.py         embeds and indexes records into Qdrant
  search_qdrant.py        searches indexed evidence from the terminal

evaluation/
  cases.py                loads retrieval evaluation cases
  eval.py                 runs retrieval metrics against Qdrant
  tests.jsonl             retrieval test cases

tests/                    unit tests for data prep, retrieval, eval, and API
docs/                     setup notes, dataset notes, developer notes, plan
```

## Setup

Use Python 3.11. The local conda environment in this workspace has the needed dependencies:

```powershell
.\.conda-clauselens\python.exe -m pytest
```

For a fresh environment:

```powershell
python -m pip install -r requirements.txt
```

The `.venv` folder in this workspace may not contain the full dependency set. If tests fail with missing packages, install `requirements.txt` into that environment or use `.conda-clauselens`.

## Prepare Data

Expected CUAD files:

```text
data/cuad/CUAD_v1/master_clauses.csv
data/cuad/CUAD_v1/CUAD_v1.json
data/cuad/CUAD_v1/full_contract_txt/Part_I
data/cuad/CUAD_v1/full_contract_txt/Part_II
```

Create the starter evidence records:

```powershell
python scripts\prepare_cuad_subset.py
```

This writes:

```text
data/processed/starter_clause_evidence.jsonl
data/processed/starter_summary.json
```

## Index Into Qdrant

Embedded local Qdrant mode does not require Docker:

```powershell
python scripts\index_qdrant.py --qdrant-path data/qdrant_local --recreate
```

Server mode expects Qdrant at `http://localhost:6333`:

```powershell
python scripts\index_qdrant.py --recreate
```

With Docker installed:

```powershell
docker run -p 6333:6333 -p 6334:6334 -v ${PWD}\data\qdrant_storage:/qdrant/storage qdrant/qdrant
```

## Run Search

CLI search:

```powershell
python scripts\search_qdrant.py "Does the contract restrict assignment?"
```

Filter by clause type:

```powershell
python scripts\search_qdrant.py "What audit rights does the customer have?" --clause-type "Audit Rights"
```

## Run The API

Start FastAPI:

```powershell
uvicorn app.api:app --reload
```

Health check:

```powershell
curl http://localhost:8000/health
```

Search request:

```powershell
curl -X POST http://localhost:8000/search `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"Does the contract restrict assignment?\",\"clause_type\":\"Anti-Assignment\",\"limit\":5}"
```

Example response shape:

```json
{
  "query": "Does the contract restrict assignment?",
  "clause_type": "Anti-Assignment",
  "limit": 5,
  "result_count": 1,
  "results": [
    {
      "score": 0.87,
      "clause_type": "Anti-Assignment",
      "source_pdf": "Example.pdf",
      "source_txt": "data/cuad/CUAD_v1/full_contract_txt/Part_I/Example.txt",
      "document_id": "Example",
      "answer": "Yes",
      "text": "This Agreement may not be assigned without consent..."
    }
  ]
}
```

## Run The Streamlit Demo

```powershell
streamlit run app\streamlit_app.py
```

The UI provides query search, clause-type filtering, top-k controls, citation-rich result cards, and a compact evaluation panel. To populate the evaluation panel, save an eval report first:

```powershell
python evaluation\eval.py --top-k 5 --output data\processed\eval_results.json
```

Screenshot placeholder:

```text
docs/assets/streamlit-demo.png
```

## Evaluation

Run retrieval evaluation after indexing:

```powershell
python evaluation\eval.py --top-k 5
```

Save detailed rows:

```powershell
python evaluation\eval.py --top-k 5 --output data\processed\eval_results.json
```

The evaluation checks whether natural-language contract questions retrieve the expected clause type and evidence keywords in the top-k results.

## Tests

Run:

```powershell
python -m pytest
python -m py_compile app\api.py app\cuad.py app\rag.py app\streamlit_app.py scripts\prepare_cuad_subset.py scripts\index_qdrant.py scripts\search_qdrant.py evaluation\cases.py evaluation\eval.py
```

Current tests cover:

- CUAD filename and evidence parsing helpers.
- starter-record selection.
- retrieval query validation and Qdrant call shape.
- retrieval evaluation scoring and export.
- FastAPI health, clause type, search, and validation endpoints.

## Current Status

Implemented:

- CUAD evidence extraction and starter JSONL generation.
- embedded-local and server Qdrant indexing.
- reusable retrieval helpers.
- CLI search.
- FastAPI search service.
- Streamlit demo UI.
- retrieval evaluation CLI and JSONL test cases.
- unit tests for core behavior and API endpoints.

Next:

- full-contract chunking with `start_char` and `end_char`.
- reranking for harder semantic queries.
- grounded LLM answer generation with citations.
- citation correctness and answer faithfulness evaluation.
- deployment packaging.

## Notes

ClauseLens is a learning and portfolio project, not a legal advice tool. Search results and future generated answers should always be checked against the original contract text.
