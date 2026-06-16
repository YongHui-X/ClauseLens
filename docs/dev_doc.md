# ClauseLens Developer Notes

This document explains what the current ClauseLens code is doing, how the data
flows through the project, and what the important Python/RAG syntax means.

## Big Picture

ClauseLens is being built as a contract intelligence RAG system.

RAG means Retrieval-Augmented Generation:

1. Prepare contract data.
2. Embed contract evidence into vectors.
3. Store vectors in Qdrant.
4. Search Qdrant for relevant evidence.
5. Later, give that evidence to an LLM to answer questions with citations.

The current starter milestone is not the full final app. The starter milestone is:

1. Prepare CUAD evidence records.
2. Embed those records.
3. Store them in Qdrant.
4. Run a basic semantic search.

## Main Files

### `app/cuad.py`

This is the CUAD data preparation layer.

It does not embed text.
It does not call Qdrant.
It does not call an LLM.

Its job is to read CUAD's raw dataset and convert it into cleaner records that
are easier to embed later.

Important inputs:

```text
data/cuad/CUAD_v1/master_clauses.csv
data/cuad/CUAD_v1/full_contract_txt/Part_I/*.txt
data/cuad/CUAD_v1/full_contract_txt/Part_II/*.txt
```

Important output:

```text
data/processed/starter_clause_evidence.jsonl
```

That JSONL file contains records like:

```json
{
  "id": "SomeAgreement::Anti-Assignment::1",
  "document_id": "SomeAgreement",
  "source_pdf": "SomeAgreement.pdf",
  "source_txt": "data/cuad/CUAD_v1/full_contract_txt/Part_I/SomeAgreement.txt",
  "clause_type": "Anti-Assignment",
  "answer": "Yes",
  "text": "This Agreement may not be assigned without consent..."
}
```

### `app/rag.py`

This is the shared RAG helper layer.

It defines:

```python
COLLECTION = "contracts_clause_evidence"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
QDRANT_PATH = Path("data/qdrant_local")
QDRANT_URL = "http://localhost:6333"
```

It also defines helper functions:

```python
create_qdrant_client(...)
ensure_collection(...)
load_embedding_model(...)
```

Important: `rag.py` does not run embeddings by itself. It only defines the model
name and gives us a function to load the model when we explicitly want to.

This is intentional. Loading an embedding model can be slow, and connecting to
Qdrant can fail if Qdrant is not running. So we avoid doing that automatically
at import time.

### `scripts/prepare_cuad_subset.py`

This script uses `app/cuad.py` to create the starter JSONL file.

Run:

```powershell
.\.conda-clauselens\python.exe scripts\prepare_cuad_subset.py
```

It writes:

```text
data/processed/starter_clause_evidence.jsonl
data/processed/starter_summary.json
```

### `scripts/index_qdrant.py`

This script embeds the prepared JSONL records and stores them in Qdrant.

No Docker version:

```powershell
.\.conda-clauselens\python.exe scripts\index_qdrant.py --qdrant-path data/qdrant_local --recreate
```

Docker/server Qdrant version:

```powershell
.\.conda-clauselens\python.exe scripts\index_qdrant.py --recreate
```

The success output should look like:

```text
Upserted 463 records
Collection count: 463
```

### `notebooks/clauselens_ingestion.ipynb`

This notebook is the learning version of ingestion.

It walks through:

1. Load the prepared JSONL records.
2. Connect to embedded local Qdrant.
3. Load the embedding model.
4. Embed all evidence texts.
5. Upsert the vectors into Qdrant.
6. Run a sanity search.

## CUAD CSV vs TXT Files

CUAD gives us both a CSV and text files.

The CSV contains labels and evidence for each contract. Its `Filename` column
uses PDF filenames, for example:

```text
Example Agreement.pdf
```

The full text files use `.txt`, for example:

```text
Example Agreement.txt
```

We are not converting PDF files to TXT in `cuad.py`.

Instead, CUAD already provides the TXT files. `cuad.py` uses the PDF filename
from the CSV as an identifier, removes the `.pdf` extension, and matches it to
the `.txt` file with the same stem.

Example:

```python
Path("Example Agreement.pdf").stem
```

returns:

```text
Example Agreement
```

Then the code can match:

```text
Example Agreement.pdf
Example Agreement.txt
```

We still store `source_pdf` because it is useful for citations and traceability.

## What To Search For In Excel

In CUAD Excel/CSV, search the `Filename` column.

If your TXT file is:

```text
ABILITYINC_06_15_2020-EX-4.25-SERVICES AGREEMENT.txt
```

search for:

```text
ABILITYINC_06_15_2020-EX-4.25-SERVICES AGREEMENT.pdf
```

or just:

```text
ABILITYINC_06_15_2020-EX-4.25-SERVICES AGREEMENT
```

That row contains the labels and evidence for that contract.

## Python Syntax Notes

### Type Hints

Example:

```python
def filename_stem(filename: str) -> str:
```

Meaning:

```text
filename: str
```

The input parameter should be a string.

```text
-> str
```

The function returns a string.

Python does not enforce these by default. They help humans and editors understand
the code.

### `str | None`

Example:

```python
def parse_evidence_list(raw_value: str | None) -> list[str]:
```

This means `raw_value` can be either:

```text
str
None
```

This is useful because `row.get(...)` may return `None` if a CSV column is
missing.

### `list[dict[str, object]]`

Example:

```python
records: list[dict[str, object]] = []
```

Meaning:

```text
records is a list
each item is a dictionary
each dictionary has string keys
the values can be mixed Python objects
```

We use `object` because a record may contain strings now, but later could contain
numbers, booleans, or lists.

### `-> None`

Example:

```python
def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
```

`-> None` means the function does not return a useful value. It is used for its
side effect, such as writing a file.

### `@dataclass`

Example:

```python
@dataclass(frozen=True)
class CuadPaths:
    root: Path = Path("data/cuad/CUAD_v1")
```

`@dataclass` asks Python to automatically create common class methods, including
an `__init__`.

`frozen=True` means the object is read-only after creation.

So this:

```python
paths = CuadPaths()
```

creates an object with:

```python
paths.root
```

set to:

```text
data/cuad/CUAD_v1
```

### `@property`

Example:

```python
@property
def master_clauses_csv(self) -> Path:
    return self.root / "master_clauses.csv"
```

This lets you write:

```python
paths.master_clauses_csv
```

instead of:

```python
paths.master_clauses_csv()
```

It looks like an attribute, but it is computed by a method.

### `Path` and `/`

Example:

```python
self.root / "full_contract_txt" / "Part_I"
```

`Path` overloads the `/` operator to join path pieces.

This is cleaner than manually writing:

```python
"data/cuad/CUAD_v1/full_contract_txt/Part_I"
```

It also works better across operating systems.

### `with ... as file`

Example:

```python
with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
    return list(csv.DictReader(file))
```

This opens the file and automatically closes it when the block ends.

### List Comprehension

Example:

```python
return [
    row
    for row in rows
    if filename_stem(row["Filename"]) in text_lookup
]
```

Meaning:

```text
make a new list
take each row from rows
keep only rows whose filename matches a TXT file
```

### Set Comprehension

Example:

```python
selected_documents = {
    document_id
    for document_id, _ in document_scores.most_common(max_contracts)
}
```

This creates a set. Sets remove duplicates and make membership checks fast.

### `enumerate`

Example:

```python
for index, text in enumerate(evidence_items, start=1):
```

This loops over `evidence_items`, while also giving a counter.

If `evidence_items` has three items:

```text
index = 1, text = first item
index = 2, text = second item
index = 3, text = third item
```

### f-strings

Example:

```python
f"{document_stem}::{clause_type}::{index}"
```

This inserts variable values into a string.

### Keyword-Only Arguments With `*`

Example from `rag.py`:

```python
def create_qdrant_client(
    *,
    url: str | None = None,
    path: str | Path | None = None,
) -> QdrantClient:
```

The `*` means `url` and `path` must be passed by name.

Correct:

```python
create_qdrant_client(path=QDRANT_PATH)
```

Not allowed:

```python
create_qdrant_client(QDRANT_PATH)
```

This makes the code clearer because URL mode and path mode are different.

## Qdrant Modes

There are two ways to use Qdrant.

### Server Mode

This requires Qdrant to be running, usually with Docker:

```powershell
docker run -p 6333:6333 -p 6334:6334 -v ${PWD}\data\qdrant_storage:/qdrant/storage qdrant/qdrant
```

Then Python connects with:

```python
client = QdrantClient(url="http://localhost:6333")
```

If Qdrant is not running, you get:

```text
WinError 10061
No connection could be made because the target machine actively refused it
```

### Embedded Local Mode

This does not require Docker.

Python connects directly to a local folder:

```python
client = QdrantClient(path="../data/qdrant_local")
```

For learning, embedded local mode is easier.

## Stable Qdrant Point IDs

The notebook uses this function:

```python
def stable_point_id(raw_id: str) -> int:
    """Turn a text ID into a stable integer ID for Qdrant."""
    digest = hashlib.sha256(raw_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)
```

Purpose:

Qdrant needs every vector point to have an ID. Our original record ID is a long
string:

```text
SomeAgreement::Anti-Assignment::1
```

We convert that text ID into a stable integer.

### Line By Line

```python
def stable_point_id(raw_id: str) -> int:
```

Defines a function.

`raw_id: str` means the input should be a string.

`-> int` means the function returns an integer.

```python
digest = hashlib.sha256(raw_id.encode("utf-8")).digest()
```

`raw_id.encode("utf-8")` converts the string into bytes.

Hash functions work on bytes, not normal Python strings.

`hashlib.sha256(...)` creates a SHA-256 hash. A hash is like a fingerprint of
the input text.

`.digest()` returns the hash as raw bytes.

The same `raw_id` always produces the same hash.

```python
return int.from_bytes(digest[:8], byteorder="big", signed=False)
```

`digest[:8]` takes the first 8 bytes of the hash.

`int.from_bytes(...)` converts those bytes into an integer.

`byteorder="big"` means read the bytes from left to right.

`signed=False` means the number should not be negative.

This call:

```python
stable_point_id(records[0]["id"])
```

takes the first record's text ID and converts it into a stable integer.

Why stable IDs matter:

If you rerun ingestion, the same clause gets the same Qdrant ID. Qdrant can
update/replace that point instead of creating duplicate random IDs.

## Embedding Code Explained

### Extract Texts

```python
texts = [record["text"] for record in records]
```

Each record contains metadata plus text.

This line extracts only the text field from every record.

Example input record:

```python
{
    "id": "...",
    "document_id": "...",
    "source_pdf": "...",
    "clause_type": "Anti-Assignment",
    "answer": "Yes",
    "text": "This Agreement may not be assigned..."
}
```

Result:

```python
[
    "This Agreement may not be assigned...",
    "TO THE MAXIMUM EXTENT...",
    "Subject to the terms and conditions..."
]
```

### Create Embeddings

```python
embeddings = model.encode(
    texts,
    batch_size=32,
    normalize_embeddings=True,
    show_progress_bar=True,
)
```

This is where embeddings are actually created.

`model.encode(texts)` sends each text string into the embedding model.

The model converts each text into a vector, which is a list of numbers.

For:

```text
BAAI/bge-small-en-v1.5
```

each vector has 384 numbers.

Example idea:

```text
"This Agreement may not be assigned..."
```

becomes something like:

```python
[0.012, -0.044, 0.091, ..., 0.006]
```

These numbers represent semantic meaning. Similar texts should have similar
vectors.

Arguments:

```python
batch_size=32
```

Process 32 texts at a time.

```python
normalize_embeddings=True
```

Makes each vector have a consistent length. This works well with cosine
similarity.

```python
show_progress_bar=True
```

Shows progress while embedding.

### Build Qdrant Points

```python
points = [
    PointStruct(
        id=stable_point_id(record["id"]),
        vector=embedding.tolist(),
        payload=record,
    )
    for record, embedding in zip(records, embeddings)
]
```

This creates the objects Qdrant stores.

Each Qdrant point has:

```text
id
vector
payload
```

`id`:

```python
id=stable_point_id(record["id"])
```

Unique stable ID for this clause.

`vector`:

```python
vector=embedding.tolist()
```

The embedding numbers. `.tolist()` converts a NumPy array into a normal Python
list, which Qdrant can store.

`payload`:

```python
payload=record
```

Metadata stored beside the vector. This is how search results can tell us:

```text
source PDF
clause type
original evidence text
answer label
```

`zip(records, embeddings)` pairs each record with its matching embedding:

```text
record 1 -> embedding 1
record 2 -> embedding 2
record 3 -> embedding 3
```

### Upsert Into Qdrant

```python
client.upsert(collection_name=CONTRACT_COLLECTION, points=points)
```

This sends the points into Qdrant.

Upsert means:

```text
if the point ID does not exist: insert it
if the point ID already exists: update/replace it
```

### Verify Count

```python
count = client.count(collection_name=CONTRACT_COLLECTION, exact=True).count
print(f"Upserted {len(points)} records")
print(f"Collection count: {count}")
```

This checks how many points are in the Qdrant collection.

For the starter data, success should be:

```text
Upserted 463 records
Collection count: 463
```

## Sanity Search Explained

The notebook search cell uses:

```python
query = "Does the contract restrict assignment?"
query_vector = model.encode(query, normalize_embeddings=True).tolist()
```

This embeds the user query into the same vector space as the clause evidence.

Then:

```python
results = client.query_points(
    collection_name=CONTRACT_COLLECTION,
    query=query_vector,
    limit=5,
    with_payload=True,
)
```

This asks Qdrant:

```text
Find the 5 stored vectors most similar to this query vector.
Return their payload metadata too.
```

Then the code prints:

```python
for index, point in enumerate(results.points, start=1):
    payload = point.payload
    print(f"Result {index}: score={point.score:.3f}")
    print(f"Clause type: {payload['clause_type']}")
    print(f"Source: {payload['source_pdf']}")
    print(payload["text"][:500])
```

This displays:

1. Search result rank.
2. Similarity score.
3. Clause type.
4. Source contract.
5. First 500 characters of evidence text.

## Current Progress

Approximate status:

```text
Dataset preparation:        mostly done
Starter JSONL creation:     done
RAG helper module:          started and cleaned up
Notebook ingestion flow:    written
Embedding/upsert execution: run it and confirm count
Retrieval/search:           basic sanity search in notebook
LLM answer generation:      not built yet
Frontend/API app:           not built yet
```

For the starter RAG milestone, the next success signal is:

```text
Collection count: 463
```

After that, the next step is to turn the sanity search into a reusable search
function in `app/rag.py` or a new script.

## Recommended Next Steps

1. Open `notebooks/clauselens_ingestion.ipynb`.
2. Run all cells top to bottom.
3. Confirm:

```text
Upserted 463 records
Collection count: 463
```

4. Run the sanity search cell.
5. Inspect whether the returned clauses make sense.
6. Then implement a reusable search function.

