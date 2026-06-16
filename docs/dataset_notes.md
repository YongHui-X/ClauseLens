# CUAD Dataset Notes

## What Is Present Locally

Downloaded files:

```text
data/cuad/CUAD_v1/master_clauses.csv
data/cuad/CUAD_v1/CUAD_v1.json
data/cuad/CUAD_v1/full_contract_txt/Part_I
data/cuad/CUAD_v1/full_contract_txt/Part_II
```

Current local TXT contracts:

```text
Part_I: 100
Part_II: 100
Total: 200
```

The `master_clauses.csv` file contains:

```text
510 labeled contract rows
83 columns
```

Of those 510 labeled rows, 194 currently match local TXT files by filename stem.

## Why There Are Many Empty Lists

CUAD stores clause evidence as lists.

```text
[] = no clause evidence found
['...'] = evidence text found
```

For ClauseLens, do not embed or show `[]`. Use it only as a signal that a clause type was absent.

## Strong Starter Clause Types

Best first clause types based on enough positive examples in the local TXT subset:

```text
Anti-Assignment
Cap On Liability
License Grant
Audit Rights
Termination For Convenience
Post-Termination Services
Insurance
Exclusivity
Minimum Commitment
Revenue/Profit Sharing
```

Positive label counts in the local TXT-matching subset:

```text
Anti-Assignment: 143
Cap On Liability: 104
License Grant: 99
Audit Rights: 85
Termination For Convenience: 76
Post-Termination Services: 72
Insurance: 66
Exclusivity: 65
Minimum Commitment: 58
Revenue/Profit Sharing: 53
```

## Recommended First Prototype Scope

Start with:

```text
20-50 contracts
5 clause types
text-file citations
no PDF page citations yet
no local LLM answer generation yet
```

Suggested first 5 clause types:

```text
Anti-Assignment
Cap On Liability
License Grant
Audit Rights
Termination For Convenience
```

These are frequent enough to make retrieval testing meaningful.

## What To Learn Next

Before building Qdrant ingestion, manually inspect:

```text
1. A few TXT contracts
2. Matching rows in master_clauses.csv
3. The evidence text for selected clause types
4. Whether the evidence text appears exactly inside the TXT file
```

This tells you how easy or hard it will be to map CUAD labels back to contract chunks.

