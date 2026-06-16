# ClauseLens Setup

## 1. Use Python 3.11

This project uses a local conda environment at:

```text
.conda-clauselens
```

Activate it with:

```powershell
conda activate "C:\Users\User\OneDrive - National University of Singapore\Latest personal projects\ClauseLens\.conda-clauselens"
```

Note: Conda may warn that this path contains spaces. It still works, but if dependency installs become painful later, move the project to a shorter path such as `C:\projects\ClauseLens`.

## 2. Install Dependencies

After activating `.conda-clauselens`, install:

```powershell
python -m pip install -r requirements.txt
```

## 3. Start Qdrant

Docker was not detected during setup. Install Docker Desktop first, then use the command below.

If Docker Desktop is installed:

```powershell
docker run -p 6333:6333 -p 6334:6334 -v ${PWD}\data\qdrant_storage:/qdrant/storage qdrant/qdrant
```

Qdrant dashboard:

```text
http://localhost:6333/dashboard
```

## 4. Dataset Check

Expected starter files:

```text
data\cuad\CUAD_v1\master_clauses.csv
data\cuad\CUAD_v1\CUAD_v1.json
data\cuad\CUAD_v1\full_contract_txt\Part_I
data\cuad\CUAD_v1\full_contract_txt\Part_II
```

You should have 200 text contracts total across `Part_I` and `Part_II`.

## 5. Recommended First Build Target

Do not start with all 200 contracts.

Start with:

```text
20 contracts
5-10 clause types
one Qdrant collection
simple vector search
metadata filters
citations from filename and chunk location
```

Then expand.
