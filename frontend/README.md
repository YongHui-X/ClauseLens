# QFind React frontend

This directory contains the React/Vite chat UI used by the current Cloud Run
deployment. The production FastAPI service serves the built files from
`frontend/dist` at the same origin as the API.

## Local frontend checks

Run from this directory:

```powershell
npm install
npm run check
npm run build
```

The built UI is written to:

```text
frontend/dist
```

Then start the FastAPI app from the repository root:

```powershell
$env:QDRANT_MODE="embedded"
$env:MODEL_WARMUP_ENABLED="false"
$env:SESSION_SIGNING_SECRET="local-dev-session-secret"
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

## Current browser/API behavior

- The frontend bootstraps a signed browser session through `GET /api/session`.
- Chat requests are sent with credentials so the `HttpOnly` session cookie is
  included.
- Protected API routes reject direct unauthenticated calls with `401`.
- The chat UI requests three evidence passages and shows the retrieved evidence
  under each answer.
