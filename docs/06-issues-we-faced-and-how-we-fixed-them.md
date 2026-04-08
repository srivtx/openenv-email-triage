# 06 - Issues We Faced and How We Fixed Them

This chapter is based on real issues encountered while building and deploying this exact project.

## Issue 1 - Pydantic Build Failure on Python 3.14

### Symptom

`pip install -r requirements.txt` failed while building `pydantic-core`.

### Root Cause

Older pydantic-core builds were incompatible with Python 3.14 toolchain expectations.

### Fix

Use modern Pydantic line:

- `pydantic>=2.12.0,<3.0.0`

### Prevention

Keep Python and dependency versions aligned, especially when Rust-backed wheels are involved.

## Issue 2 - Async Tests Were Skipped

### Symptom

Pytest output showed skipped async tests and unknown mark warnings.

### Root Cause

`pytest-asyncio` missing.

### Fix

Add:

- `pytest-asyncio==0.26.0`
- configure `pytest.ini` with async mode and loop scope.

### Prevention

Any async environment project should include async test plugin by default.

## Issue 3 - OpenEnv CLI Dependency Conflicts

### Symptom

Installing requirements downgraded packages and broke OpenEnv tooling constraints.

### Root Cause

Pinned versions in `requirements.txt` were lower than `openenv-core` requirements.

### Fix

Align pins:

- `openai>=2.7.2,<3.0.0`
- `uvicorn>=0.35,<1.0`
- `python-dotenv>=1.1.0,<2.0`
- include `openenv-core==0.2.3`

### Prevention

Treat OpenEnv core constraints as source of truth for shared runtime libs.

## Issue 4 - Wrong Package Confusion (`openenv` vs `openenv-core`)

### Symptom

Installed package but expected CLI command behavior was missing/inconsistent.

### Root Cause

`openenv` and `openenv-cli` package names on PyPI are not the official full workflow tool used here.

### Fix

Install and use:

- `openenv-core`
- command: `openenv ...`

### Prevention

Always check official repository docs for canonical package name.

## Issue 5 - `openenv validate` Failed Due to Missing Project Files

### Symptom

Validation failed with missing files like:

- `pyproject.toml`
- `uv.lock`
- `server/app.py`
- `client.py`
- root `__init__.py`

### Root Cause

Runtime worked, but OpenEnv packaging contract expected template-compatible structure.

### Fix

Add required root/server scaffolding and run:

```bash
uv lock
openenv validate
```

### Prevention

Run `openenv validate` early, not at submission time.

## Issue 6 - Hugging Face Space Config Error (README Metadata)

### Symptom

Space showed "Missing configuration in README" or invalid `colorFrom`/`colorTo`.

### Root Cause

Missing/invalid README frontmatter for Spaces.

### Fix

Add valid frontmatter at top of README, for example:

```yaml
---
title: OpenEnv Email Triage
emoji: '📧'
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
```

### Prevention

Keep metadata values from allowed options and redeploy.

## Issue 7 - Space URL Looked "Empty"

### Symptom

Space root did not show a fancy interactive UI.

### Root Cause

This project is API-first; root returns JSON health info.

### Fix

Use these endpoints to verify:

- `/health`
- `/docs`
- `/reset`

### Prevention

Understand that OpenEnv evaluation is endpoint-based, not UI-based.

## Issue 8 - Docker Build Failures Under Time Pressure

### Symptom

Local `docker build` failed while deadline was near.

### Root Cause

Usually dependency mismatch or daemon/setup differences.

### Fix Pattern

1. verify daemon (`docker info`)
2. verify dependency lock state
3. keep Dockerfile simple
4. test API endpoints after container boots

## Emergency Checklist (When You Have 5 Minutes)

Run these and stop changing code unless required:

```bash
python -m pytest -q
openenv validate
python inference.py
curl -s https://<space>.hf.space/health
curl -s -X POST https://<space>.hf.space/reset -H 'Content-Type: application/json' -d '{}'
```

If all pass, submit immediately.
