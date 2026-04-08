# 03 - Build This Project Step by Step (Beginner Friendly)

This chapter rebuilds the project from first step to final validation.

## Step 0 - Prerequisites

Install:

- Python 3.10+ (3.11 or 3.12 recommended)
- Git
- Docker

Create and activate venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Step 1 - Start with the Contract, Not Code

Before coding, lock down required outputs:

- OpenEnv API (`reset`, `step`, `state`)
- 3 tasks (easy, medium, hard)
- deterministic grader in [0, 1]
- `inference.py` at repo root
- Docker + `openenv.yaml`

This avoids late-stage rework.

## Step 2 - Scaffold Files and Folders

Create core structure:

- `src/email_triage_env/`
- `src/email_triage_env/fixtures/`
- `tests/`
- root config files

Install dependencies:

```bash
pip install -r requirements.txt
```

## Step 3 - Define Task Fixtures First

Put deterministic task cases in JSON:

- easy inbox
- medium mixed inbox
- hard ambiguous inbox

Why this first?

- environment logic depends on task shape
- grader logic depends on expected labels

## Step 4 - Build Typed Data Models

Create models for:

- priority/category/team enums
- expectation schema
- email case schema
- action schema
- observation/state/step result schemas

This gives strict contracts and prevents payload drift.

## Step 5 - Build Grading Logic

Create deterministic score components:

- category correctness
- priority distance scoring
- team correctness
- spam correctness
- keyword quality

Then combine into weighted step score and final task score.

## Step 6 - Build Environment Engine

Implement:

1. `reset(task_name)`

- load task
- clear history
- set step counters
- return first observation

2. `step(action)`

- score action
- apply penalties
- update decision history
- compute done
- return step result

3. `state()`

- return deep copy of current state

## Step 7 - Build FastAPI Endpoints

Expose:

- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`
- optional `GET /tasks`

Checkpoint:

```bash
uvicorn email_triage_env.server:app --app-dir src --host 127.0.0.1 --port 7860
curl -s http://127.0.0.1:7860/health
```

## Step 8 - Add Required inference.py

Your script must:

- use OpenAI client
- read `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- print strict lines in order:
  - `[START]`
  - `[STEP]` per action
  - `[END]` always

You can fallback to heuristic actions if token/model call fails.

## Step 9 - Add Tests Early

Minimum checks:

- environment reset and step path works
- final score in [0, 1]
- grader deterministic for same decisions

Run:

```bash
python -m pytest -q
```

## Step 10 - Add OpenEnv Packaging Compatibility

For OpenEnv CLI push flow, include required root files:

- `__init__.py`
- `client.py`
- `models.py`
- `server/app.py`
- `pyproject.toml`
- `uv.lock`

Generate lock file:

```bash
uv lock
```

## Step 11 - Validate Locally

```bash
openenv validate
python inference.py
```

If both pass, you are close.

## Step 12 - Deploy and Submit

- push GitHub repo
- push Space
- verify `/health` and `/reset`
- submit GitHub URL + Space URL

## Build Order Summary

Use this order every time:

1. Task fixtures
2. Typed models
3. Grader
4. Environment loop
5. API endpoints
6. Inference script
7. Tests
8. Packaging/deploy compatibility
9. Submission
