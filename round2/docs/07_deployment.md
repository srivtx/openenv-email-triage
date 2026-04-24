# Chapter 7: Deployment — Docker, HF Spaces, and OpenEnv

This chapter explains how we package the environment and make it accessible to judges, training scripts, and the OpenEnv validator.

---

## 7.1 Why We Need Deployment

Our environment runs on your laptop right now. But the hackathon needs:
- Judges to access it from their computers
- The OpenEnv validator to test it
- Training scripts to call it remotely

Solution: Deploy it as a **web API** on HuggingFace Spaces.

```
Before deployment:         After deployment:
Your laptop only           Anyone on the internet
  ↓                          ↓
python environment.py      https://srivtx-openenv-conflict-resolver-v2.hf.space/
```

---

## 7.2 FastAPI — Turning Python into a Web Server

FastAPI converts Python functions into HTTP endpoints.

```python
# Without FastAPI — only works on your machine
env = PersonalAssistantConflictEnv()
result = await env.reset()  # Only you can call this

# With FastAPI — anyone can call it via HTTP
from fastapi import FastAPI
app = FastAPI()

env = PersonalAssistantConflictEnv()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/reset")
async def reset(body: dict):
    result = await env.reset(task_name=body.get("task_name"))
    return result.model_dump()

@app.post("/step")
async def step(action: ConflictAction):
    result = await env.step(action)
    return result.model_dump()

@app.get("/state")
async def state():
    s = await env.state()
    return s.model_dump()
```

**Now anyone can call**:
```bash
curl http://localhost:7860/health           # → {"status": "ok"}
curl -X POST http://localhost:7860/reset    # → {observation, reward, done}
curl -X POST http://localhost:7860/step \
  -d '{"intent":"reschedule_event",...}'    # → {observation, reward, done}
```

**Why FastAPI?** OpenEnv requires 4 HTTP endpoints: `/health`, `/reset`, `/step`, `/state`. FastAPI is the standard for Python web APIs — it's fast, has automatic docs, and handles validation.

---

## 7.3 Docker — Packaging Everything

Docker puts your entire application (code + dependencies + Python version) into a portable container.

**The problem without Docker**: 
- Your code needs Python 3.11, pydantic 2.5, fastapi 0.110
- Judge's machine has Python 3.9, pydantic 1.10
- Everything breaks

**With Docker**:
- You define exactly what's inside the container
- It runs the same everywhere — your laptop, HF Spaces, judge's machine

Here's our Dockerfile:

```dockerfile
# Start from Python 3.11
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy our code into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Tell Docker to expose port 7860 (HF Spaces requirement)
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
```

**Line by line**:
| Line | What It Does | Why |
|---|---|---|
| `FROM python:3.11-slim` | Uses Python 3.11 as the base | Consistent Python version |
| `WORKDIR /app` | Sets working directory | Organization |
| `COPY . .` | Copies all project files into container | Container needs our code |
| `RUN pip install ...` | Installs pydantic, fastapi, etc. | Container needs our dependencies |
| `EXPOSE 7860` | Opens port 7860 | HF Spaces talks to this port |
| `CMD [...]` | Starts the web server | This runs when container starts |

---

## 7.4 HuggingFace Spaces — Free Hosting

HuggingFace Spaces is like GitHub Pages but for ML apps. It runs your Docker container for free.

**What you need**:
1. A HuggingFace account (you have: `srivtx`)
2. A `README.md` with frontmatter metadata
3. Your code files

**The README.md frontmatter**:
```yaml
---
title: Personal Assistant Conflict Resolver v2
emoji: '🗓️'
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---
```

| Field | What It Means |
|---|---|
| `title` | Display name on HF |
| `sdk: docker` | "Run this as a Docker container" (not Gradio/Streamlit) |
| `app_port: 7860` | Which port to expose |
| `colorFrom/colorTo` | Card colors on HF (must be valid CSS color names) |

---

## 7.5 The OpenEnv Push Command

Instead of manually uploading files to HF, we use the `openenv` CLI:

```bash
openenv push -r srivtx/openenv-conflict-resolver-v2 --no-interface
```

**What it does**:
1. Validates your `openenv.yaml` (checks all required fields)
2. Authenticates with your HF token
3. Creates the Space if it doesn't exist
4. Uploads all files
5. HF Spaces builds and deploys the Docker container

**The `--no-interface` flag**: Skips adding a Gradio web UI. Our environment is API-only.

---

## 7.6 openenv.yaml — The Environment Manifest

This file tells OpenEnv about your environment:

```yaml
schema_version: '0.1'
name: personal-assistant-conflict-resolver-v2
benchmark: personal_assistant_conflict_resolution
description: |
  A multi-step conflict resolution environment where an agent manages 
  cascading scheduling conflicts across work, family, travel, and finance.

runtime:
  type: docker
  dockerfile: Dockerfile
  container_port: 7860

api:
  health: { method: GET,  path: /health }
  reset:  { method: POST, path: /reset  }
  step:   { method: POST, path: /step   }
  state:  { method: GET,  path: /state  }

implementation:
  module: assistant_conflict_env.environment
  class: PersonalAssistantConflictEnv

tasks:
  - id: easy_evening_planner
    difficulty: easy
  - id: medium_multi_party_negotiation
    difficulty: medium
  - id: hard_cascade_replanning
    difficulty: hard
```

**Why each section matters**:

| Section | What The Validator Checks |
|---|---|
| `name` | Must be unique |
| `runtime` | Dockerfile exists, port is correct |
| `api` | All 4 endpoints defined |
| `implementation` | Module + class are importable |
| `tasks` | At least 1 task defined |

---

## 7.7 The openenv validate Command

```bash
openenv validate
```

This checks:
- ✅ `openenv.yaml` exists and has valid schema
- ✅ Dockerfile exists
- ✅ The Python module is importable
- ✅ The class has `reset()`, `step()`, `state()` methods
- ✅ Tasks are defined

**If validation fails**: Your submission won't be accepted. Always run this before pushing.

---

## 7.8 Verifying the Deployment

After pushing, verify your Space is alive:

```bash
# Check health
curl https://srivtx-openenv-conflict-resolver-v2.hf.space/health
# → {"status": "ok"}

# Reset an episode
curl -X POST https://srivtx-openenv-conflict-resolver-v2.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "easy_evening_planner"}'
# → {observation: ..., reward: 0.0, done: false}
```

---

## 7.9 The Complete File Structure

```
round2/
├── README.md                  # HF Spaces metadata
├── Dockerfile                 # How to build the container
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project metadata
├── openenv.yaml               # OpenEnv manifest
├── server.py                  # FastAPI web server
├── inference.py               # Runs model against environment
├── src/
│   └── assistant_conflict_env/
│       ├── __init__.py        # Package init
│       ├── environment.py     # The RL environment (reset/step/state)
│       ├── models.py          # Pydantic data models
│       ├── graders.py         # Reward scoring logic
│       ├── tasks.py           # Task loading from JSON
│       └── fixtures/
│           └── conflict_cases.json  # The conflict scenarios
├── scripts/
│   └── train_grpo_stub.py    # GRPO training script
├── notebooks/
│   └── train_grpo_colab.ipynb # Colab notebook for training
├── tests/
│   ├── test_environment.py   # Environment tests
│   └── test_graders.py       # Grader tests
└── docs/                     # This documentation!
```

**Every file has a purpose**. Nothing is there "just because."

---

## What's Next?

Chapter 8 ties everything together — the complete architecture, how all pieces connect, and the full journey from Round 1 to Round 2.
