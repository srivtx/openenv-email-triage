# Email Triage OpenEnv

Email Triage OpenEnv is a real-world reinforcement-learning environment where an agent triages inbound email by selecting category, urgency, routing team, and abuse/spam status.

The environment is designed for OpenEnv Round 1 requirements:

- Real-world workflow simulation (customer operations inbox triage)
- Typed action, observation, and reward models
- Deterministic task graders with normalized scores in [0.0, 1.0]
- Baseline inference script with strict structured logs
- Dockerized runtime for Hugging Face Spaces deployment

## Why This Environment

Production support teams process high-volume mixed inboxes with billing, account access, legal, abuse, sales, and engineering issues. This environment evaluates whether an agent can route accurately under varied difficulty and ambiguity.

## Action Space

`EmailTriageAction`

- `category`: one of `billing`, `bug`, `account`, `sales`, `abuse`, `legal`, `other`
- `priority`: one of `low`, `normal`, `high`, `urgent`
- `team`: one of `support`, `billing`, `engineering`, `sales`, `trust_safety`, `legal`
- `mark_spam`: boolean
- `response_template`: short text (0-400 chars)

## Observation Space

`EmailTriageObservation`

- Task metadata (`task_id`, `task_difficulty`)
- Progress (`step_index`, `max_steps`, `processed_count`, `remaining_count`)
- Current email (`email_id`, sender, subject, body, expected schema hidden from policy agent usage)
- Rolling history of recent routing decisions
- Last feedback hint for shaping

## Reward Design

Per-step score is composed from deterministic components:

- Category correctness: 35%
- Priority correctness (distance-aware): 25%
- Team correctness: 25%
- Spam flag correctness: 10%
- Response keyword coverage: 5%

Shaped reward formula:

- `reward = clamp(0.15 + 0.85 * step_score - penalties, 0.0, 1.0)`

Penalties include:

- Under-informative short responses
- Repetitive routing behavior
- Over-aggressive escalation bias

This provides dense learning signal, partial credit, and behavior discouragement beyond binary episode success.

## Tasks and Difficulty

1. `easy_priority_routing` (easy)

- Straightforward billing/account/spam routing cases.

2. `medium_mixed_inbox` (medium)

- Mixed queue including engineering incidents, billing disputes, security concerns, sales, and GDPR legal requests.

3. `hard_ambiguous_escalations` (hard)

- Ambiguous multi-signal cases requiring careful escalation and ownership tradeoffs.

Each task has deterministic grader behavior and normalized scores in [0.0, 1.0].

## Project Layout

- `src/email_triage_env/models.py`: typed models
- `src/email_triage_env/tasks.py`: fixture-backed task registry
- `src/email_triage_env/graders.py`: deterministic graders
- `src/email_triage_env/environment.py`: `reset()`, `step()`, `state()` core loop
- `src/email_triage_env/server.py`: API endpoints for validator/runtime
- `src/email_triage_env/fixtures/inbox_cases.json`: reproducible inbox fixtures
- `openenv.yaml`: OpenEnv metadata
- `inference.py`: mandatory baseline script

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run API Locally

```bash
uvicorn email_triage_env.server:app --app-dir src --host 0.0.0.0 --port 7860
```

Health check:

```bash
curl -s http://127.0.0.1:7860/health
```

## Run Baseline Inference

Required environment variables:

- `API_BASE_URL`: LLM API base endpoint
- `MODEL_NAME`: model identifier
- `HF_TOKEN`: API token

Optional:

- `LOCAL_IMAGE_NAME`: local image reference if using `from_docker_image(...)`
- `TASK_NAMES`: comma-separated task ids

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="<token>"
python inference.py
```

The script emits strict stdout format:

- `[START] task=<task_name> env=<benchmark> model=<model_name>`
- `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>`

## Baseline Scores (Example)

Example reproducible baseline using low-temperature routing prompts:

- `easy_priority_routing`: 0.87
- `medium_mixed_inbox`: 0.76
- `hard_ambiguous_escalations`: 0.68

Scores may vary slightly by model snapshot and endpoint, but are clamped to [0, 1].

## Docker

Build:

```bash
docker build -t email-triage-openenv:latest .
```

Run:

```bash
docker run --rm -p 7860:7860 email-triage-openenv:latest
```

## Hugging Face Spaces Deployment

1. Push repository to a Space configured for Docker.
2. Ensure `openenv` tag is present in project metadata.
3. Set Space secrets/variables:
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`
4. Confirm `/health` returns HTTP 200 and `/reset` responds.

## Local Validation Checklist

```bash
pytest -q
python inference.py
docker build -t email-triage-openenv:latest .
```

Also validate `openenv.yaml` via your OpenEnv CLI in the target environment:

```bash
openenv validate
```
