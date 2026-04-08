# 02 - OpenEnv From Scratch

Now that RL basics are clear, this chapter explains OpenEnv specifically.

## 1. What OpenEnv Standardizes

OpenEnv gives a standard contract so all environments look similar to agents and evaluators.

Core API:

- `reset()`
- `step(action)`
- `state()`

This is why validators can test many environments the same way.

## 2. Mandatory Building Blocks

A production-ready OpenEnv environment usually needs:

1. Typed models

- Action
- Observation
- Reward/StepResult
- State

2. Environment engine

- reset
- step
- state

3. Task definitions

- multiple tasks
- increasing difficulty

4. Deterministic graders

- normalized score in [0, 1]

5. API server

- FastAPI endpoints (`/reset`, `/step`, `/state`, `/health`)

6. Packaging and deployment

- `openenv.yaml`
- `Dockerfile`
- reproducible dependencies

## 3. Our Environment Design (Email Triage)

### Observation Includes

- task id and difficulty
- current email text
- progress counters
- recent decisions
- last feedback hint

### Action Includes

- category
- priority
- team
- mark_spam
- response_template

### Step Output Includes

- updated observation
- reward
- done flag
- info dictionary

## 4. Why Typed Models Matter

Typed models prevent invalid payloads from silently entering your system.

Benefits:

- safer API contracts
- easier debugging
- cleaner auto-validation
- better tool and IDE support

## 5. Why Deterministic Tasks Matter

Hackathon scoring needs reproducibility.

If the same action sequence gives different score each run, evaluator trust breaks.

We use fixed fixtures in JSON and deterministic grading functions.

## 6. What openenv.yaml Does

`openenv.yaml` is the environment manifest used by tooling and validation.

It declares:

- metadata
- runtime mode (docker)
- endpoint paths
- models and task listing
- inference script expectations

## 7. Why We Needed Extra Root Files

OpenEnv push enforces a specific project contract in addition to runtime checks.

That is why root-level files like below were required:

- `__init__.py`
- `client.py`
- `models.py`
- `pyproject.toml`
- `uv.lock`
- `server/app.py`

Even if your runtime works, push can fail without this scaffold contract.

## 8. Two Different Validations You Must Distinguish

1. Runtime/API validation

- Are endpoints alive?
- Does reset/step/state work?

2. Packaging/CLI validation

- Are required files present?
- Is structure compatible with OpenEnv CLI tooling?

You need both to pass for smooth deployment.
