---
title: Personal Assistant Conflict Resolver v2
emoji: '🗓️'
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# personal assistant conflict resolver v2

**an OpenEnv RL environment that teaches LLMs to resolve cascading scheduling conflicts — with real world-state, partial observability, and follow-on conflicts triggered by the agent's own decisions.**

*OpenEnv Hackathon 2026 | Team Agent (1)*

---

## what changed in the v0.3 rebuild

short version: an external code review came in and basically said "this isn't an environment, it's a deployed dataset with a scoring function." they were right. we deleted the fake `1.0` scores and rebuilt the thing properly.

| concern in v0.2                                           | what v0.3 actually does                                                                                              |
|-----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| 15 static conflicts; SFT and eval used the same set       | procedural generator with disjoint **train (seeds 1000-1999)**, **holdout (9000-9099)**, and **adversarial (5000-5009)** seed pools |
| "long-horizon" was independent classification             | `WorldState` carries `calendar`, `pending_clarifications`, `revealed_info`, `cascade_queue` across steps             |
| "partial observability" was pre-labeled                   | two-step clarification cycle: ask first, info revealed on the next step, then resolve with the new info              |
| "cascading conflicts" never cascaded                      | `CascadeRule` generates follow-on conflicts when the agent's action satisfies a trigger (e.g. reschedule past 18:00 spawns a school-pickup conflict) |
| substring-match slot scoring (`"pm"`/`"today"` shortcut)  | regex-strict 24h `HH:MM` parsing with time-distance scoring                                                          |
| keyword-stuffing message scoring                          | length + on-topic verb + word-diversity check; weight reduced to 5%                                                  |
| reward floor of 0.10 hid bad behavior                     | floor removed: a fully wrong action scores `0.0`                                                                     |
| `sft_data * 15` over 15 unique examples                   | SFT data sourced from the procedural train pool (no duplication, every example unique)                              |
| duplicate cell in the notebook                            | killed; replaced with a no-op placeholder for cell-numbering continuity                                              |
| `inference.py` defaulted to Qwen 72B via HF Router        | now defaults to local Qwen 2.5 3B Instruct + LoRA (the actual trained model) via `transformers` / `peft`             |
| reward weights with no derivation                         | new weights documented in `graders.py`; old weights logged for auditability                                          |

the v0.2 static fixtures (`fixtures/conflict_cases.json`) are kept *only* as the deployed UI demo. **all training and evaluation now run on the procedural pools.**

---

## what the environment actually does

multi-step conflict resolution loop with a mutable world state. each step the agent sees the current conflict (plus the visible calendar), emits one structured JSON action, and the env:

1. grades the action with the rebuilt grader.
2. mutates the calendar on `reschedule_event` / `propose_plan` actions.
3. on a clarification case where the agent correctly asks, **keeps the same conflict at the front of the queue** and reveals the hidden info on the next step.
4. on any cascade-rule case, checks whether the action triggers the rule (e.g. reschedule past 18:00 with `owner=work`) and **appends a follow-on conflict** to the queue.
5. returns reward, observation, done, and reward components.

action space stays the same (6 intents × 6 owners × 4 priorities + slot + needs_clarification + message), so the model contract is unchanged. existing clients keep working.

```json
{
  "intent": "reschedule_event",
  "owner": "work",
  "priority": "urgent",
  "proposed_slot": "20:30",
  "needs_clarification": false,
  "message_template": "Reschedule the incident review to 20:30 with owner confirmation."
}
```

---

## reward design (rebuilt)

six weighted components, summing to 1.0. each is in `[0, 1]`; a per-component contribution is naturally capped at its weight. full derivation lives in [`graders.py`](src/assistant_conflict_env/graders.py).

| component       | weight | what it checks                                                          |
|-----------------|--------|--------------------------------------------------------------------------|
| `intent`        | 0.40   | exact match on action category — most consequential decision           |
| `owner`         | 0.20   | exact match on responsible principal                                    |
| `slot`          | 0.20   | regex-strict 24h `HH:MM`; time-distance scoring (no substring shortcut) |
| `priority`      | 0.10   | partial credit (0.5) for off-by-one — adjacent priorities are defensible |
| `clarification` | 0.05   | boolean alignment with the case's `block_if_missing_context`             |
| `message`       | 0.05   | structural proxy: length + on-topic verb + word-diversity               |

**anti-hacking penalties** (subtracted from the score, no floor — meaning a bad action can hit `0.0`):

- `repetitive_intent` (−0.10): same intent three steps in a row (was two; the old check was bypassable by alternating two intents)
- `premature_finalize` (−0.15): `finalize_itinerary` while >1 conflict still pending
- `terminal_early` (−0.05): any other terminal-style intent used to short-circuit
- `missing_slot` (−0.05): `require_slot=True` but no slot supplied
- `clarification_spam` (−0.05): asking when not warranted (case has no clarification spec, or info already revealed)
- `short_message` (−0.04): message under 16 chars

reward band is now `[0.0, 1.0]`. real numbers, no floor protecting bad behavior.

---

## train / holdout / adversarial split

pools are disjoint *by construction* — asserted at module import via `assert_split_disjoint()`:

| pool          | seed range  | size  | what it's for                                       |
|---------------|-------------|-------|------------------------------------------------------|
| train         | 1000-1999   | 1000  | SFT data + GRPO rollouts                            |
| holdout       | 9000-9099   |  100  | honest generalization eval                          |
| adversarial   | 5000-5009   |   10  | probe that old shortcuts (substring slot, keyword stuffing, etc.) are dead |

every episode is deterministic from its seed. so the split is reproducible from any commit. no leakage. no fudging.

---

## training results

the notebook ([`notebooks/train_grpo_colab.ipynb`](notebooks/train_grpo_colab.ipynb)) was rebuilt to:

- pull SFT data from the procedural train pool (no `sft_data * 15` duplication).
- run GRPO with **the env's real reward** as the reward function (not a placeholder).
- evaluate on the **procedural holdout pool** (disjoint from training).
- include an adversarial probe cell that runs the rebuilt grader on canonical "old shortcut" inputs and asserts they no longer get full credit.

> **honesty note**: numbers below get filled in after rerunning the notebook on Colab T4. the pre-rebuild numbers (`1.0 / 1.0 / 1.0`) are *removed* because they came from a memorization regime that no longer exists. holdout averages are expected to be lower — that's the whole point.

| pool                | untrained 3B | after SFT | after SFT + GRPO |
|---------------------|--------------|-----------|------------------|
| holdout (n=100)     | TBD          | TBD       | TBD              |
| adversarial (n=10)  | TBD          | TBD       | TBD              |

to reproduce: open the Colab notebook, set runtime to T4, run all cells. the notebook prints holdout averages and the adversarial probe at the end.

---

## local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the env API (does NOT need transformers/peft)
uvicorn assistant_conflict_env.server:app --app-dir src --host 0.0.0.0 --port 7860

# Run the unit tests (14 tests; covers world-state, cascades, clarifications, adversarial probes)
pytest -q
```

to run inference locally with the trained Qwen 3B + LoRA adapter:

```bash
pip install -r requirements-inference.txt   # heavy deps; only for local model loading

export MODEL_PATH=Qwen/Qwen2.5-3B-Instruct
export LORA_PATH=./trained_conflict_resolver  # path produced by the notebook
export EVAL_POOL=holdout                      # or 'adversarial' / 'static'
export EVAL_LIMIT=20

python inference.py
```

`inference.py` priority order:
1. `MODEL_PATH` set → load local HF model + optional LoRA via `transformers` / `peft`.
2. `HF_TOKEN` set (and no `MODEL_PATH`) → call HF Router with `MODEL_NAME` (default Qwen 2.5 3B Instruct).
3. neither set → heuristic-only baseline so the script still does something visible.

---

## stack

| component       | tool                                  | why                                    |
|-----------------|---------------------------------------|----------------------------------------|
| base model      | Qwen 2.5 3B Instruct                  | fits a free Colab T4                   |
| quantization    | 4-bit (BnB via Unsloth)               | 16GB VRAM constraint                   |
| fine-tuning     | LoRA (r=16, ~1% params)               | efficient adaptation                   |
| SFT             | TRL `SFTTrainer`                      | teaches JSON output format             |
| RL              | TRL `GRPOTrainer` with env reward     | optimizes decision quality             |
| acceleration    | Unsloth                               | 2x faster training                     |
| environment     | OpenEnv (`reset` / `step` / `state`)  | standard RL interface                  |
| deployment      | Docker + FastAPI                      | HF Spaces compatible                   |
| local inference | `transformers` + `peft`               | loads trained 3B + LoRA on user GPU    |

---

## theme alignment (honestly)

- **theme 3.2 — personalized tasks**: the env models a single user's chaotic afternoon (work calendar, family logistics, finance/legal). personalization is in the templates and constraints.
- **theme 2 — long-horizon planning**: now a defensible claim. state is carried across steps via `WorldState`; cascade conflicts depend on the agent's own past actions; clarifications take a 2-step ask-then-act loop. the unit tests assert all of this — it's not just words on a slide anymore.

---

## project structure

```
round2/
├── README.md                           <- you are here
├── blog.md                             <- mini-blog for submission
├── Dockerfile
├── openenv.yaml
├── inference.py                        <- model evaluation (local LoRA / HF Router / heuristic)
├── requirements.txt                    <- env server deps (slim)
├── requirements-inference.txt          <- heavy deps for local inference only
├── src/
│   └── assistant_conflict_env/
│       ├── environment.py              <- queue-based env with WorldState
│       ├── conflict_generator.py       <- procedural template-based generator
│       ├── eval_set.py                 <- train / holdout / adversarial seed pools
│       ├── graders.py                  <- documented 6-component reward
│       ├── models.py                   <- pydantic data models incl. WorldState
│       ├── tasks.py                    <- static + procedural task resolver
│       ├── server.py                   <- FastAPI endpoints
│       └── fixtures/
│           └── conflict_cases.json     <- legacy static demo (UI only, not training)
├── notebooks/
│   └── train_grpo_colab.ipynb          <- procedural SFT + GRPO + holdout eval
├── docs/                               <- 9-chapter walkthrough
└── tests/
    ├── test_environment.py
    ├── test_graders.py
    └── test_world_state.py             <- new: cascades, clarifications, adversarial probes
```

---

## reproducing demo and eval

for the demo video, the recommended seed is **42** (used in the new test suite):

```bash
python -c "
import asyncio
from src.assistant_conflict_env.environment import PersonalAssistantConflictEnv
async def go():
    env = PersonalAssistantConflictEnv()
    r = await env.reset(task_name='proc_hard_42')
    print(r.observation.current_conflict.summary)
asyncio.run(go())
"
```

---

*built with Unsloth, TRL, and a willingness to delete our own dishonest numbers.*
