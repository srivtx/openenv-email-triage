---
title: Personal Assistant Conflict Resolver v2
emoji: '🗓️'
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Personal Assistant Conflict Resolver v2

**an OpenEnv RL environment that teaches LLMs to resolve cascading scheduling conflicts — because real life doesn't come one email at a time.**

*OpenEnv Hackathon 2026 | Team Agent (1)*

---

## the problem

every AI assistant can set a timer or read the weather. but hand it a real afternoon — board review overlapping school pickup, visa deadline with missing docs, insurance payment failing, hotel cancellation window closing — and it breaks. they handle one thing at a time. they don't plan.

we built an RL environment that forces models to plan across multiple competing priorities, handle missing information, and make decisions that cascade.

---

## what we built

a multi-step conflict resolution environment with **15 realistic conflicts** across 3 difficulty levels:

| difficulty | conflicts | what makes it hard |
|---|---|---|
| **easy** | 3 | simple overlaps — dinner vs review, medication vs commute |
| **medium** | 5 | missing info (timezones, attachments), multi-party coordination |
| **hard** | 7 | cascading chaos — every decision affects downstream conflicts |

for each conflict, the model outputs a structured JSON decision:

```json
{
  "intent": "reschedule_event",
  "owner": "work",
  "priority": "urgent",
  "proposed_slot": "after 20:30",
  "needs_clarification": false,
  "message_template": "Reschedule board review to protect school pickup."
}
```

6 possible intents × 6 owners × 4 priorities = hundreds of combinations per conflict.

---

## reward design

**6 weighted scoring components** — not pass/fail, but dense per-step feedback:

| component | weight | what it checks |
|---|---|---|
| intent correctness | 34% | right action type? |
| owner correctness | 20% | right responsible party? |
| priority accuracy | 15% | right urgency? (partial credit for close) |
| slot compliance | 14% | valid time slot when needed? |
| clarification behavior | 10% | asked for info only when missing? |
| message quality | 7% | relevant keywords present? |

**5 anti-gaming penalties** to prevent reward hacking:
- repetitive intent spam (-0.05)
- premature finalization (-0.08)
- clarification spam (-0.03)
- missing slot when rescheduling (-0.05)
- lazy one-word messages (-0.04)

---

## training results

**two-stage pipeline: SFT then GRPO** (same recipe as ChatGPT/InstructGPT)

| model | easy | medium | hard | average |
|---|---|---|---|---|
| untrained 3B | 0.5613 | 0.6346 | 0.4741 | **0.5567** |
| GRPO only (failed) | 0.5247 | 0.4704 | 0.4514 | **0.4822** |
| after SFT | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| after SFT + GRPO | 1.0000 | 1.0000 | 1.0000 | **1.0000** |

**+80% improvement** from untrained to trained.

> GRPO alone actually *worsened* performance — classic reward hacking. the model gamed the training signal but couldn't produce valid JSON. SFT teaches format first, GRPO optimizes decisions after. same pattern OpenAI used for InstructGPT.

---

## links

| resource | link |
|---|---|
| **live environment** | [HuggingFace Space](https://huggingface.co/spaces/srivtx/openenv-conflict-resolver-v2) |
| **mini-blog** | [blog.md](./blog.md) |
| **training notebook** | [Colab Notebook](./notebooks/train_grpo_colab.ipynb) |
| **full documentation** | [docs/](./docs/) (9 chapters, from python basics to full architecture) |

---

## theme alignment

- **theme 3.2 — personalized tasks**: real personal assistant conflict handling
- **theme 2 — long-horizon planning**: multi-step cascading decisions across 3-12 steps

---

## stack

| component | tool | why |
|---|---|---|
| base model | Qwen 2.5 3B Instruct | fits on free Colab T4 |
| quantization | 4-bit (BnB via Unsloth) | 16GB VRAM constraint |
| fine-tuning | LoRA (r=16, 0.96% params) | efficient adaptation |
| SFT | TRL SFTTrainer | teaches correct JSON format |
| RL | TRL GRPOTrainer | optimizes decision quality |
| acceleration | Unsloth | 2x faster training |
| environment | OpenEnv (reset/step/state) | standard RL interface |
| deployment | Docker + FastAPI | HF Spaces compatible |

---

## project structure

```
round2/
├── README.md                  <- you are here
├── blog.md                    <- mini-blog for submission
├── Dockerfile
├── openenv.yaml
├── server.py                  <- FastAPI endpoints
├── inference.py               <- model evaluation
├── src/
│   └── assistant_conflict_env/
│       ├── environment.py     <- RL environment (reset/step/state)
│       ├── models.py          <- pydantic data models
│       ├── graders.py         <- 6-component reward scoring
│       ├── tasks.py           <- task loader
│       └── fixtures/
│           └── conflict_cases.json
├── notebooks/
│   └── train_grpo_colab.ipynb <- SFT + GRPO training
├── docs/                      <- 9-chapter documentation
│   ├── 01_foundations.md
│   ├── 02_what_is_rl.md
│   ├── 03_building_the_environment.md
│   ├── 04_reward_engineering.md
│   ├── 05_training_pipeline.md
│   ├── 05b_why_sft_before_grpo.md
│   ├── 06_inference_and_logging.md
│   ├── 07_deployment.md
│   └── 08_full_architecture.md
└── tests/
    ├── test_environment.py
    └── test_graders.py
```

---

## local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn assistant_conflict_env.server:app --app-dir src --host 0.0.0.0 --port 7860
```

```bash
pytest -q
openenv validate
python inference.py
```

---

## documentation

full walkthrough available in [docs/](./docs/) — 9 chapters covering everything from python basics to the complete architecture, including why we needed SFT before GRPO (the hard way).

---

*built with Unsloth, TRL, and way too much caffeine*
