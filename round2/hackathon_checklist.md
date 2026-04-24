# Round 2 Hackathon Checklist

This checklist maps your provided requirements to concrete actions in this repo.

## Mandatory requirements

- [x] Use OpenEnv environment pattern
  - Implemented in `src/assistant_conflict_env/*`
  - Contract: `reset`, `step`, `state`

- [x] Show minimal training script with TRL/Unsloth
  - Script: `scripts/train_grpo_stub.py`
  - Real rollout loop included
  - Optional TRL: `--train-with-trl`
  - Optional Unsloth: `--use-unsloth`

- [ ] Host OpenEnv compliant environment on Hugging Face Spaces
  - Ensure Docker Space is linked to this Round 2 folder repo
  - Validate `/health`, `/reset`, `/step`, `/state`

- [ ] Publish mini-blog (<2 min read) or mini-video (<2 min)
  - Recommended sections:
    1. Problem statement + theme mapping
    2. Reward design and anti-hacking checks
    3. Before vs after metrics

## Runbook

1. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Validate environment

```bash
PYTHONPATH=$PYTHONPATH:. pytest -q .
openenv validate
python inference.py
```

3. Training deps (Colab/Linux GPU recommended)

```bash
pip install -r requirements-train.txt
```

4. Collect rollouts

```bash
python scripts/train_grpo_stub.py --episodes 24 --max-prompt-rows 160
```

5. Start GRPO training

```bash
python scripts/train_grpo_stub.py --episodes 24 --max-prompt-rows 160 --train-with-trl
```

6. Optional Unsloth acceleration

```bash
python scripts/train_grpo_stub.py --episodes 24 --max-prompt-rows 160 --train-with-trl --use-unsloth
```

## Demo evidence to capture

- Baseline inference score by task
- Trained model score by task
- Reward component trends
- At least one example trajectory showing improvement
