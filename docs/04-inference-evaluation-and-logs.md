# 04 - Inference, Evaluation, and Log Format

This chapter explains why `inference.py` is mandatory and how scoring works.

## 1. Why inference.py Is Mandatory

The hackathon uses this script to verify that your environment can actually be used by an agent.

If no model can interact and get reward, environment quality is considered weak.

## 2. Required Environment Variables

Set these before running inference:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="<your_token>"
```

Optional:

```bash
export LOCAL_IMAGE_NAME="<only_if_using_from_docker_image>"
```

## 3. Required Log Format

The evaluator expects this structure exactly:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

Rules:

- one START per episode
- one STEP per `env.step()`
- one END always, even on exception
- rewards with 2 decimals
- booleans in lowercase

## 4. Why We Keep a Heuristic Fallback

If model output is malformed or API fails:

- episode should not crash
- script still emits valid structured logs
- scoring remains reproducible

So a fallback action strategy keeps the pipeline robust.

## 5. Score Interpretation in This Project

Two levels:

1. Step reward

- dense signal with components + penalties

2. Final score

- deterministic task-grade normalized to [0, 1]

`success` in END is based on a threshold (`SUCCESS_SCORE_THRESHOLD`).

## 6. Reproducibility Tips

- keep temperature low (`0.0` or near)
- clamp all score values to [0, 1]
- avoid random task generation
- keep fixtures static

## 7. Quick Verification Command

```bash
python inference.py | head -n 25
```

You should see START/STEP/END lines immediately.
