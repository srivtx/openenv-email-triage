# OpenEnv Email Triage - Beginner Documentation

This folder is a full beginner-friendly guide for what we built, why we built it, and how to rebuild it from scratch.

## What You Built

You built a real-world RL environment called **Email Triage OpenEnv**.
An agent reads incoming email and must decide:

- category
- priority
- team routing
- spam/abuse flag

The environment returns step rewards, final scores, and deterministic task grading.

## Who This Guide Is For

Use this docs folder if you are:

- new to Reinforcement Learning
- new to OpenEnv
- unsure why each file exists
- trying to understand the exact hackathon submission flow

## Learning Path (Read In Order)

1. [01-rl-from-scratch.md](01-rl-from-scratch.md)
2. [02-openenv-from-scratch.md](02-openenv-from-scratch.md)
3. [03-build-project-step-by-step.md](03-build-project-step-by-step.md)
4. [04-inference-evaluation-and-logs.md](04-inference-evaluation-and-logs.md)
5. [05-deploy-and-submit.md](05-deploy-and-submit.md)
6. [06-issues-we-faced-and-how-we-fixed-them.md](06-issues-we-faced-and-how-we-fixed-them.md)
7. [07-next-steps-from-beginner-to-pro.md](07-next-steps-from-beginner-to-pro.md)

## Fast Reality Check

If you only have 10 minutes before a deadline, run this sequence:

```bash
python -m pytest -q
openenv validate
python inference.py
```

Then confirm deployment endpoint health:

```bash
curl -s https://srivtx-openenv-email-triage.hf.space/health
```

## Project Architecture at a Glance

- `src/email_triage_env/` - core environment logic
- `server/app.py` - OpenEnv-compatible server entry point
- `openenv.yaml` - environment manifest
- `inference.py` - required baseline script
- `Dockerfile` - container deployment config
- `tests/` - deterministic checks

## Final Advice

Do not optimize too early. Build in this order:

1. Make reset/step/state work
2. Make rewards meaningful
3. Make grading deterministic
4. Make inference reproducible
5. Make deployment boring and reliable
