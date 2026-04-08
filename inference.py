from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
import warnings
from pathlib import Path
from typing import List, Optional

# Keep stdout restricted to structured lines for evaluator parsing.
warnings.filterwarnings("ignore", category=UserWarning, module="openai")

from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from email_triage_env.environment import EmailTriageEnv
from email_triage_env.models import Category, EmailTriageAction, Priority, Team


API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
TASK_NAMES = [
    task.strip()
    for task in os.getenv(
        "TASK_NAMES",
        "easy_priority_routing,medium_mixed_inbox,hard_ambiguous_escalations",
    ).split(",")
    if task.strip()
]
MAX_STEPS = int(os.getenv("MAX_STEPS", "12"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "220"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.70"))
BENCHMARK = os.getenv("BENCHMARK", "email_triage")

ALLOWED_CATEGORIES = [item.value for item in Category]
ALLOWED_PRIORITIES = [item.value for item in Priority]
ALLOWED_TEAMS = [item.value for item in Team]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an email triage agent.
    Respond with exactly one JSON object using these keys:
    category, priority, team, mark_spam, response_template.

    Allowed category values: billing, bug, account, sales, abuse, legal, other
    Allowed priority values: low, normal, high, urgent
    Allowed team values: support, billing, engineering, sales, trust_safety, legal

    Keep response_template short but concrete. Mention domain-specific intent.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = "null" if not error else str(error).replace("\n", " ")
    done_value = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_value} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    reward_str = ",".join(f"{item:.2f}" for item in rewards)
    success_value = str(success).lower()
    print(
        f"[END] success={success_value} steps={steps} score={score:.2f} rewards={reward_str}",
        flush=True,
    )


def normalize_choice(raw: str, allowed: List[str], fallback: str) -> str:
    candidate = str(raw).strip().lower()
    return candidate if candidate in allowed else fallback


def heuristic_action(subject: str, body: str) -> EmailTriageAction:
    text = f"{subject} {body}".lower()

    category = "other"
    priority = "normal"
    team = "support"
    mark_spam = False

    if any(token in text for token in ["outage", "incident", "error", "bug", "failed"]):
        category = "bug"
        team = "engineering"
        priority = "urgent"
    elif any(token in text for token in ["invoice", "refund", "charged", "payment", "billing"]):
        category = "billing"
        team = "billing"
        priority = "high"
    elif any(token in text for token in ["password", "login", "account", "access", "reset"]):
        category = "account"
        team = "support"
        priority = "high"
    elif any(token in text for token in ["pricing", "enterprise", "quote", "contract", "procurement"]):
        category = "sales"
        team = "sales"
        priority = "normal"
    elif any(token in text for token in ["gdpr", "court", "legal", "preservation", "notice"]):
        category = "legal"
        team = "legal"
        priority = "urgent"
    elif any(token in text for token in ["phishing", "abuse", "spam", "hate", "suspension", "verify account"]):
        category = "abuse"
        team = "trust_safety"
        priority = "high"
        mark_spam = True

    response = "Routing to the appropriate team with next-step confirmation."
    return EmailTriageAction(
        category=category,
        priority=priority,
        team=team,
        mark_spam=mark_spam,
        response_template=response,
    )


def parse_model_json(raw_text: str) -> Optional[dict]:
    candidate = raw_text.strip()
    if not candidate:
        return None

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    left = candidate.find("{")
    right = candidate.rfind("}")
    if left == -1 or right == -1 or right <= left:
        return None

    chunk = candidate[left : right + 1]
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        return None


def request_model_action(client: Optional[OpenAI], subject: str, body: str, history: List[str]) -> EmailTriageAction:
    fallback = heuristic_action(subject, body)
    if client is None:
        return fallback

    history_block = "\n".join(history[-4:]) if history else "None"
    user_prompt = textwrap.dedent(
        f"""
        Current email subject: {subject}
        Current email body: {body}
        Recent routing history:
        {history_block}

        Return a JSON object only.
        """
    ).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        content = completion.choices[0].message.content or ""
        payload = parse_model_json(content)
        if payload is None:
            return fallback

        return EmailTriageAction(
            category=normalize_choice(payload.get("category", ""), ALLOWED_CATEGORIES, fallback.category.value),
            priority=normalize_choice(payload.get("priority", ""), ALLOWED_PRIORITIES, fallback.priority.value),
            team=normalize_choice(payload.get("team", ""), ALLOWED_TEAMS, fallback.team.value),
            mark_spam=bool(payload.get("mark_spam", fallback.mark_spam)),
            response_template=str(payload.get("response_template", fallback.response_template))[:400],
        )
    except Exception:
        return fallback


def action_to_string(action: EmailTriageAction) -> str:
    return (
        "route("
        f"category={action.category.value},"
        f"priority={action.priority.value},"
        f"team={action.team.value},"
        f"mark_spam={str(action.mark_spam).lower()}"
        ")"
    )


async def run_task(env: EmailTriageEnv, client: Optional[OpenAI], task_name: str) -> None:
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_name=task_name)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            if result.observation.current_email is None:
                break

            current = result.observation.current_email
            action = request_model_action(
                client=client,
                subject=current.subject,
                body=current.body,
                history=history,
            )

            result = await env.step(action)
            reward = float(result.reward or 0.0)
            done = bool(result.done)
            error = result.info.get("last_action_error")

            rewards.append(reward)
            steps_taken = step
            history.append(f"{current.email_id}:{action.category.value}/{action.team.value}/{action.priority.value}")

            log_step(
                step=step,
                action=action_to_string(action),
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        state = await env.state()
        score = float(state.final_score or state.average_step_score)
        if score < 0.0:
            score = 0.0
        elif score > 1.0:
            score = 1.0
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception:
        success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None

    env = await EmailTriageEnv.from_docker_image(LOCAL_IMAGE_NAME)
    try:
        for task_name in TASK_NAMES:
            await run_task(env=env, client=client, task_name=task_name)
    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(main())
