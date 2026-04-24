from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
import warnings
from pathlib import Path
from typing import List, Optional

warnings.filterwarnings("ignore", category=UserWarning, module="openai")

from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from assistant_conflict_env.environment import PersonalAssistantConflictEnv
from assistant_conflict_env.models import ActionIntent, ConflictAction, Owner, Priority


API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
TASK_NAMES = [
    task.strip()
    for task in os.getenv(
        "TASK_NAMES",
        "easy_evening_planner,medium_multi_party_negotiation,hard_cascade_replanning",
    ).split(",")
    if task.strip()
]
MAX_STEPS = int(os.getenv("MAX_STEPS", "14"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "260"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.72"))
BENCHMARK = os.getenv("BENCHMARK", "personal_assistant_conflict_resolution")

ALLOWED_INTENTS = [item.value for item in ActionIntent]
ALLOWED_OWNERS = [item.value for item in Owner]
ALLOWED_PRIORITIES = [item.value for item in Priority]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a personal assistant conflict resolver.
    Return exactly one JSON object with keys:
    intent, owner, priority, proposed_slot, needs_clarification, message_template

    Allowed intent values:
    route_message, propose_plan, reschedule_event, delegate_task, ask_clarification, finalize_itinerary

    Allowed owner values:
    self, work, family, travel, finance, legal

    Allowed priority values:
    low, normal, high, urgent

    Keep message_template concise, actionable, and constraint-aware.
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


def heuristic_action(summary: str, constraints: List[str], history: List[str]) -> ConflictAction:
    text = f"{summary} {' '.join(constraints)}".lower()

    intent = "route_message"
    owner = "self"
    priority = "normal"
    proposed_slot = ""
    needs_clarification = False

    if any(token in text for token in ["missing", "unclear", "timezone", "attachment", "ambiguous"]):
        intent = "ask_clarification"
        owner = "work" if "client" in text or "demo" in text else "legal"
        priority = "high"
        needs_clarification = True
    elif any(token in text for token in ["overlap", "delay", "commute", "check-in", "reservation"]):
        intent = "reschedule_event"
        owner = "travel" if any(tok in text for tok in ["flight", "hotel", "commute", "driver"]) else "work"
        priority = "high"
        proposed_slot = "after 20:30"
    elif any(token in text for token in ["pickup", "gift", "cancel window", "cancel"]):
        intent = "delegate_task"
        owner = "family" if any(tok in text for tok in ["gift", "school", "pickup"]) else "travel"
        priority = "normal"
    elif any(token in text for token in ["payment", "renewal", "insurance", "invoice", "fee"]):
        intent = "route_message"
        owner = "finance"
        priority = "urgent"
    elif any(token in text for token in ["final", "consolidated", "itinerary", "summary"]):
        intent = "finalize_itinerary"
        owner = "self"
        priority = "high"

    response = "Action chosen with constraints and fallback ownership in mind."
    if intent == "ask_clarification":
        response = "Need clarification before execution to avoid invalid scheduling risk."
    elif intent == "finalize_itinerary":
        response = "Timeline confirmed with owners, risk notes, and fallback path."

    if history and history[-1].startswith("finalize_itinerary") and intent == "finalize_itinerary":
        intent = "propose_plan"

    return ConflictAction(
        intent=intent,
        owner=owner,
        priority=priority,
        proposed_slot=proposed_slot,
        needs_clarification=needs_clarification,
        message_template=response,
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


def request_model_action(
    client: Optional[OpenAI],
    summary: str,
    constraints: List[str],
    history: List[str],
) -> ConflictAction:
    fallback = heuristic_action(summary, constraints, history)
    if client is None:
        return fallback

    history_block = "\n".join(history[-4:]) if history else "None"
    constraints_block = "\n".join(f"- {item}" for item in constraints)
    user_prompt = textwrap.dedent(
        f"""
        Conflict summary: {summary}
        Constraints:
        {constraints_block if constraints_block else '- none'}

        Recent action history:
        {history_block}

        Return one JSON object only.
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

        return ConflictAction(
            intent=normalize_choice(payload.get("intent", ""), ALLOWED_INTENTS, fallback.intent.value),
            owner=normalize_choice(payload.get("owner", ""), ALLOWED_OWNERS, fallback.owner.value),
            priority=normalize_choice(payload.get("priority", ""), ALLOWED_PRIORITIES, fallback.priority.value),
            proposed_slot=str(payload.get("proposed_slot", fallback.proposed_slot))[:80],
            needs_clarification=bool(payload.get("needs_clarification", fallback.needs_clarification)),
            message_template=str(payload.get("message_template", fallback.message_template))[:500],
        )
    except Exception:
        return fallback


def action_to_string(action: ConflictAction) -> str:
    return (
        "act("
        f"intent={action.intent.value},"
        f"owner={action.owner.value},"
        f"priority={action.priority.value},"
        f"clarify={str(action.needs_clarification).lower()}"
        ")"
    )


async def run_task(env: PersonalAssistantConflictEnv, client: Optional[OpenAI], task_name: str) -> None:
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

            if result.observation.current_conflict is None:
                break

            current = result.observation.current_conflict
            action = request_model_action(
                client=client,
                summary=current.summary,
                constraints=current.constraints,
                history=history,
            )

            result = await env.step(action)
            reward = float(result.reward or 0.0)
            done = bool(result.done)
            error = result.info.get("last_action_error")

            rewards.append(reward)
            steps_taken = step
            history.append(f"{action.intent.value}/{action.owner.value}/{action.priority.value}")

            log_step(step=step, action=action_to_string(action), reward=reward, done=done, error=error)

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

    env = await PersonalAssistantConflictEnv.from_docker_image(LOCAL_IMAGE_NAME)
    try:
        for task_name in TASK_NAMES:
            await run_task(env=env, client=client, task_name=task_name)
    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(main())
