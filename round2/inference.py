"""Inference / evaluation harness for the Personal Assistant Conflict env.

Three model-selection paths, in priority order:

1. ``MODEL_PATH``: load a local HF causal LM and (optionally) attach a LoRA
   adapter from ``LORA_PATH``. This is the path the trained Qwen 2.5 3B
   Instruct + LoRA adapter takes. **This is the default we point users to in
   the README.** Heavy deps (``transformers``, ``peft``, ``torch``) are
   imported lazily only on this path.

2. ``HF_TOKEN`` (and ``MODEL_NAME``): call HuggingFace Router-served chat
   completions API. Useful for running a baseline 72B comparison without
   downloading weights, but ``MODEL_NAME`` defaults to a *3B* model so the
   default reported numbers match what we trained.

3. Heuristic fallback: a small if-else classifier mirroring the model's job.
   Used when neither of the above is configured.

Evaluation pool selection:

- ``EVAL_POOL=static``  (default): runs the legacy static demo tasks, useful
  for the live UI demo on the HF Space.
- ``EVAL_POOL=holdout``: runs procedurally generated holdout episodes
  (disjoint from training seeds), reporting honest generalization numbers.
- ``EVAL_POOL=adversarial``: runs the small adversarial probe pool to verify
  shortcut shutdowns.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
import warnings
from pathlib import Path
from typing import List, Optional, Protocol

warnings.filterwarnings("ignore", category=UserWarning, module="openai")


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from assistant_conflict_env.environment import PersonalAssistantConflictEnv
from assistant_conflict_env.eval_set import (
    adversarial_episodes,
    holdout_episodes,
)
from assistant_conflict_env.models import ActionIntent, ConflictAction, Owner, Priority
from assistant_conflict_env.tasks import list_tasks


API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
# Default name now matches the trained model size; override with env var to use 72B.
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-3B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_PATH = os.getenv("MODEL_PATH")  # local HF model id or path; if set, use transformers
LORA_PATH = os.getenv("LORA_PATH")  # optional LoRA adapter path
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
EVAL_POOL = os.getenv("EVAL_POOL", "static").lower()
EVAL_DIFFICULTY = os.getenv("EVAL_DIFFICULTY", "hard")
EVAL_LIMIT = int(os.getenv("EVAL_LIMIT", "0"))
TASK_NAMES_RAW = os.getenv("TASK_NAMES", "")
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
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

    Use 24h HH:MM format inside proposed_slot whenever a time is required.
    Keep message_template concise, actionable, and constraint-aware.
    """
).strip()


# ---------------------------------------------------------------------------
# Model client abstractions
# ---------------------------------------------------------------------------


class ModelClient(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class HeuristicClient:
    name = "heuristic"

    def complete(self, system: str, user: str) -> str:
        # Heuristic doesn't use the prompt; the caller falls back to its own
        # heuristic when this client returns an empty payload.
        return ""


class HFRouterClient:
    """OpenAI-compatible HuggingFace Router chat completion client."""

    def __init__(self, base_url: str, token: str, model_name: str) -> None:
        from openai import OpenAI  # local import to avoid mandatory dep at module load

        self._client = OpenAI(base_url=base_url, api_key=token)
        self._model_name = model_name
        self.name = f"hf-router:{model_name}"

    def complete(self, system: str, user: str) -> str:
        completion = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return completion.choices[0].message.content or ""


class LocalHFClient:
    """Local HF causal LM client with optional LoRA adapter.

    Loads ``MODEL_PATH`` (e.g. ``Qwen/Qwen2.5-3B-Instruct``) and, if
    ``LORA_PATH`` is set, attaches the adapter via PEFT. Inference uses the
    chat template baked into the tokenizer.
    """

    def __init__(self, model_path: str, lora_path: Optional[str] = None) -> None:
        # Heavy imports happen lazily so the module can still be imported in
        # environments that lack torch/transformers (e.g. the deployed env
        # server image).
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        device_map = "auto" if torch.cuda.is_available() else None
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        if lora_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, lora_path)

        self._model.eval()
        self.name = f"local:{model_path}" + (f"+lora:{lora_path}" if lora_path else "")

    def complete(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with self._torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                do_sample=TEMPERATURE > 0.0,
                temperature=max(TEMPERATURE, 1e-5),
                pad_token_id=self._tokenizer.pad_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)


def build_client() -> ModelClient:
    if MODEL_PATH:
        try:
            return LocalHFClient(model_path=MODEL_PATH, lora_path=LORA_PATH)
        except Exception as exc:
            print(f"[WARN] Failed to load local model from {MODEL_PATH}: {exc}", flush=True)
    if HF_TOKEN:
        try:
            return HFRouterClient(base_url=API_BASE_URL, token=HF_TOKEN, model_name=MODEL_NAME)
        except Exception as exc:
            print(f"[WARN] Failed to init HF Router client: {exc}", flush=True)
    return HeuristicClient()


# ---------------------------------------------------------------------------
# Heuristic fallback (kept for the bottom of the model-selection priority list)
# ---------------------------------------------------------------------------


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
        proposed_slot = "20:30"
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

    response = "Action chosen with constraints, owners, and fallback path documented for follow up."
    if intent == "ask_clarification":
        response = "Need clarification on the missing detail before scheduling to avoid an invalid commitment."
    elif intent == "finalize_itinerary":
        response = "Timeline confirmed across owners with risk notes and a fallback path for late changes."

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


# ---------------------------------------------------------------------------
# JSON parsing and action construction
# ---------------------------------------------------------------------------


def parse_model_json(raw_text: str) -> Optional[dict]:
    candidate = (raw_text or "").strip()
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


def normalize_choice(raw: str, allowed: List[str], fallback: str) -> str:
    candidate = str(raw).strip().lower()
    return candidate if candidate in allowed else fallback


def request_model_action(
    client: ModelClient,
    summary: str,
    constraints: List[str],
    history: List[str],
) -> ConflictAction:
    fallback = heuristic_action(summary, constraints, history)
    if isinstance(client, HeuristicClient):
        return fallback

    history_block = "\n".join(history[-4:]) if history else "None"
    constraints_block = "\n".join(f"- {item}" for item in constraints) or "- none"
    user_prompt = textwrap.dedent(
        f"""
        Conflict summary: {summary}
        Constraints:
        {constraints_block}

        Recent action history:
        {history_block}

        Return one JSON object only.
        """
    ).strip()

    try:
        raw = client.complete(SYSTEM_PROMPT, user_prompt)
    except Exception as exc:
        print(f"[WARN] model call failed: {exc}", flush=True)
        return fallback

    payload = parse_model_json(raw)
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


# ---------------------------------------------------------------------------
# Logging + episode loop
# ---------------------------------------------------------------------------


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


def action_to_string(action: ConflictAction) -> str:
    return (
        "act("
        f"intent={action.intent.value},"
        f"owner={action.owner.value},"
        f"priority={action.priority.value},"
        f"clarify={str(action.needs_clarification).lower()}"
        ")"
    )


async def run_task(env: PersonalAssistantConflictEnv, client: ModelClient, task_name: str) -> float:
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=client.name)

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
        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        print(f"[ERROR] task={task_name} exception={exc}", flush=True)
        success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


def resolve_task_names() -> List[str]:
    if TASK_NAMES_RAW.strip():
        return [t.strip() for t in TASK_NAMES_RAW.split(",") if t.strip()]
    if EVAL_POOL == "static":
        return [task.id for task in list_tasks()]
    if EVAL_POOL == "holdout":
        episodes = holdout_episodes(EVAL_DIFFICULTY, limit=EVAL_LIMIT or 20)
        return [task.id for task in episodes]
    if EVAL_POOL == "adversarial":
        return [task.id for task in adversarial_episodes(EVAL_DIFFICULTY)]
    raise ValueError(f"Unknown EVAL_POOL '{EVAL_POOL}'.")


async def main() -> None:
    client = build_client()
    task_names = resolve_task_names()

    env = await PersonalAssistantConflictEnv.from_docker_image(LOCAL_IMAGE_NAME)
    scores: List[float] = []
    try:
        for task_name in task_names:
            scores.append(await run_task(env=env, client=client, task_name=task_name))
    finally:
        await env.close()

    if scores:
        avg = sum(scores) / len(scores)
        print(
            f"[SUMMARY] pool={EVAL_POOL} count={len(scores)} mean_score={avg:.3f} "
            f"min={min(scores):.3f} max={max(scores):.3f} success_rate="
            f"{sum(1 for s in scores if s >= SUCCESS_SCORE_THRESHOLD) / len(scores):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
