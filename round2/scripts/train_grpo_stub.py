"""Round 2 real rollout loop + minimal TRL GRPO entrypoint.

This script matches the hackathon recommendation:
Environment -> verifier/reward -> TRL (GRPO) -> optional Unsloth acceleration.

It does two concrete things:
1) Collects real environment rollouts and saves a prompt dataset.
2) Optionally launches GRPO with an online verifier reward function.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.assistant_conflict_env.environment import PersonalAssistantConflictEnv
from src.assistant_conflict_env.models import ConflictAction


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    output_dir: str = "./outputs/grpo-round2"
    rollouts_jsonl: str = "./outputs/grpo-round2/rollouts.jsonl"
    episodes: int = 24
    max_prompt_rows: int = 160
    max_steps: int = 12
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 1
    logging_steps: int = 5
    max_prompt_length: int = 1024
    max_completion_length: int = 256
    train_with_trl: bool = False
    use_unsloth: bool = False
    seed: int = 42


TASK_IDS = [
    "easy_evening_planner",
    "medium_multi_party_negotiation",
    "hard_cascade_replanning",
]


def make_prompt(observation: Dict[str, Any], history_actions: List[Dict[str, Any]]) -> str:
    current = observation.get("current_conflict") or {}
    constraints = current.get("constraints", [])
    constraints_block = "\n".join(f"- {item}" for item in constraints)
    history_block = "\n".join(
        f"- intent={item.get('intent')} owner={item.get('owner')} priority={item.get('priority')}"
        for item in history_actions[-3:]
    )
    if not history_block:
        history_block = "- none"

    return (
        "You are a personal assistant conflict resolver. Return one JSON object only with keys:\n"
        "intent, owner, priority, proposed_slot, needs_clarification, message_template\n\n"
        f"Task: {observation.get('task_id', '')} ({observation.get('task_difficulty', '')})\n"
        f"Conflict: {current.get('summary', '')}\n"
        f"Constraints:\n{constraints_block if constraints_block else '- none'}\n"
        f"Recent history:\n{history_block}\n"
        f"Feedback hint: {observation.get('last_feedback', '')}\n"
    )


def parse_action_json(raw_text: str) -> Optional[Dict[str, Any]]:
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

    try:
        return json.loads(candidate[left : right + 1])
    except json.JSONDecodeError:
        return None


def normalize_action_dict(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = payload or {}
    return {
        "intent": str(payload.get("intent", "route_message")).strip().lower() or "route_message",
        "owner": str(payload.get("owner", "self")).strip().lower() or "self",
        "priority": str(payload.get("priority", "normal")).strip().lower() or "normal",
        "proposed_slot": str(payload.get("proposed_slot", ""))[:80],
        "needs_clarification": bool(payload.get("needs_clarification", False)),
        "message_template": str(payload.get("message_template", "Action with constraints considered."))[:500],
    }


def heuristic_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    current = observation.get("current_conflict") or {}
    text = f"{current.get('summary', '')} {' '.join(current.get('constraints', []))}".lower()

    action = {
        "intent": "route_message",
        "owner": "self",
        "priority": "normal",
        "proposed_slot": "",
        "needs_clarification": False,
        "message_template": "Routing action chosen with fallback ownership.",
    }

    if any(token in text for token in ["missing", "unclear", "timezone", "attachment"]):
        action.update(
            {
                "intent": "ask_clarification",
                "owner": "work",
                "priority": "high",
                "needs_clarification": True,
                "message_template": "Need clarification on missing constraints before safe execution.",
            }
        )
    elif any(token in text for token in ["overlap", "delay", "reservation", "check-in", "driver"]):
        action.update(
            {
                "intent": "reschedule_event",
                "owner": "travel",
                "priority": "high",
                "proposed_slot": "after 20:30",
                "message_template": "Reschedule event and protect hard constraints first.",
            }
        )
    elif any(token in text for token in ["pickup", "gift", "cancel window"]):
        action.update(
            {
                "intent": "delegate_task",
                "owner": "family",
                "priority": "normal",
                "message_template": "Delegate actionable task with clear completion confirmation.",
            }
        )
    elif any(token in text for token in ["itinerary", "consolidated", "final"]):
        action.update(
            {
                "intent": "finalize_itinerary",
                "owner": "self",
                "priority": "high",
                "message_template": "Finalize timeline with fallback plan and risks.",
            }
        )

    return action


async def rollout_episode(
    env: PersonalAssistantConflictEnv,
    task_id: str,
    max_steps: int,
    max_rows_left: int,
) -> Dict[str, Any]:
    result = await env.reset(task_name=task_id)
    history_actions: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    total_reward = 0.0
    steps = 0

    while not result.done and steps < max_steps and len(rows) < max_rows_left:
        obs_dict = result.observation.model_dump(mode="json")
        prompt = make_prompt(obs_dict, history_actions)

        rows.append(
            {
                "prompt": prompt,
                "task_name": task_id,
                "history_actions": [dict(item) for item in history_actions],
                "step_index": obs_dict.get("step_index", 0),
            }
        )

        action_dict = heuristic_action(obs_dict)
        action = ConflictAction.model_validate(action_dict)
        result = await env.step(action)

        history_actions.append(action.model_dump(mode="json"))
        total_reward += float(result.reward or 0.0)
        steps += 1

    state = await env.state()
    return {
        "rows": rows,
        "episode_summary": {
            "task_id": task_id,
            "steps": steps,
            "cumulative_reward": round(total_reward, 4),
            "final_score": round(float(state.final_score or 0.0), 4),
        },
    }


async def collect_prompt_rows(cfg: TrainConfig) -> Dict[str, Any]:
    env = PersonalAssistantConflictEnv()
    rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for episode in range(cfg.episodes):
        task_id = TASK_IDS[episode % len(TASK_IDS)]
        pack = await rollout_episode(
            env=env,
            task_id=task_id,
            max_steps=cfg.max_steps,
            max_rows_left=max(cfg.max_prompt_rows - len(rows), 0),
        )
        rows.extend(pack["rows"])
        summaries.append(pack["episode_summary"])
        if len(rows) >= cfg.max_prompt_rows:
            break

    await env.close()
    return {"rows": rows, "summaries": summaries}


async def reward_from_completion(task_name: str, history_actions: List[Dict[str, Any]], completion_text: str) -> float:
    payload = parse_action_json(completion_text)
    if payload is None:
        return 0.0

    env = PersonalAssistantConflictEnv()
    result = await env.reset(task_name=task_name)

    for step_action in history_actions:
        if result.done:
            await env.close()
            return 0.0
        try:
            action = ConflictAction.model_validate(step_action)
        except Exception:
            await env.close()
            return 0.0
        result = await env.step(action)

    if result.done:
        await env.close()
        return 0.0

    try:
        candidate = ConflictAction.model_validate(normalize_action_dict(payload))
    except Exception:
        await env.close()
        return 0.0

    step_result = await env.step(candidate)
    await env.close()
    reward = float(step_result.reward or 0.0)
    return max(0.0, min(1.0, reward))


def completion_to_text(raw_completion: Any) -> str:
    if isinstance(raw_completion, str):
        return raw_completion

    if isinstance(raw_completion, dict):
        for key in ("content", "text", "generated_text"):
            value = raw_completion.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(raw_completion)

    if isinstance(raw_completion, list) and raw_completion:
        first = raw_completion[0]
        if isinstance(first, dict):
            content = first.get("content")
            if isinstance(content, str):
                return content
        return str(first)

    return str(raw_completion)


def build_reward_fn():
    def reward_func(prompts, completions, task_name=None, history_actions=None, **kwargs):
        tasks = task_name or kwargs.get("task_name") or [TASK_IDS[0]] * len(completions)
        histories = history_actions or kwargs.get("history_actions") or [[] for _ in completions]

        rewards: List[float] = []
        for completion, row_task, row_history in zip(completions, tasks, histories):
            text = completion_to_text(completion)
            score = asyncio.run(reward_from_completion(row_task, row_history, text))
            rewards.append(score)

        return rewards

    return reward_func


def save_rollouts_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def run_trl_training(cfg: TrainConfig, rows: List[Dict[str, Any]]) -> None:
    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Install training deps first: pip install datasets trl transformers accelerate"
        ) from exc

    model_ref: Any = cfg.model_name
    maybe_tokenizer = None

    if cfg.use_unsloth:
        try:
            from unsloth import FastLanguageModel

            model_ref, maybe_tokenizer = FastLanguageModel.from_pretrained(
                model_name=cfg.model_name,
                max_seq_length=cfg.max_prompt_length + cfg.max_completion_length,
                load_in_4bit=True,
            )
            model_ref = FastLanguageModel.get_peft_model(model_ref)
            print("[INFO] Unsloth model initialized.")
        except Exception as exc:
            print(f"[WARN] Unsloth unavailable, falling back to model name string: {exc}")
            model_ref = cfg.model_name
            maybe_tokenizer = None

    dataset = Dataset.from_list(rows)
    reward_func = build_reward_fn()

    grpo_cfg = GRPOConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        num_train_epochs=cfg.num_train_epochs,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        seed=cfg.seed,
    )

    trainer_kwargs = {
        "model": model_ref,
        "reward_funcs": reward_func,
        "args": grpo_cfg,
        "train_dataset": dataset,
    }
    if maybe_tokenizer is not None:
        trainer_kwargs["processing_class"] = maybe_tokenizer

    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Round2 rollout + optional GRPO trainer")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--max-prompt-rows", type=int, default=160)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--train-with-trl", action="store_true")
    parser.add_argument("--use-unsloth", action="store_true")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-dir", type=str, default="./outputs/grpo-round2")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    cfg = TrainConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        rollouts_jsonl=str(Path(args.output_dir) / "rollouts.jsonl"),
        episodes=args.episodes,
        max_prompt_rows=args.max_prompt_rows,
        max_steps=args.max_steps,
        train_with_trl=bool(args.train_with_trl),
        use_unsloth=bool(args.use_unsloth),
        seed=args.seed,
    )

    pack = asyncio.run(collect_prompt_rows(cfg))
    rows = pack["rows"]
    summaries = pack["summaries"]

    save_rollouts_jsonl(Path(cfg.rollouts_jsonl), rows)
    avg_final = sum(item["final_score"] for item in summaries) / float(len(summaries)) if summaries else 0.0

    print(f"[INFO] Collected prompt rows: {len(rows)}")
    print(f"[INFO] Saved rollout dataset: {cfg.rollouts_jsonl}")
    print(f"[INFO] Episode count: {len(summaries)} | Avg final score: {avg_final:.4f}")

    if cfg.train_with_trl:
        run_trl_training(cfg, rows)
    else:
        print("[INFO] Skipping GRPO train. Use --train-with-trl to run policy optimization.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
