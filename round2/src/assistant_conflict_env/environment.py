from __future__ import annotations

from typing import Dict, List, Optional

from .graders import grade_task_decisions, score_action
from .models import (
    ConflictAction,
    ConflictCase,
    ConflictDecision,
    ConflictObservation,
    ConflictState,
    ConflictStepResult,
    TaskDefinition,
)
from .tasks import DEFAULT_TASK_ID, get_task, list_tasks


class PersonalAssistantConflictEnv:
    def __init__(self, default_task_id: str = DEFAULT_TASK_ID) -> None:
        self._default_task_id = default_task_id
        self._task: Optional[TaskDefinition] = None
        self._state: Optional[ConflictState] = None
        self._last_feedback: str = ""

    @classmethod
    async def from_docker_image(cls, _image_name: Optional[str] = None) -> "PersonalAssistantConflictEnv":
        return cls()

    async def reset(self, task_name: Optional[str] = None) -> ConflictStepResult:
        task_id = task_name or self._default_task_id
        task = get_task(task_id)

        self._task = task
        self._state = ConflictState(
            task_id=task.id,
            task_difficulty=task.difficulty,
            step_index=0,
            max_steps=task.max_steps,
            done=False,
            cumulative_reward=0.0,
            average_step_score=0.0,
            final_score=0.0,
            decisions=[],
            score_history=[],
            reward_history=[],
            clarification_requests=0,
            last_action_error=None,
        )
        self._last_feedback = "Episode reset. Review the current conflict and choose one action."

        return ConflictStepResult(
            observation=self._observation(),
            reward=0.0,
            done=False,
            info={"task_id": task.id, "last_action_error": None},
        )

    async def step(self, action: ConflictAction) -> ConflictStepResult:
        self._require_state()
        assert self._task is not None
        assert self._state is not None

        if self._state.done:
            return ConflictStepResult(
                observation=self._observation(),
                reward=0.0,
                done=True,
                info={
                    "last_action_error": "Episode already completed. Call reset() before step().",
                    "step_score": 0.0,
                    "final_score": self._state.final_score,
                },
            )

        current_case = self._task.conflicts[self._state.step_index]
        score_details = score_action(current_case.expected, action)

        penalties = self._compute_penalties(action=action, current_case=current_case)
        penalty_total = sum(penalties.values())
        reward = max(0.0, min(1.0, 0.10 + 0.90 * score_details.score - penalty_total))

        decision = ConflictDecision(
            conflict_id=current_case.conflict_id,
            intent=action.intent,
            owner=action.owner,
            priority=action.priority,
            proposed_slot=action.proposed_slot,
            needs_clarification=action.needs_clarification,
            message_template=action.message_template,
        )

        self._state.decisions.append(decision)
        self._state.score_history.append(score_details.score)
        self._state.reward_history.append(reward)
        self._state.cumulative_reward += reward
        if action.needs_clarification:
            self._state.clarification_requests += 1

        self._state.step_index += 1
        is_step_limit = self._state.step_index >= self._state.max_steps
        is_queue_done = self._state.step_index >= len(self._task.conflicts)
        self._state.done = bool(is_step_limit or is_queue_done)

        if self._state.score_history:
            self._state.average_step_score = sum(self._state.score_history) / len(self._state.score_history)

        if self._state.done:
            report = grade_task_decisions(self._task, self._state.decisions)
            self._state.final_score = report.score
            self._last_feedback = "Episode complete."
        else:
            self._last_feedback = self._feedback_line(score_details.components)

        info: Dict[str, object] = {
            "task_id": self._task.id,
            "step_score": round(score_details.score, 4),
            "reward_components": score_details.components,
            "penalties": penalties,
            "last_action_error": None,
        }
        if self._state.done:
            info["final_score"] = round(self._state.final_score, 4)

        return ConflictStepResult(
            observation=self._observation(),
            reward=reward,
            done=self._state.done,
            info=info,
        )

    async def state(self) -> ConflictState:
        self._require_state()
        assert self._state is not None
        return self._state.model_copy(deep=True)

    async def close(self) -> None:
        return None

    def tasks(self) -> List[TaskDefinition]:
        return list_tasks()

    def _require_state(self) -> None:
        if self._state is None or self._task is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

    def _observation(self) -> ConflictObservation:
        self._require_state()
        assert self._state is not None
        assert self._task is not None

        current_conflict = None
        open_risks: List[str] = []
        if not self._state.done and self._state.step_index < len(self._task.conflicts):
            current_conflict = self._task.conflicts[self._state.step_index]
            open_risks = current_conflict.constraints[:3]

        return ConflictObservation(
            task_id=self._state.task_id,
            task_difficulty=self._state.task_difficulty,
            step_index=self._state.step_index,
            max_steps=self._state.max_steps,
            processed_count=len(self._state.decisions),
            remaining_count=max(len(self._task.conflicts) - self._state.step_index, 0),
            current_conflict=current_conflict,
            history=self._state.decisions[-3:],
            open_risks=open_risks,
            last_feedback=self._last_feedback,
        )

    def _compute_penalties(self, action: ConflictAction, current_case: ConflictCase) -> Dict[str, float]:
        assert self._state is not None

        penalties = {
            "short_message": 0.0,
            "repetitive_intent": 0.0,
            "premature_finalize": 0.0,
            "missing_slot": 0.0,
            "clarification_spam": 0.0,
        }

        if action.message_template and len(action.message_template) < 16:
            penalties["short_message"] = 0.04

        if len(self._state.decisions) >= 2:
            last_two = self._state.decisions[-2:]
            if all(item.intent == action.intent for item in last_two):
                penalties["repetitive_intent"] = 0.05

        remaining = max(len(self._task.conflicts) - self._state.step_index, 0)
        if action.intent.value == "finalize_itinerary" and remaining > 1:
            penalties["premature_finalize"] = 0.08

        if current_case.expected.require_slot and not action.proposed_slot:
            penalties["missing_slot"] = 0.05

        if action.needs_clarification and not current_case.expected.block_if_missing_context:
            penalties["clarification_spam"] = 0.03

        return penalties

    @staticmethod
    def _feedback_line(components: Dict[str, float]) -> str:
        hints = []
        if components.get("intent", 0.0) < 1.0:
            hints.append("intent")
        if components.get("owner", 0.0) < 1.0:
            hints.append("owner")
        if components.get("priority", 0.0) < 1.0:
            hints.append("priority")
        if components.get("slot", 0.0) < 0.8:
            hints.append("slot")
        if components.get("clarification", 0.0) < 1.0:
            hints.append("clarification")
        if components.get("keyword", 0.0) < 0.5:
            hints.append("message quality")

        if not hints:
            return "Strong conflict-resolution action."

        return "Improve: " + ", ".join(hints) + "."
