from __future__ import annotations

from typing import Dict, List, Optional

from .graders import grade_task_decisions, score_action
from .models import (
    EmailDecision,
    EmailTriageAction,
    EmailTriageObservation,
    EmailTriageState,
    EmailTriageStepResult,
    TaskDefinition,
)
from .tasks import DEFAULT_TASK_ID, get_task, list_tasks


class EmailTriageEnv:
    def __init__(self, default_task_id: str = DEFAULT_TASK_ID) -> None:
        self._default_task_id = default_task_id
        self._task: Optional[TaskDefinition] = None
        self._state: Optional[EmailTriageState] = None
        self._last_feedback: str = ""

    @classmethod
    async def from_docker_image(cls, _image_name: Optional[str] = None) -> "EmailTriageEnv":
        return cls()

    async def reset(self, task_name: Optional[str] = None) -> EmailTriageStepResult:
        task_id = task_name or self._default_task_id
        task = get_task(task_id)

        self._task = task
        self._state = EmailTriageState(
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
            last_action_error=None,
        )
        self._last_feedback = "Episode reset. Review the current email and route it."

        return EmailTriageStepResult(
            observation=self._observation(),
            reward=0.0,
            done=False,
            info={"task_id": task.id, "last_action_error": None},
        )

    async def step(self, action: EmailTriageAction) -> EmailTriageStepResult:
        self._require_state()
        assert self._task is not None
        assert self._state is not None

        if self._state.done:
            return EmailTriageStepResult(
                observation=self._observation(),
                reward=0.0,
                done=True,
                info={
                    "last_action_error": "Episode already completed. Call reset() before step().",
                    "step_score": 0.0,
                    "final_score": self._state.final_score,
                },
            )

        current_case = self._task.emails[self._state.step_index]
        score_details = score_action(current_case.expected, action)

        penalties = self._compute_penalties(action)
        penalty_total = sum(penalties.values())
        reward = max(0.0, min(1.0, 0.15 + 0.85 * score_details.score - penalty_total))

        decision = EmailDecision(
            email_id=current_case.email_id,
            category=action.category,
            priority=action.priority,
            team=action.team,
            mark_spam=action.mark_spam,
            response_template=action.response_template,
        )

        self._state.decisions.append(decision)
        self._state.score_history.append(score_details.score)
        self._state.reward_history.append(reward)
        self._state.cumulative_reward += reward

        self._state.step_index += 1
        is_step_limit = self._state.step_index >= self._state.max_steps
        is_inbox_done = self._state.step_index >= len(self._task.emails)
        self._state.done = bool(is_step_limit or is_inbox_done)

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

        return EmailTriageStepResult(
            observation=self._observation(),
            reward=reward,
            done=self._state.done,
            info=info,
        )

    async def state(self) -> EmailTriageState:
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

    def _observation(self) -> EmailTriageObservation:
        self._require_state()
        assert self._state is not None
        assert self._task is not None

        current_email = None
        if not self._state.done and self._state.step_index < len(self._task.emails):
            current_email = self._task.emails[self._state.step_index]

        return EmailTriageObservation(
            task_id=self._state.task_id,
            task_difficulty=self._state.task_difficulty,
            step_index=self._state.step_index,
            max_steps=self._state.max_steps,
            processed_count=len(self._state.decisions),
            remaining_count=max(len(self._task.emails) - self._state.step_index, 0),
            current_email=current_email,
            history=self._state.decisions[-3:],
            last_feedback=self._last_feedback,
        )

    def _compute_penalties(self, action: EmailTriageAction) -> Dict[str, float]:
        assert self._state is not None

        penalties = {
            "short_response": 0.0,
            "repetitive_routing": 0.0,
            "aggressive_escalation": 0.0,
        }

        if action.response_template and len(action.response_template) < 12:
            penalties["short_response"] = 0.05

        if len(self._state.decisions) >= 2:
            last_two = self._state.decisions[-2:]
            if all(item.category == action.category and item.team == action.team for item in last_two):
                penalties["repetitive_routing"] = 0.05

        if action.priority.value == "urgent":
            penalties["aggressive_escalation"] = 0.02

        return penalties

    @staticmethod
    def _feedback_line(components: Dict[str, float]) -> str:
        hints = []
        if components.get("category", 0.0) < 1.0:
            hints.append("category")
        if components.get("priority", 0.0) < 1.0:
            hints.append("priority")
        if components.get("team", 0.0) < 1.0:
            hints.append("team")
        if components.get("spam", 0.0) < 1.0:
            hints.append("spam")
        if components.get("keyword", 0.0) < 0.5:
            hints.append("response quality")

        if not hints:
            return "Strong routing decision."

        return "Improve: " + ", ".join(hints) + "."
