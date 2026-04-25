"""Personal Assistant Conflict Resolver environment.

This module implements the OpenEnv ``step``/``reset`` loop on top of a mutable
:class:`WorldState`. Conflicts are processed via a queue rather than a fixed
index, which lets us:

- **Re-present a conflict after a clarification.** When the agent picks
  ``ask_clarification`` on a case that has hidden information, the env keeps the
  same conflict at the front of the queue, attaches the revealed info on the
  next observation, and grades the follow-up action against the post-reveal
  expectation.
- **Mutate the calendar on reschedule actions.** ``reschedule_event`` updates a
  matching calendar event's start time so the agent's history is visible in the
  world state.
- **Cascade follow-on conflicts.** When a case carries a :class:`CascadeRule`
  and the agent's action satisfies it (e.g. rescheduling past 18:00), a new
  conflict is generated and appended to the queue. The cascade reflects a real
  consequence of the agent's choice instead of being pre-scripted.

The reward floor of 0.10 from the previous version has been removed; the grader
now returns honest 0.0 for fully-wrong actions. Anti-hacking penalties are
strengthened (see :meth:`_compute_penalties`).
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

from .graders import grade_decisions, score_action
from .models import (
    ActionIntent,
    CalendarEvent,
    CascadeRule,
    ClarificationSpec,
    ConflictAction,
    ConflictCase,
    ConflictDecision,
    ConflictExpectation,
    ConflictObservation,
    ConflictState,
    ConflictStepResult,
    PendingClarification,
    TaskDefinition,
    TERMINAL_INTENTS,
    WorldState,
)
from .tasks import DEFAULT_TASK_ID, get_task, list_tasks


PENALTY_REPETITIVE_INTENT = 0.10
PENALTY_PREMATURE_FINALIZE = 0.15
PENALTY_TERMINAL_EARLY = 0.05
PENALTY_SHORT_MESSAGE = 0.04
PENALTY_MISSING_SLOT = 0.05
PENALTY_CLARIFICATION_SPAM = 0.05


class PersonalAssistantConflictEnv:
    def __init__(self, default_task_id: str = DEFAULT_TASK_ID) -> None:
        self._default_task_id = default_task_id
        self._task: Optional[TaskDefinition] = None
        self._state: Optional[ConflictState] = None
        self._queue: List[ConflictCase] = []
        self._last_feedback: str = ""

    @classmethod
    async def from_docker_image(cls, _image_name: Optional[str] = None) -> "PersonalAssistantConflictEnv":
        return cls()

    async def reset(self, task_name: Optional[str] = None) -> ConflictStepResult:
        task_id = task_name or self._default_task_id
        task = get_task(task_id)

        self._task = task
        self._queue = [case.model_copy(deep=True) for case in task.conflicts]
        world = WorldState(
            calendar=[event.model_copy(deep=True) for event in task.initial_calendar],
            pending_clarifications=[],
            revealed_info={},
            cascade_queue=[],
            cascade_count=0,
        )
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
            world=world,
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

        if not self._queue:
            self._state.done = True
            return ConflictStepResult(
                observation=self._observation(),
                reward=0.0,
                done=True,
                info={"last_action_error": "Conflict queue empty.", "step_score": 0.0},
            )

        current_case = self._queue[0]
        revealed_for_current = self._state.world.revealed_info.get(current_case.conflict_id)
        effective_expected = self._effective_expectation(current_case, revealed_for_current is not None)

        score_details = score_action(effective_expected, action)
        penalties = self._compute_penalties(
            action=action,
            current_case=current_case,
            already_revealed=revealed_for_current is not None,
        )
        penalty_total = sum(penalties.values())
        # Reward floor removed: honest 0.0 if the agent fully fails.
        reward = max(0.0, min(1.0, score_details.score - penalty_total))

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
        if action.needs_clarification or action.intent == ActionIntent.ASK_CLARIFICATION:
            self._state.clarification_requests += 1

        # World-state mutations and queue management run AFTER scoring so the
        # agent's reward only reflects what they could have known at decision
        # time.
        keep_at_front = self._handle_clarification(current_case, action, revealed_for_current is not None)
        self._mutate_calendar(current_case, action)
        cascade_added = self._maybe_trigger_cascade(current_case, action)

        if not keep_at_front:
            self._queue.pop(0)

        self._state.step_index += 1
        is_step_limit = self._state.step_index >= self._state.max_steps
        is_queue_done = not self._queue
        self._state.done = bool(is_step_limit or is_queue_done)

        if self._state.score_history:
            self._state.average_step_score = sum(self._state.score_history) / len(self._state.score_history)

        if self._state.done:
            report = grade_decisions(self._state.decisions, self._state.score_history)
            self._state.final_score = report.score
            self._last_feedback = "Episode complete."
        else:
            self._last_feedback = self._feedback_line(score_details.components)

        info: Dict[str, object] = {
            "task_id": self._task.id,
            "step_score": round(score_details.score, 4),
            "reward_components": score_details.components,
            "penalties": penalties,
            "queue_remaining": len(self._queue),
            "cascade_added": cascade_added,
            "kept_for_clarification": keep_at_front,
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_state(self) -> None:
        if self._state is None or self._task is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

    def _effective_expectation(
        self,
        case: ConflictCase,
        already_revealed: bool,
    ) -> ConflictExpectation:
        """Return the expectation to grade against given the reveal state."""

        if already_revealed and case.clarification and case.clarification.post_reveal_expected is not None:
            return case.clarification.post_reveal_expected
        return case.expected

    def _handle_clarification(
        self,
        case: ConflictCase,
        action: ConflictAction,
        already_revealed: bool,
    ) -> bool:
        """Manage the clarification reveal cycle.

        Returns True if the case should remain at the front of the queue (so the
        agent can act on the revealed info next step).
        """

        assert self._state is not None
        if case.clarification is None:
            return False

        if already_revealed:
            # The reveal already happened on a previous step. Move on regardless
            # of the agent's action so we don't loop forever.
            return False

        agent_asked = (
            action.intent == ActionIntent.ASK_CLARIFICATION or action.needs_clarification
        )
        if not agent_asked:
            # Agent skipped the clarification. They've already taken the action,
            # which will score poorly. Advance the queue.
            return False

        # Schedule the reveal: set the revealed_info now so that the NEXT
        # observation includes it, and the next step grades against the post-
        # reveal expectation.
        self._state.world.revealed_info[case.conflict_id] = case.clarification.revealed_summary_suffix
        self._state.world.pending_clarifications.append(
            PendingClarification(
                conflict_id=case.conflict_id,
                reveal_at_step=self._state.step_index + 1,
                revealed_summary_suffix=case.clarification.revealed_summary_suffix,
                post_reveal_expected=case.clarification.post_reveal_expected,
            )
        )
        return True

    def _mutate_calendar(self, case: ConflictCase, action: ConflictAction) -> None:
        """Reflect agent action in the calendar so the world state evolves visibly.

        - ``RESCHEDULE_EVENT``: update the first non-locked event matching the
          action owner; if no match, append a new event for traceability.
        - ``PROPOSE_PLAN``: append a new tentative event with the proposed slot.
        Other intents do not directly mutate the calendar.
        """

        assert self._state is not None
        slot_minutes = _slot_to_minutes(action.proposed_slot)

        if action.intent == ActionIntent.RESCHEDULE_EVENT and slot_minutes is not None:
            new_start = _minutes_to_time(slot_minutes)
            new_end = _minutes_to_time(min(24 * 60 - 1, slot_minutes + 60))
            for event in self._state.world.calendar:
                if event.locked:
                    continue
                if event.owner != action.owner:
                    continue
                event.start = new_start
                event.end = new_end
                return
            self._state.world.calendar.append(
                CalendarEvent(
                    event_id=f"E-resched-{case.conflict_id}",
                    title=f"rescheduled ({case.source})",
                    start=new_start,
                    end=new_end,
                    owner=action.owner,
                    locked=False,
                )
            )
            return

        if action.intent == ActionIntent.PROPOSE_PLAN and slot_minutes is not None:
            new_start = _minutes_to_time(slot_minutes)
            new_end = _minutes_to_time(min(24 * 60 - 1, slot_minutes + 60))
            self._state.world.calendar.append(
                CalendarEvent(
                    event_id=f"E-plan-{case.conflict_id}",
                    title=f"planned ({case.source})",
                    start=new_start,
                    end=new_end,
                    owner=action.owner,
                    locked=False,
                )
            )

    def _maybe_trigger_cascade(self, case: ConflictCase, action: ConflictAction) -> bool:
        """If the case carries a cascade rule and the action satisfies it, append a follow-on conflict."""

        assert self._state is not None
        rule = case.cascade_rule
        if rule is None:
            return False
        if action.intent != rule.trigger_intent:
            return False
        if rule.trigger_owner is not None and action.owner != rule.trigger_owner:
            return False
        if rule.trigger_slot_after_minutes is not None:
            slot_minutes = _slot_to_minutes(action.proposed_slot)
            if slot_minutes is None or slot_minutes < rule.trigger_slot_after_minutes:
                return False

        self._state.world.cascade_count += 1
        cascade_id = f"{rule.follow_on_id_prefix}-{case.conflict_id}-cas{self._state.world.cascade_count:02d}"
        cascade_case = ConflictCase(
            conflict_id=cascade_id,
            source=rule.follow_on_source,
            summary=rule.follow_on_summary,
            constraints=list(rule.follow_on_constraints),
            expected=rule.follow_on_expected,
            is_cascade=True,
        )
        # Append to end of queue: cascades resolve after current pending work.
        self._queue.append(cascade_case)
        self._state.world.cascade_queue.append(cascade_case)
        return True

    def _observation(self) -> ConflictObservation:
        self._require_state()
        assert self._state is not None
        assert self._task is not None

        current_conflict: Optional[ConflictCase] = None
        open_risks: List[str] = []
        if not self._state.done and self._queue:
            base = self._queue[0]
            revealed = self._state.world.revealed_info.get(base.conflict_id)
            current_conflict = base.model_copy(deep=True)
            if revealed:
                current_conflict.summary = base.summary + revealed
                current_conflict.revealed_info = revealed
                # Surface the post-reveal expectation hint so the agent knows
                # the info is now available; expectations themselves stay
                # hidden of course.
            open_risks = current_conflict.constraints[:3]

        return ConflictObservation(
            task_id=self._state.task_id,
            task_difficulty=self._state.task_difficulty,
            step_index=self._state.step_index,
            max_steps=self._state.max_steps,
            processed_count=len(self._state.decisions),
            remaining_count=len(self._queue),
            current_conflict=current_conflict,
            history=self._state.decisions[-3:],
            open_risks=open_risks,
            last_feedback=self._last_feedback,
            available_calendar=[ev.model_copy(deep=True) for ev in self._state.world.calendar],
            pending_clarifications=[pc.model_copy(deep=True) for pc in self._state.world.pending_clarifications],
            cascade_count=self._state.world.cascade_count,
        )

    def _compute_penalties(
        self,
        action: ConflictAction,
        current_case: ConflictCase,
        already_revealed: bool,
    ) -> Dict[str, float]:
        assert self._state is not None

        penalties = {
            "short_message": 0.0,
            "repetitive_intent": 0.0,
            "premature_finalize": 0.0,
            "terminal_early": 0.0,
            "missing_slot": 0.0,
            "clarification_spam": 0.0,
        }

        if action.message_template and len(action.message_template) < 16:
            penalties["short_message"] = PENALTY_SHORT_MESSAGE

        if len(self._state.decisions) >= 2:
            last_three = self._state.decisions[-2:]
            recent_intents = [d.intent for d in last_three] + [action.intent]
            if len(set(recent_intents)) == 1 and len(recent_intents) >= 3:
                penalties["repetitive_intent"] = PENALTY_REPETITIVE_INTENT

        # Premature finalization penalty applies to FINALIZE_ITINERARY when work
        # remains. ``terminal_early`` covers any other terminal-style intent
        # used to short-circuit an episode.
        remaining_after_this = max(len(self._queue) - 1, 0)
        if action.intent == ActionIntent.FINALIZE_ITINERARY and remaining_after_this >= 1:
            penalties["premature_finalize"] = PENALTY_PREMATURE_FINALIZE
        elif action.intent in TERMINAL_INTENTS and remaining_after_this >= 1:
            penalties["terminal_early"] = PENALTY_TERMINAL_EARLY

        effective_expected = self._effective_expectation(current_case, already_revealed)
        if effective_expected.require_slot and not action.proposed_slot:
            penalties["missing_slot"] = PENALTY_MISSING_SLOT

        clarification_warranted = (
            current_case.expected.block_if_missing_context
            and current_case.clarification is not None
            and not already_revealed
        )
        if (
            (action.needs_clarification or action.intent == ActionIntent.ASK_CLARIFICATION)
            and not clarification_warranted
        ):
            penalties["clarification_spam"] = PENALTY_CLARIFICATION_SPAM

        return penalties

    @staticmethod
    def _feedback_line(components: Dict[str, float]) -> str:
        hints: List[str] = []
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
        if components.get("message", 0.0) < 0.5:
            hints.append("message quality")
        if not hints:
            return "Strong conflict-resolution action."
        return "Improve: " + ", ".join(hints) + "."


def _slot_to_minutes(slot: str) -> Optional[int]:
    """Extract a 24h HH:MM time from a free-form slot string. Returns None if not found."""

    import re

    if not slot:
        return None
    match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", slot)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    return hour * 60 + minute


def _minutes_to_time(total_minutes: int) -> str:
    total_minutes = max(0, min(24 * 60 - 1, total_minutes))
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"
