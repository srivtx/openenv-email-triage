"""Reward grading for the personal assistant environment.

Weight derivation
-----------------

The total step score is a weighted sum of six components, each in ``[0, 1]``.
Weights sum to ``1.0`` and were chosen as follows (the previous, undocumented
weights of ``[0.34, 0.20, 0.15, 0.14, 0.10, 0.07]`` are documented here for
auditability):

- ``intent`` (0.40): which action category to take is the single most
  consequential choice. Mis-routed intent makes everything downstream wrong.
- ``owner``  (0.20): assigning the right principal (work / family / legal /
  finance) is the second-most consequential routing decision.
- ``slot``   (0.20): when a conflict requires a specific time, slot precision
  matters as much as ownership; when no slot is required the component is a
  free 1.0 (so its contribution becomes a constant 0.20 across actions).
- ``priority`` (0.10): a four-class noisy proxy; partial credit (0.5) is given
  for off-by-one mistakes since adjacent priorities are often defensible.
- ``clarification`` (0.05): a single boolean; coarse signal so it gets a
  small weight. ``1.0`` only when the agent's ``needs_clarification`` matches
  the expected ``block_if_missing_context``.
- ``message`` (0.05): a structural proxy for message quality (length plus a
  required action verb derived from the intent). Deliberately a tiebreaker
  weight because text proxies are easy to game.

Slot scoring
------------

The previous version used substring matching on tokens like ``"pm"``, ``"after"``,
or ``"today"``, which let the agent earn partial credit by stuffing magic
words. The new ``_slot_score`` extracts a ``HH:MM`` 24h time from both the
proposed slot and the expected hint and scores by time-distance:

- exact match: ``1.0``
- within 30 min: ``0.7``
- within 60 min: ``0.4``
- within 120 min: ``0.2``
- otherwise: ``0.0``

Message scoring
---------------

The previous keyword-substring score rewarded literal keyword copy-paste. The
new ``_message_score`` checks (a) length above a small threshold and (b) the
presence of an action-verb tied to the chosen intent. This is still a proxy,
but it is harder to game without saying something on-topic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .models import (
    ActionIntent,
    ConflictAction,
    ConflictDecision,
    ConflictExpectation,
    Priority,
    TaskDefinition,
)


_PRIORITY_INDEX = {
    Priority.LOW: 0,
    Priority.NORMAL: 1,
    Priority.HIGH: 2,
    Priority.URGENT: 3,
}


_INTENT_VERBS: Dict[ActionIntent, Sequence[str]] = {
    ActionIntent.ROUTE_MESSAGE: ("route", "forward", "send"),
    ActionIntent.PROPOSE_PLAN: ("propose", "plan", "schedule"),
    ActionIntent.RESCHEDULE_EVENT: ("reschedule", "move", "shift"),
    ActionIntent.DELEGATE_TASK: ("delegate", "assign", "hand"),
    ActionIntent.ASK_CLARIFICATION: ("clarify", "confirm", "ask", "verify"),
    ActionIntent.FINALIZE_ITINERARY: ("finalize", "summary", "summarize", "wrap"),
}


WEIGHTS: Dict[str, float] = {
    "intent": 0.40,
    "owner": 0.20,
    "slot": 0.20,
    "priority": 0.10,
    "clarification": 0.05,
    "message": 0.05,
}


_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")


@dataclass(frozen=True)
class ScoreBreakdown:
    score: float
    components: Dict[str, float]


@dataclass(frozen=True)
class GradeReport:
    task_id: str
    total_cases: int
    covered_cases: int
    score: float


def _extract_minutes(text: str) -> Optional[int]:
    match = _TIME_RE.search(text or "")
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def _priority_score(expected: Priority, actual: Priority) -> float:
    distance = abs(_PRIORITY_INDEX[expected] - _PRIORITY_INDEX[actual])
    if distance == 0:
        return 1.0
    if distance == 1:
        return 0.5
    return 0.0


def _slot_score(expected_slot_hint: str, proposed_slot: str, require_slot: bool) -> float:
    if not require_slot:
        return 1.0

    if not proposed_slot or not proposed_slot.strip():
        return 0.0

    proposed_minutes = _extract_minutes(proposed_slot)
    expected_minutes = _extract_minutes(expected_slot_hint)

    if proposed_minutes is None:
        # No parseable HH:MM time at all - fail. (No more credit for "later today".)
        return 0.0

    if expected_minutes is None:
        # The expectation didn't pin down a time but the agent supplied one;
        # award best-effort credit since we can't rank precision.
        return 0.5

    distance = abs(proposed_minutes - expected_minutes)
    if distance == 0:
        return 1.0
    if distance <= 30:
        return 0.7
    if distance <= 60:
        return 0.4
    if distance <= 120:
        return 0.2
    return 0.0


def _clarification_score(block_if_missing_context: bool, needs_clarification: bool) -> float:
    if block_if_missing_context:
        return 1.0 if needs_clarification else 0.0
    # When clarification isn't warranted, asking is mildly penalized in the
    # component score (separately from the "clarification_spam" env penalty).
    return 0.75 if needs_clarification else 1.0


def _message_score(intent: ActionIntent, message_template: str) -> float:
    """Structural quality proxy.

    Three sub-checks combined:

    - ``length`` (40%): >=30 chars full credit, >=16 partial credit, else 0.
    - ``verb`` (30%): on-topic action verb derived from the chosen intent.
    - ``diversity`` (30%): at least 4 distinct words; >=6 distinct words full
      credit. This kills the simplest keyword-stuffing strategy where the agent
      just dumps the keyword list as comma-separated tokens.
    """

    text = (message_template or "").strip().lower()
    if not text:
        return 0.0

    length_score = 1.0 if len(text) >= 30 else 0.5 if len(text) >= 16 else 0.0
    verbs = _INTENT_VERBS.get(intent, ())
    verb_score = 1.0 if any(verb in text for verb in verbs) else 0.0

    words = {w.strip(".,;:!?") for w in text.split() if w.strip(".,;:!?")}
    diversity_score = 1.0 if len(words) >= 6 else 0.5 if len(words) >= 4 else 0.0

    return 0.4 * length_score + 0.3 * verb_score + 0.3 * diversity_score


def score_action(expected: ConflictExpectation, action: ConflictAction) -> ScoreBreakdown:
    intent_score = 1.0 if action.intent == expected.intent else 0.0
    owner_score = 1.0 if action.owner == expected.owner else 0.0
    priority_score = _priority_score(expected.priority, action.priority)
    slot_score = _slot_score(expected.expected_slot_hint, action.proposed_slot, expected.require_slot)
    clarification_score = _clarification_score(expected.block_if_missing_context, action.needs_clarification)
    message_score = _message_score(expected.intent, action.message_template)

    components = {
        "intent": intent_score,
        "owner": owner_score,
        "priority": priority_score,
        "slot": slot_score,
        "clarification": clarification_score,
        "message": message_score,
    }

    weighted = sum(WEIGHTS[name] * value for name, value in components.items())
    return ScoreBreakdown(score=max(0.0, min(1.0, weighted)), components=components)


def grade_task_decisions(task: TaskDefinition, decisions: Iterable[ConflictDecision]) -> GradeReport:
    """Backward-compatible aggregate grader: averages per-case scores against the task's expected fields.

    Used by the legacy static fixtures and existing tests. The procedural env
    prefers :func:`grade_decisions` which uses the per-step scores already
    computed by ``score_action`` during the episode.
    """

    decision_by_id = {item.conflict_id: item for item in decisions}

    case_scores: List[float] = []
    for case in task.conflicts:
        decision = decision_by_id.get(case.conflict_id)
        if decision is None:
            case_scores.append(0.0)
            continue

        action = ConflictAction(
            intent=decision.intent,
            owner=decision.owner,
            priority=decision.priority,
            proposed_slot=decision.proposed_slot,
            needs_clarification=decision.needs_clarification,
            message_template=decision.message_template,
        )
        case_scores.append(score_action(case.expected, action).score)

    score = sum(case_scores) / float(len(task.conflicts)) if task.conflicts else 0.0
    return GradeReport(
        task_id=task.id,
        total_cases=len(task.conflicts),
        covered_cases=len(decision_by_id),
        score=max(0.0, min(1.0, score)),
    )


def grade_decisions(
    decisions: Iterable[ConflictDecision],
    score_history: Iterable[float],
) -> GradeReport:
    """Aggregate the per-step scores already computed by the env.

    Used at end-of-episode in ``PersonalAssistantConflictEnv.step``. Averaging
    here lets the env reflect what really happened during the episode (including
    re-presented conflicts after a clarification reveal) instead of re-grading
    from scratch as if every conflict were independent.
    """

    decisions = list(decisions)
    history = list(score_history)
    if not history:
        return GradeReport(task_id="", total_cases=len(decisions), covered_cases=0, score=0.0)
    score = sum(history) / float(len(history))
    return GradeReport(
        task_id="",
        total_cases=len(decisions),
        covered_cases=len(decisions),
        score=max(0.0, min(1.0, score)),
    )


def get_task_graders() -> Dict[str, str]:
    return {
        "easy_evening_planner": "grade_task_decisions",
        "medium_multi_party_negotiation": "grade_task_decisions",
        "hard_cascade_replanning": "grade_task_decisions",
    }
