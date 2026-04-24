from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from .models import ConflictAction, ConflictDecision, ConflictExpectation, Priority, TaskDefinition


_PRIORITY_INDEX = {
    Priority.LOW: 0,
    Priority.NORMAL: 1,
    Priority.HIGH: 2,
    Priority.URGENT: 3,
}


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

    slot = proposed_slot.strip().lower()
    if not slot:
        return 0.0

    hint = expected_slot_hint.strip().lower()
    if hint and hint in slot:
        return 1.0

    if any(token in slot for token in ["am", "pm", "today", "tomorrow", "after", "before", ":"]):
        return 0.6

    return 0.2


def _clarification_score(block_if_missing_context: bool, needs_clarification: bool) -> float:
    if block_if_missing_context:
        return 1.0 if needs_clarification else 0.0
    return 0.75 if needs_clarification else 1.0


def _keyword_score(required_keywords: List[str], message_template: str) -> float:
    if not required_keywords:
        return 1.0

    text = message_template.lower()
    hits = sum(1 for token in required_keywords if token.lower() in text)
    return hits / float(len(required_keywords))


def score_action(expected: ConflictExpectation, action: ConflictAction) -> ScoreBreakdown:
    intent_score = 1.0 if action.intent == expected.intent else 0.0
    owner_score = 1.0 if action.owner == expected.owner else 0.0
    priority_score = _priority_score(expected.priority, action.priority)
    slot_score = _slot_score(expected.expected_slot_hint, action.proposed_slot, expected.require_slot)
    clarification_score = _clarification_score(expected.block_if_missing_context, action.needs_clarification)
    keyword_score = _keyword_score(expected.required_keywords, action.message_template)

    components = {
        "intent": intent_score,
        "owner": owner_score,
        "priority": priority_score,
        "slot": slot_score,
        "clarification": clarification_score,
        "keyword": keyword_score,
    }

    weighted = (
        0.34 * intent_score
        + 0.20 * owner_score
        + 0.15 * priority_score
        + 0.14 * slot_score
        + 0.10 * clarification_score
        + 0.07 * keyword_score
    )

    return ScoreBreakdown(score=max(0.0, min(1.0, weighted)), components=components)


def grade_task_decisions(task: TaskDefinition, decisions: Iterable[ConflictDecision]) -> GradeReport:
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


def get_task_graders() -> Dict[str, str]:
    return {
        "easy_evening_planner": "grade_task_decisions",
        "medium_multi_party_negotiation": "grade_task_decisions",
        "hard_cascade_replanning": "grade_task_decisions",
    }
