from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from .models import EmailCase, EmailDecision, EmailExpectation, EmailTriageAction, Priority, TaskDefinition


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


def _keyword_score(required_keywords: List[str], response_template: str) -> float:
    if not required_keywords:
        return 1.0

    text = response_template.lower()
    hits = sum(1 for token in required_keywords if token.lower() in text)
    return hits / float(len(required_keywords))


def score_action(expected: EmailExpectation, action: EmailTriageAction) -> ScoreBreakdown:
    category_score = 1.0 if action.category == expected.category else 0.0
    priority_score = _priority_score(expected.priority, action.priority)
    team_score = 1.0 if action.team == expected.team else 0.0
    spam_score = 1.0 if action.mark_spam == expected.mark_spam else 0.0
    keyword_score = _keyword_score(expected.required_keywords, action.response_template)

    components = {
        "category": category_score,
        "priority": priority_score,
        "team": team_score,
        "spam": spam_score,
        "keyword": keyword_score,
    }

    weighted = (
        0.35 * category_score
        + 0.25 * priority_score
        + 0.25 * team_score
        + 0.10 * spam_score
        + 0.05 * keyword_score
    )

    return ScoreBreakdown(score=max(0.0, min(1.0, weighted)), components=components)


def grade_task_decisions(task: TaskDefinition, decisions: Iterable[EmailDecision]) -> GradeReport:
    decision_by_id = {item.email_id: item for item in decisions}

    case_scores: List[float] = []
    for case in task.emails:
        decision = decision_by_id.get(case.email_id)
        if decision is None:
            case_scores.append(0.0)
            continue

        action = EmailTriageAction(
            category=decision.category,
            priority=decision.priority,
            team=decision.team,
            mark_spam=decision.mark_spam,
            response_template=decision.response_template,
        )
        case_scores.append(score_action(case.expected, action).score)

    score = sum(case_scores) / float(len(task.emails)) if task.emails else 0.0
    return GradeReport(
        task_id=task.id,
        total_cases=len(task.emails),
        covered_cases=len(decision_by_id),
        score=max(0.0, min(1.0, score)),
    )


def get_task_graders() -> Dict[str, str]:
    return {
        "easy_priority_routing": "grade_task_decisions",
        "medium_mixed_inbox": "grade_task_decisions",
        "hard_ambiguous_escalations": "grade_task_decisions",
    }
