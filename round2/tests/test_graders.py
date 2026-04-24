from src.assistant_conflict_env.graders import grade_task_decisions
from src.assistant_conflict_env.models import ConflictDecision
from src.assistant_conflict_env.tasks import get_task


def test_grade_score_is_bounded() -> None:
    task = get_task("easy_evening_planner")

    decisions = [
        ConflictDecision(
            conflict_id=case.conflict_id,
            intent=case.expected.intent,
            owner=case.expected.owner,
            priority=case.expected.priority,
            proposed_slot=case.expected.expected_slot_hint,
            needs_clarification=case.expected.block_if_missing_context,
            message_template="timeline confirm with owner and fallback",
        )
        for case in task.conflicts
    ]

    report = grade_task_decisions(task, decisions)
    assert 0.0 <= report.score <= 1.0


def test_grader_is_deterministic() -> None:
    task = get_task("medium_multi_party_negotiation")

    decisions = [
        ConflictDecision(
            conflict_id=case.conflict_id,
            intent=case.expected.intent,
            owner=case.expected.owner,
            priority=case.expected.priority,
            proposed_slot=case.expected.expected_slot_hint,
            needs_clarification=case.expected.block_if_missing_context,
            message_template="clarify plan with fallback and timeline",
        )
        for case in task.conflicts
    ]

    first = grade_task_decisions(task, decisions).score
    second = grade_task_decisions(task, decisions).score
    assert first == second
