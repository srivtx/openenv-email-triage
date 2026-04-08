from src.email_triage_env.graders import grade_task_decisions
from src.email_triage_env.models import EmailDecision
from src.email_triage_env.tasks import get_task


def test_grade_score_is_bounded() -> None:
    task = get_task("easy_priority_routing")

    decisions = [
        EmailDecision(
            email_id=case.email_id,
            category=case.expected.category,
            priority=case.expected.priority,
            team=case.expected.team,
            mark_spam=case.expected.mark_spam,
            response_template="invoice payment follow-up",
        )
        for case in task.emails
    ]

    report = grade_task_decisions(task, decisions)
    assert 0.0 <= report.score <= 1.0


def test_grader_is_deterministic() -> None:
    task = get_task("medium_mixed_inbox")

    decisions = [
        EmailDecision(
            email_id=case.email_id,
            category=case.expected.category,
            priority=case.expected.priority,
            team=case.expected.team,
            mark_spam=case.expected.mark_spam,
            response_template="security incident handled",
        )
        for case in task.emails
    ]

    first = grade_task_decisions(task, decisions).score
    second = grade_task_decisions(task, decisions).score
    assert first == second
