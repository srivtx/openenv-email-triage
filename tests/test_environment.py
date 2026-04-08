import pytest

from src.email_triage_env.environment import EmailTriageEnv
from src.email_triage_env.models import EmailTriageAction


@pytest.mark.asyncio
async def test_reset_and_step_flow() -> None:
    env = EmailTriageEnv()
    result = await env.reset("easy_priority_routing")

    assert result.done is False
    assert result.observation.current_email is not None

    action = EmailTriageAction(
        category="billing",
        priority="high",
        team="billing",
        mark_spam=False,
        response_template="invoice payment follow-up routed to billing",
    )

    step_result = await env.step(action)

    assert 0.0 <= step_result.reward <= 1.0
    assert step_result.observation.step_index == 1


@pytest.mark.asyncio
async def test_episode_completes_within_limits() -> None:
    env = EmailTriageEnv()
    result = await env.reset("easy_priority_routing")

    for _ in range(10):
        if result.done:
            break
        result = await env.step(
            EmailTriageAction(
                category="other",
                priority="normal",
                team="support",
                mark_spam=False,
                response_template="routed",
            )
        )

    state = await env.state()
    assert state.done is True
    assert 0.0 <= state.final_score <= 1.0
