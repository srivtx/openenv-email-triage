import asyncio

from src.assistant_conflict_env.environment import PersonalAssistantConflictEnv
from src.assistant_conflict_env.models import ConflictAction


def test_reset_and_step_flow() -> None:
    async def _run() -> None:
        env = PersonalAssistantConflictEnv()
        result = await env.reset("easy_evening_planner")

        assert result.done is False
        assert result.observation.current_conflict is not None

        action = ConflictAction(
            intent="reschedule_event",
            owner="work",
            priority="high",
            proposed_slot="after 20:30",
            needs_clarification=False,
            message_template="Reschedule incident review after dinner and confirm owner.",
        )

        step_result = await env.step(action)

        assert 0.0 <= step_result.reward <= 1.0
        assert step_result.observation.step_index == 1

    asyncio.run(_run())


def test_episode_completes_within_limits() -> None:
    async def _run() -> None:
        env = PersonalAssistantConflictEnv()
        result = await env.reset("easy_evening_planner")

        for _ in range(10):
            if result.done:
                break
            result = await env.step(
                ConflictAction(
                    intent="route_message",
                    owner="self",
                    priority="normal",
                    proposed_slot="",
                    needs_clarification=False,
                    message_template="Routing conflict with basic fallback.",
                )
            )

        state = await env.state()
        assert state.done is True
        assert 0.0 <= state.final_score <= 1.0

    asyncio.run(_run())
