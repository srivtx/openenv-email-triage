"""Tests for world-state mechanics: clarification reveals, cascades, calendar mutation."""

from __future__ import annotations

import asyncio

import pytest

from src.assistant_conflict_env.conflict_generator import generate_episode
from src.assistant_conflict_env.environment import PersonalAssistantConflictEnv
from src.assistant_conflict_env.models import (
    ActionIntent,
    ConflictAction,
    Owner,
    Priority,
)


def _oracle_action(case, world) -> ConflictAction:
    """Pick the action exactly matching a case's expected expectation.

    Used to drive the env with perfect play so we can verify mechanics fire
    correctly without confounding training noise.
    """

    revealed = world.revealed_info.get(case.conflict_id) if world else None
    if revealed and case.clarification and case.clarification.post_reveal_expected:
        expected = case.clarification.post_reveal_expected
    else:
        expected = case.expected
    return ConflictAction(
        intent=expected.intent,
        owner=expected.owner,
        priority=expected.priority,
        proposed_slot=expected.expected_slot_hint or "",
        needs_clarification=expected.block_if_missing_context and (revealed is None),
        message_template=(
            f"{expected.intent.value.replace('_', ' ')} for {expected.owner.value}; "
            f"keywords: {', '.join(expected.required_keywords) or 'follow up'}."
        ),
    )


def _run_episode_with_oracle(seed: int, difficulty: str = "hard"):
    async def _go():
        env = PersonalAssistantConflictEnv()
        task = generate_episode(seed=seed, difficulty=difficulty)
        from src.assistant_conflict_env.models import ConflictState, WorldState

        env._task = task
        env._queue = [c.model_copy(deep=True) for c in task.conflicts]
        env._state = ConflictState(
            task_id=task.id,
            task_difficulty=task.difficulty,
            step_index=0,
            max_steps=task.max_steps,
            done=False,
            world=WorldState(calendar=[c.model_copy(deep=True) for c in task.initial_calendar]),
        )
        env._last_feedback = "reset"

        events = []
        steps = 0
        while not env._state.done and steps < env._state.max_steps:
            current = env._queue[0]
            action = _oracle_action(current, env._state.world)
            res = await env.step(action)
            events.append(
                {
                    "conflict_id": current.conflict_id,
                    "is_cascade": current.is_cascade,
                    "reward": res.reward,
                    "components": res.info["reward_components"],
                    "kept": res.info["kept_for_clarification"],
                    "cascade_added": res.info["cascade_added"],
                }
            )
            steps += 1
        return env._state, events

    return asyncio.run(_go())


def test_oracle_play_scores_high_on_procedural_episode() -> None:
    state, events = _run_episode_with_oracle(seed=42, difficulty="hard")
    # With perfect play we expect very high (but not necessarily 1.0) average step
    # score because the message-quality proxy gives partial credit for proxies.
    assert state.final_score >= 0.85, (state.final_score, events)
    assert state.done is True
    # At least one event should have been kept-for-clarification (proves reveal cycle worked).
    assert any(e["kept"] for e in events), "Expected at least one clarification reveal"


def test_clarification_re_presents_same_conflict() -> None:
    state, events = _run_episode_with_oracle(seed=42, difficulty="hard")
    # Find a clarification case and verify the same conflict_id appeared back-to-back.
    seen_ids = [e["conflict_id"] for e in events]
    repeats = [(a, b) for a, b in zip(seen_ids, seen_ids[1:]) if a == b]
    assert repeats, f"Expected clarification re-presentation but ids were {seen_ids}"


def test_cascade_appends_followup_conflict() -> None:
    """Hard episodes include at least one cascade rule. With perfect oracle play
    rescheduling to a late slot, a cascade must fire."""

    state, events = _run_episode_with_oracle(seed=42, difficulty="hard")
    cascade_added = [e for e in events if e["cascade_added"]]
    assert cascade_added, "Expected at least one cascade trigger"
    cascade_processed = [e for e in events if e["is_cascade"]]
    assert cascade_processed, "Expected the cascade conflict to be processed"


def test_calendar_mutates_on_propose_plan() -> None:
    """A PROPOSE_PLAN action with a parsed slot should append a calendar event."""

    async def _go():
        env = PersonalAssistantConflictEnv()
        await env.reset()  # static demo task
        before = len(env._state.world.calendar)
        await env.step(
            ConflictAction(
                intent=ActionIntent.PROPOSE_PLAN,
                owner=Owner.WORK,
                priority=Priority.HIGH,
                proposed_slot="14:30",
                message_template="Propose plan for the work review at the new slot.",
                needs_clarification=False,
            )
        )
        after = len(env._state.world.calendar)
        return before, after

    before, after = asyncio.run(_go())
    assert after == before + 1, (before, after)


def test_holdout_pool_disjoint_from_train_pool() -> None:
    from src.assistant_conflict_env.eval_set import (
        HOLDOUT_SEEDS,
        TRAIN_SEEDS,
        ADVERSARIAL_SEEDS,
        assert_split_disjoint,
    )

    assert_split_disjoint()
    assert set(TRAIN_SEEDS).isdisjoint(HOLDOUT_SEEDS)
    assert set(TRAIN_SEEDS).isdisjoint(ADVERSARIAL_SEEDS)
    assert set(HOLDOUT_SEEDS).isdisjoint(ADVERSARIAL_SEEDS)


def test_no_reward_floor_when_action_fully_wrong() -> None:
    """The 0.10 reward floor was removed; fully wrong action should score very low."""

    async def _go():
        env = PersonalAssistantConflictEnv()
        await env.reset()
        # Wrong intent, wrong owner, wrong priority, no slot, empty message.
        res = await env.step(
            ConflictAction(
                intent=ActionIntent.FINALIZE_ITINERARY,
                owner=Owner.LEGAL,
                priority=Priority.LOW,
                proposed_slot="",
                message_template="x",
                needs_clarification=False,
            )
        )
        return res.reward, res.info["reward_components"]

    reward, components = asyncio.run(_go())
    # No floor: the agent should score below the old 0.10 floor on this case.
    assert reward <= 0.10, (reward, components)


def test_clarification_spam_penalized_when_not_warranted() -> None:
    async def _go():
        env = PersonalAssistantConflictEnv()
        # easy_evening_planner first conflict has block_if_missing_context=False
        # in the static fixtures, so asking clarification is unwarranted.
        await env.reset()
        first_case = env._queue[0]
        if first_case.expected.block_if_missing_context:
            pytest.skip("Static fixture starts with a block_if_missing case; can't probe spam here.")
        res = await env.step(
            ConflictAction(
                intent=ActionIntent.ASK_CLARIFICATION,
                owner=first_case.expected.owner,
                priority=first_case.expected.priority,
                proposed_slot="",
                message_template="Asking just because, no real reason here.",
                needs_clarification=True,
            )
        )
        return res.info["penalties"]

    penalties = asyncio.run(_go())
    assert penalties.get("clarification_spam", 0.0) > 0.0, penalties


def test_repetitive_intent_penalty_kicks_in_at_three() -> None:
    async def _go():
        env = PersonalAssistantConflictEnv()
        await env.reset()
        same_action = ConflictAction(
            intent=ActionIntent.ROUTE_MESSAGE,
            owner=Owner.SELF,
            priority=Priority.NORMAL,
            proposed_slot="",
            message_template="Route via fallback channel and follow up with the owner.",
            needs_clarification=False,
        )
        # First two repetitions: no penalty. Third triggers it.
        res1 = await env.step(same_action)
        res2 = await env.step(same_action)
        if res1.done or res2.done:
            pytest.skip("Static task too short to probe 3-step repetition")
        res3 = await env.step(same_action)
        return res1.info["penalties"], res2.info["penalties"], res3.info["penalties"]

    p1, p2, p3 = asyncio.run(_go())
    assert p1.get("repetitive_intent", 0.0) == 0.0, p1
    assert p2.get("repetitive_intent", 0.0) == 0.0, p2
    assert p3.get("repetitive_intent", 0.0) > 0.0, p3


def test_substring_slot_shortcut_no_longer_works() -> None:
    """The slot scorer requires a parseable HH:MM and rewards proximity to the hint.

    Natural-language slot strings (e.g. 'later today', '3pm tomorrow') cannot be
    parsed as 24h HH:MM and must score 0.0 even when require_slot is True.
    """

    from src.assistant_conflict_env.graders import _slot_score

    assert _slot_score("after 20:30", "later today", require_slot=True) == 0.0
    assert _slot_score("after 20:30", "3pm tomorrow", require_slot=True) == 0.0
    assert _slot_score("after 20:30", "reschedule to 20:30", require_slot=True) == 1.0
    assert _slot_score("20:30", "21:00", require_slot=True) == 0.7
    assert _slot_score("20:30", "08:00", require_slot=True) == 0.0


def test_keyword_stuffing_no_longer_dominates_message_score() -> None:
    """A keyword-stuffed stub must score lower than an on-topic real message."""

    from src.assistant_conflict_env.graders import _message_score
    from src.assistant_conflict_env.models import ActionIntent

    stuffed = "reschedule, work, urgent."
    rescheduled_real = "Reschedule the incident review to 20:30 with owner confirmation."

    stuffed_score = _message_score(ActionIntent.RESCHEDULE_EVENT, stuffed)
    real_score = _message_score(ActionIntent.RESCHEDULE_EVENT, rescheduled_real)
    assert real_score > stuffed_score, (stuffed_score, real_score)
    assert real_score >= 0.95, real_score
