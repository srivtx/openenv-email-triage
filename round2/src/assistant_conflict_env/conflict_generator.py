"""Procedural conflict generator for the personal assistant environment.

This module replaces the static fixture-based task definitions for training and
evaluation. Each call to ``generate_episode(seed, difficulty)`` produces a
deterministic ``TaskDefinition`` whose conflicts are drawn from parameterized
templates with variation in times, owners, urgencies, and target events.

Difficulty levels add structure on top of the base templates:

- **easy**:   3 simple conflicts, no clarifications, no cascades
- **medium**: 5 conflicts, ~1 clarification, no cascades
- **hard**:   7 conflicts, ~2 clarifications, ~1 cascade rule

Templates can declare:

- ``clarification``: information hidden from the agent until ``ask_clarification``
  is taken; once revealed, a different post-reveal expected action applies.
- ``cascade_rule``: an action condition that generates a follow-on conflict on
  the fly (e.g. rescheduling past 18:00 spawns a school pickup conflict).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from .models import (
    ActionIntent,
    CalendarEvent,
    CascadeRule,
    ClarificationSpec,
    ConflictCase,
    ConflictExpectation,
    Owner,
    Priority,
    TaskDefinition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def time_to_minutes(time_str: str) -> int:
    """Convert HH:MM string to minutes from midnight. Returns -1 for invalid."""

    if not time_str or ":" not in time_str:
        return -1
    try:
        hh, mm = time_str.strip().split(":", 1)
        return int(hh) * 60 + int(mm)
    except (ValueError, IndexError):
        return -1


def minutes_to_time(minutes: int) -> str:
    """Convert minutes from midnight to HH:MM string."""

    minutes = max(0, min(24 * 60 - 1, minutes))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


# ---------------------------------------------------------------------------
# Template data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemplateContext:
    """Random parameters drawn for one template instantiation."""

    rng: random.Random
    conflict_id: str
    primary_time: str
    secondary_time: str
    primary_minutes: int
    secondary_minutes: int


def _draw_time(rng: random.Random, low: int = 9 * 60, high: int = 22 * 60) -> int:
    """Draw a random minute-of-day rounded to 15-minute increments."""

    minutes = rng.randint(low // 15, high // 15) * 15
    return minutes


def _make_context(rng: random.Random, conflict_id: str) -> TemplateContext:
    primary = _draw_time(rng)
    secondary = primary + rng.choice([30, 45, 60, 90])
    return TemplateContext(
        rng=rng,
        conflict_id=conflict_id,
        primary_time=minutes_to_time(primary),
        secondary_time=minutes_to_time(secondary),
        primary_minutes=primary,
        secondary_minutes=secondary,
    )


# Each template function returns a ConflictCase given a TemplateContext.
TemplateFn = Callable[[TemplateContext], ConflictCase]


# ---------------------------------------------------------------------------
# Reschedule templates (with optional cascade)
# ---------------------------------------------------------------------------


def _tpl_reschedule_overlap(ctx: TemplateContext) -> ConflictCase:
    """Two work events overlap; reschedule the lower-priority one."""

    rng = ctx.rng
    event_a, event_b = rng.sample(
        ["board review", "incident review", "1:1 with VP", "design critique", "OKR sync"],
        k=2,
    )
    priority = rng.choice([Priority.HIGH, Priority.URGENT])
    after_minutes = ctx.secondary_minutes + 30
    after_str = minutes_to_time(after_minutes)
    return ConflictCase(
        conflict_id=ctx.conflict_id,
        source="calendar",
        summary=(
            f"{event_a.title()} at {ctx.primary_time} overlaps {event_b} at {ctx.secondary_time}. "
            f"{event_a.title()} is mandatory."
        ),
        constraints=[
            f"{event_a} cannot move",
            f"{event_b} can shift later in the day",
        ],
        expected=ConflictExpectation(
            intent=ActionIntent.RESCHEDULE_EVENT,
            owner=Owner.WORK,
            priority=priority,
            require_slot=True,
            expected_slot_hint=after_str,
            block_if_missing_context=False,
            required_keywords=["reschedule"],
        ),
    )


def _tpl_reschedule_with_cascade(ctx: TemplateContext) -> ConflictCase:
    """Reschedule a work meeting; if pushed past 18:00 it spawns a pickup conflict."""

    rng = ctx.rng
    event = rng.choice(["client review", "investor call", "product demo", "exec briefing"])
    pickup_time = minutes_to_time(rng.choice([18 * 60, 18 * 60 + 30, 19 * 60]))

    cascade = CascadeRule(
        trigger_intent=ActionIntent.RESCHEDULE_EVENT,
        trigger_slot_after_minutes=18 * 60,
        trigger_owner=Owner.WORK,
        follow_on_id_prefix="C",
        follow_on_source="cascade-pickup",
        follow_on_summary=(
            f"Reschedule pushed work past 18:00; school pickup at {pickup_time} now uncovered."
        ),
        follow_on_constraints=[
            "child must be picked up",
            "user is still on the rescheduled call",
        ],
        follow_on_expected=ConflictExpectation(
            intent=ActionIntent.DELEGATE_TASK,
            owner=Owner.FAMILY,
            priority=Priority.URGENT,
            required_keywords=["delegate", "pickup"],
        ),
    )

    # Cascade designs target a post-18:00 slot so that the perfect-play oracle
    # reliably triggers the cascade. Suboptimal play (e.g. earlier slots) will
    # NOT trigger the cascade, which is the point of the rule.
    after_minutes = rng.choice([18 * 60 + 30, 19 * 60, 19 * 60 + 30, 20 * 60])
    after_str = minutes_to_time(after_minutes)
    return ConflictCase(
        conflict_id=ctx.conflict_id,
        source="calendar",
        summary=(
            f"{event.title()} at {ctx.primary_time} runs long and conflicts with later "
            f"commitments. Find a slot after 18:00 that fits."
        ),
        constraints=[
            f"{event} cannot be skipped",
            "any new slot must be after 18:00 to clear the existing conflicts",
        ],
        expected=ConflictExpectation(
            intent=ActionIntent.RESCHEDULE_EVENT,
            owner=Owner.WORK,
            priority=Priority.HIGH,
            require_slot=True,
            expected_slot_hint=after_str,
            block_if_missing_context=False,
            required_keywords=["reschedule"],
        ),
        cascade_rule=cascade,
    )


# ---------------------------------------------------------------------------
# Delegate templates
# ---------------------------------------------------------------------------


def _tpl_delegate_pickup(ctx: TemplateContext) -> ConflictCase:
    rng = ctx.rng
    item = rng.choice(["medication", "gift", "groceries", "dry cleaning", "package"])
    return ConflictCase(
        conflict_id=ctx.conflict_id,
        source="todo",
        summary=(
            f"{item.title()} pickup is due at {ctx.primary_time} but the user is in an "
            f"all-day workshop and cannot break away."
        ),
        constraints=[
            "store closes at 20:00",
            "must confirm completion",
        ],
        expected=ConflictExpectation(
            intent=ActionIntent.DELEGATE_TASK,
            owner=Owner.FAMILY,
            priority=Priority.NORMAL,
            required_keywords=["delegate", item],
        ),
    )


def _tpl_delegate_finance_route(ctx: TemplateContext) -> ConflictCase:
    rng = ctx.rng
    bill = rng.choice(["insurance", "utilities", "subscription"])
    return ConflictCase(
        conflict_id=ctx.conflict_id,
        source="finance-alert",
        summary=(
            f"{bill.title()} payment failed; grace period ends tonight at {ctx.primary_time}."
        ),
        constraints=[
            "payment must be completed today",
            "late fee applies after midnight",
        ],
        expected=ConflictExpectation(
            intent=ActionIntent.ROUTE_MESSAGE,
            owner=Owner.FINANCE,
            priority=Priority.URGENT,
            required_keywords=["payment", bill],
        ),
    )


# ---------------------------------------------------------------------------
# Clarification templates (true 2-step partial observability)
# ---------------------------------------------------------------------------


def _tpl_clarify_then_propose(ctx: TemplateContext) -> ConflictCase:
    """Demo request missing timezone. Ask first; then propose plan with revealed info."""

    rng = ctx.rng
    party = rng.choice(["client", "investor", "partner", "vendor"])
    revealed_tz = rng.choice(["EST", "PST", "GMT", "JST"])
    revealed_slot_hint = ctx.primary_time

    clarification = ClarificationSpec(
        hidden_info=f"timezone is {revealed_tz}",
        revealed_summary_suffix=f" Revealed info: timezone is {revealed_tz}; agreed slot is {revealed_slot_hint}.",
        post_reveal_expected=ConflictExpectation(
            intent=ActionIntent.PROPOSE_PLAN,
            owner=Owner.WORK,
            priority=Priority.HIGH,
            require_slot=True,
            expected_slot_hint=revealed_slot_hint,
            block_if_missing_context=False,
            required_keywords=["proposal"],
        ),
    )

    return ConflictCase(
        conflict_id=ctx.conflict_id,
        source="email",
        summary=(
            f"{party.title()} requests a demo around {ctx.primary_time} but timezone is missing. "
            f"Cannot confirm without it."
        ),
        constraints=[
            "cannot confirm demo without timezone",
            "must avoid no-show risk",
        ],
        expected=ConflictExpectation(
            intent=ActionIntent.ASK_CLARIFICATION,
            owner=Owner.WORK,
            priority=Priority.HIGH,
            block_if_missing_context=True,
            required_keywords=["clarify", "timezone"],
        ),
        clarification=clarification,
    )


def _tpl_clarify_legal_deadline(ctx: TemplateContext) -> ConflictCase:
    """Legal email with missing deadline. Clarify; then propose plan to act before deadline."""

    rng = ctx.rng
    deadline = minutes_to_time(rng.randint(18, 22) * 60)

    clarification = ClarificationSpec(
        hidden_info=f"actual deadline is {deadline}",
        revealed_summary_suffix=f" Revealed info: hard deadline is {deadline} today.",
        post_reveal_expected=ConflictExpectation(
            intent=ActionIntent.PROPOSE_PLAN,
            owner=Owner.LEGAL,
            priority=Priority.URGENT,
            require_slot=True,
            expected_slot_hint=minutes_to_time(time_to_minutes(deadline) - 60),
            block_if_missing_context=False,
            required_keywords=["proposal", "deadline"],
        ),
    )

    return ConflictCase(
        conflict_id=ctx.conflict_id,
        source="legal-email",
        summary=(
            "Visa document request has missing attachment and ambiguous deadline. "
            "Cannot proceed without clarification."
        ),
        constraints=[
            "cannot submit without correct form",
            "late submission has legal penalty",
        ],
        expected=ConflictExpectation(
            intent=ActionIntent.ASK_CLARIFICATION,
            owner=Owner.LEGAL,
            priority=Priority.URGENT,
            block_if_missing_context=True,
            required_keywords=["clarify", "deadline"],
        ),
        clarification=clarification,
    )


# ---------------------------------------------------------------------------
# Finalize templates
# ---------------------------------------------------------------------------


def _tpl_finalize(ctx: TemplateContext) -> ConflictCase:
    return ConflictCase(
        conflict_id=ctx.conflict_id,
        source="planner",
        summary="User requests final consolidated itinerary covering all open commitments.",
        constraints=[
            "include owner assignments",
            "list high-risk items first",
        ],
        expected=ConflictExpectation(
            intent=ActionIntent.FINALIZE_ITINERARY,
            owner=Owner.SELF,
            priority=Priority.HIGH,
            required_keywords=["timeline", "risk"],
        ),
    )


# ---------------------------------------------------------------------------
# Template pools by difficulty
# ---------------------------------------------------------------------------


_BASE_TEMPLATES: Sequence[TemplateFn] = (
    _tpl_reschedule_overlap,
    _tpl_delegate_pickup,
    _tpl_delegate_finance_route,
)

_CLARIFICATION_TEMPLATES: Sequence[TemplateFn] = (
    _tpl_clarify_then_propose,
    _tpl_clarify_legal_deadline,
)

_CASCADE_TEMPLATES: Sequence[TemplateFn] = (
    _tpl_reschedule_with_cascade,
)


_DIFFICULTY_PLAN: Dict[str, Dict[str, int]] = {
    "easy": {"base": 2, "clarification": 0, "cascade": 0, "finalize": 1},
    "medium": {"base": 3, "clarification": 1, "cascade": 0, "finalize": 1},
    "hard":  {"base": 3, "clarification": 2, "cascade": 1, "finalize": 1},
}


_MAX_STEPS_PER_DIFFICULTY: Dict[str, int] = {
    "easy": 6,
    "medium": 12,
    "hard": 18,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_episode(
    seed: int,
    difficulty: str = "easy",
    initial_calendar: Optional[List[CalendarEvent]] = None,
) -> TaskDefinition:
    """Build a deterministic TaskDefinition for the given (seed, difficulty)."""

    if difficulty not in _DIFFICULTY_PLAN:
        raise ValueError(f"Unknown difficulty '{difficulty}'. Choices: {list(_DIFFICULTY_PLAN)}")

    rng = random.Random(seed)
    plan = _DIFFICULTY_PLAN[difficulty]
    pool: List[TemplateFn] = []
    for _ in range(plan["base"]):
        pool.append(rng.choice(_BASE_TEMPLATES))
    for _ in range(plan["clarification"]):
        pool.append(rng.choice(_CLARIFICATION_TEMPLATES))
    for _ in range(plan["cascade"]):
        pool.append(rng.choice(_CASCADE_TEMPLATES))
    for _ in range(plan["finalize"]):
        pool.append(_tpl_finalize)

    finalize_templates = [t for t in pool if t is _tpl_finalize]
    non_finalize = [t for t in pool if t is not _tpl_finalize]
    rng.shuffle(non_finalize)
    ordered = non_finalize + finalize_templates

    conflicts: List[ConflictCase] = []
    for index, template in enumerate(ordered):
        conflict_id = f"P-{seed:05d}-{index + 1:02d}"
        ctx = _make_context(rng, conflict_id)
        case = template(ctx)
        conflicts.append(case)

    calendar = initial_calendar if initial_calendar is not None else initial_calendar_for(seed)

    return TaskDefinition(
        id=f"proc_{difficulty}_{seed}",
        title=f"Procedural {difficulty.title()} Episode (seed={seed})",
        difficulty=difficulty,
        description=(
            f"Procedurally generated {difficulty} episode with {len(conflicts)} initial conflicts."
        ),
        max_steps=_MAX_STEPS_PER_DIFFICULTY[difficulty],
        conflicts=conflicts,
        initial_calendar=calendar,
    )


def initial_calendar_for(seed: int) -> List[CalendarEvent]:
    """Return a small starter calendar so the env has world state to mutate."""

    rng = random.Random(seed * 17 + 3)
    events = [
        CalendarEvent(
            event_id=f"E-{seed:05d}-board-review",
            title="board review",
            start=minutes_to_time(_draw_time(rng, 16 * 60, 17 * 60)),
            end=minutes_to_time(_draw_time(rng, 17 * 60 + 30, 18 * 60 + 30)),
            owner=Owner.WORK,
            locked=True,
        ),
        CalendarEvent(
            event_id=f"E-{seed:05d}-school-pickup",
            title="school pickup",
            start="18:00",
            end="18:30",
            owner=Owner.FAMILY,
            locked=False,
        ),
        CalendarEvent(
            event_id=f"E-{seed:05d}-family-dinner",
            title="family dinner",
            start="20:00",
            end="21:30",
            owner=Owner.FAMILY,
            locked=False,
        ),
    ]
    return events


def generate_episode_pool(
    difficulty: str,
    seed_range: Sequence[int],
) -> List[TaskDefinition]:
    """Generate a list of episodes from a seed range. Useful for SFT / eval pools."""

    return [generate_episode(seed=seed, difficulty=difficulty) for seed in seed_range]
