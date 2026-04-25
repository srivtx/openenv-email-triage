"""Held-out evaluation episodes for the personal assistant environment.

Honest evaluation requires a held-out pool of episodes that the model has
never seen during SFT or GRPO. We achieve this by partitioning the seed
space into disjoint train and holdout ranges; episodes are deterministic per
seed, so the train/holdout split is reproducible from any commit.

Layout:

- ``TRAIN_SEEDS``:    seeds 1000..1999 (1000 episodes per difficulty pool)
- ``HOLDOUT_SEEDS``:  seeds 9000..9099 (100 episodes per difficulty pool)
- ``ADVERSARIAL_SEEDS``: seeds 5000..5009 (10 episodes used for adversarial
  probes, e.g. ensuring the trained model can't game the substring-slot
  shortcut after the regex-strict grader is in place).

The split is intentionally simple: it has no overlap by construction (the
ranges are disjoint), and increasing/decreasing pool size is a one-line edit.
"""

from __future__ import annotations

from typing import List, Sequence

from .conflict_generator import generate_episode
from .models import TaskDefinition


TRAIN_SEEDS: range = range(1000, 2000)
HOLDOUT_SEEDS: range = range(9000, 9100)
ADVERSARIAL_SEEDS: range = range(5000, 5010)


def _episodes(seeds: Sequence[int], difficulty: str) -> List[TaskDefinition]:
    return [generate_episode(seed=int(seed), difficulty=difficulty) for seed in seeds]


def train_episodes(difficulty: str = "easy", limit: int = 0) -> List[TaskDefinition]:
    """Procedurally generate training episodes for a difficulty.

    ``limit`` of 0 returns all configured seeds; a positive value truncates the
    pool (handy for SFT data generation where 50-200 episodes are typically
    enough).
    """

    seeds = list(TRAIN_SEEDS)
    if limit and limit > 0:
        seeds = seeds[:limit]
    return _episodes(seeds, difficulty)


def holdout_episodes(difficulty: str = "easy", limit: int = 0) -> List[TaskDefinition]:
    """Procedurally generate held-out evaluation episodes."""

    seeds = list(HOLDOUT_SEEDS)
    if limit and limit > 0:
        seeds = seeds[:limit]
    return _episodes(seeds, difficulty)


def adversarial_episodes(difficulty: str = "hard") -> List[TaskDefinition]:
    """Procedurally generate the small adversarial-probe pool."""

    return _episodes(list(ADVERSARIAL_SEEDS), difficulty)


def assert_split_disjoint() -> None:
    """Raise if the train, holdout, or adversarial pools overlap."""

    overlaps = []
    train = set(TRAIN_SEEDS)
    holdout = set(HOLDOUT_SEEDS)
    adv = set(ADVERSARIAL_SEEDS)
    if train & holdout:
        overlaps.append("train/holdout")
    if train & adv:
        overlaps.append("train/adversarial")
    if holdout & adv:
        overlaps.append("holdout/adversarial")
    if overlaps:
        raise AssertionError(f"Seed pools overlap: {', '.join(overlaps)}")
