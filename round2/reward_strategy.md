# Round 2 Reward Strategy (from reward_eng.md)

This file translates the reward engineering context into concrete implementation choices for this environment.

## 1) Reward engineering (base objective)

Primary objective: resolve personal/work conflicts correctly with minimal risk.

Base per-step verifiers:

- intent correctness
- owner correctness
- priority correctness
- slot compliance
- clarification correctness
- message keyword quality

## 2) Reward shaping (sample-efficiency layer)

The environment applies shaping while keeping the true objective intact:

- small positive baseline (`0.10 + 0.90 * step_score`)
- penalties for low-information responses
- penalties for repetitive intent loops
- penalties for premature finalization
- penalties for missing required slot detail

This gives dense signals on easy tasks while preserving clear end-goal behavior.

## 3) Multi-signal design to reduce reward hacking

Independent checks reduce exploitability:

1. correctness signals (intent/owner/priority)
2. process signals (clarification, slot validity)
3. quality signal (keyword coverage)
4. anti-exploit penalties (loop/finalize/slot misuse)

## 4) Curriculum and non-zero reward guarantee

- easy tasks: clear ownership and obvious routing
- medium: missing context and cross-party constraints
- hard: cascading disruptions and risk tradeoffs

This follows the hackathon guidance: make early success possible before scaling difficulty.

## 5) Monitoring recommendations

During training, monitor all columns, not just total reward:

- total reward
- intent/owner/priority components
- slot/clarification components
- penalties by type
- success threshold pass-rate

Inspect sampled trajectories every run for reward-hacking drift.
