# Teaching a 3B Model to Plan Through a Chaotic Afternoon — With RL

*OpenEnv Hackathon 2026 | Team Agent (1)*

---

Every AI assistant out there can set a timer or read you the weather. Cool. But hand one a real afternoon — board review overlapping school pickup, visa deadline with a missing attachment, insurance payment failing, hotel cancel window closing — and it falls apart. They handle one request at a time. They don't *plan*.

So we built an RL environment that teaches models to do exactly that. And then we got a brutal external code review that pointed out our v0.2 was, in substance, a deployed dataset with a scoring function — not an RL environment. The review was right. So we rebuilt it.

This is the story of v0.3, which is the version actually worth reading about.

## What was wrong with v0.2

A short version of the (correct) critique:

- **Same conflicts for SFT and eval.** 15 static fixtures. Train on them, eval on them, declare victory at 1.0. That's memorization, not learning.
- **"Long-horizon" was independent classification.** Each step was decoupled. State didn't carry over. Decisions had no downstream effect.
- **"Partial observability" was pre-labeled.** The "right" answer for missing-info conflicts was already `ask_clarification` — the model just had to detect a keyword.
- **"Cascading" never cascaded.** The hard task had three "cascade" conflicts that didn't actually depend on each other.
- **The grader had escape hatches.** Substring slot scoring meant `"3pm tomorrow"` got 0.6 even if the expected slot was `20:30`. Keyword-stuffing in the message field got full credit. A 0.10 reward floor meant the model never saw a 0 even on a fully wrong action.
- **The notebook had a duplicate cell, `sft_data * 15` over 15 unique examples, and the inference script defaulted to Qwen 72B over an HF Router endpoint instead of the 3B model we said we trained.**

Fair. All true.

## What v0.3 actually does

### Procedural episodes with disjoint train/holdout/adversarial pools

`conflict_generator.generate_episode(seed, difficulty)` builds a `TaskDefinition` from parameterized templates with random variation in times, owners, urgencies, and event names. Difficulty levels add structure on top:

- **easy**: 3 conflicts, no clarifications, no cascades.
- **medium**: 5 conflicts, ~1 clarification, no cascades.
- **hard**: 7 conflicts, ~2 clarifications, ~1 cascade rule.

Pools are disjoint by construction:

- **train**: seeds 1000-1999 (1000 episodes) — SFT data + GRPO rollouts.
- **holdout**: seeds 9000-9099 (100 episodes) — honest generalization eval.
- **adversarial**: seeds 5000-5009 (10 episodes) — verify shortcut shutdowns.

`assert_split_disjoint()` runs at module import. The split is reproducible from any commit.

### Real world state across steps

`WorldState` carries:

- `calendar`: events with start/end/owner/locked. Mutated when the agent reschedules or proposes a plan.
- `pending_clarifications`: queued info reveals.
- `revealed_info`: `conflict_id -> revealed text`.
- `cascade_queue`: follow-on conflicts the agent has spawned.

The conflict queue is dynamic, not an array index. Conflicts can be re-presented (clarification reveal), and new conflicts can be appended mid-episode (cascade trigger).

### Two-step partial observability

When a conflict has a `ClarificationSpec` and the agent picks `ask_clarification`:

1. **Step N**: env scores the ask, records the revealed info on the case, and **keeps the same conflict at the front of the queue**.
2. **Step N+1**: env presents the same conflict, with `revealed_info` attached to the summary, and grades against the `post_reveal_expected` action.
3. **Step N+2**: env advances to the next conflict.

Picking `ask_clarification` when no clarification is warranted now triggers a `clarification_spam` penalty. The "ask everything to be safe" shortcut is closed.

### Real cascades

Each procedural template can carry a `CascadeRule`. Hard-difficulty episodes get a reschedule case where pushing the slot past 18:00 with `owner=work` spawns a follow-on "school pickup uncovered" conflict. The new conflict goes onto the queue and gets resolved like any other. The agent's own decision is what generates the downstream work.

The unit tests assert a cascade fires under perfect-play oracles for `seed=42` (`test_cascade_appends_followup_conflict`).

### Rebuilt grader

- `slot`: regex-strict 24h `HH:MM` parsing. Time-distance scoring (exact = 1.0, ≤30 min = 0.7, ≤60 min = 0.4, ≤120 min = 0.2, else 0.0). The substring shortcut is dead — `_slot_score("after 20:30", "later today", require_slot=True)` returns 0.0.
- `message`: length + on-topic verb + word-diversity. Keyword stuffing (`"reschedule, work, urgent."`) scores 0.5 instead of 1.0; a real on-topic sentence scores 1.0.
- Weights: `intent 0.40 / owner 0.20 / slot 0.20 / priority 0.10 / clarification 0.05 / message 0.05`. Documented derivation lives in the module docstring.
- The 0.10 reward floor is gone. A fully wrong action scores 0.

### Strengthened penalties

- `repetitive_intent` triggers on 3 in a row (was 2; the old check was bypassable by alternating).
- `premature_finalize` (`finalize_itinerary` with > 1 case left) jumps from 0.08 to 0.15.
- `terminal_early` (other terminal-ish intents used to short-circuit) is a new −0.05.
- `clarification_spam` is now aware of `pending_clarifications` and `revealed_info`.

### Inference path

`inference.py` was rewritten to default to the **trained** Qwen 2.5 3B Instruct + LoRA adapter via `transformers` and `peft` when `MODEL_PATH` is set. The HF Router 72B path is still available for baselines but is no longer the default. `EVAL_POOL=holdout` runs honest holdout eval; `EVAL_POOL=adversarial` runs the probe pool.

## Reward design (rebuilt, honestly)

| signal           | weight | what it checks                                                                  |
|------------------|--------|----------------------------------------------------------------------------------|
| intent           | 0.40   | exact match — most consequential decision                                       |
| owner            | 0.20   | exact match — who handles it                                                    |
| slot             | 0.20   | regex-strict 24h `HH:MM`, time-distance scoring                                 |
| priority         | 0.10   | partial credit (0.5) for off-by-one — adjacent priorities are defensible        |
| clarification    | 0.05   | boolean alignment with `block_if_missing_context`                               |
| message          | 0.05   | length + on-topic verb + word-diversity                                         |

## How we trained it

**Model**: Qwen 2.5 3B Instruct (4-bit quantized via Unsloth — fits a free Colab T4).

**Approach**: SFT then GRPO.

1. SFT on **procedurally generated** train episodes (no `* 15` duplication, no train/eval overlap). Each example uses a varied on-topic message instead of the keyword list verbatim. Clarification cases generate two SFT examples — the initial ask and the post-reveal action — so the model learns to operate in the partial-observability regime.
2. GRPO on top, using the **env's actual reward** as the GRPO reward. The reward function in the notebook replays the env up to each step, applies the sampled completion, and reads `result.reward`. No placeholder rewards.

LoRA only — `r=16`, ~1% of params trained. The rest stays frozen.

## Results

The notebook prints holdout averages, adversarial averages, and a probe of `_slot_score` and `_message_score` on canonical "old shortcut" inputs.

Numbers will be filled in after rerunning the notebook on Colab T4. We are not reporting v0.2's `1.0 / 1.0 / 1.0` because those numbers came from a memorization regime that no longer exists.

| pool            | untrained 3B | after SFT | after SFT + GRPO |
|-----------------|--------------|-----------|------------------|
| holdout (n=100) | TBD          | TBD       | TBD              |
| adversarial (n=10) | TBD       | TBD       | TBD              |

We expect holdout numbers to be lower than v0.2's `1.0` — that's the point. They will reflect actual generalization to unseen procedural episodes.

## What stayed the same

- The action schema (6 intents, 6 owners, 4 priorities, slot, needs_clarification, message). Backward-compatible for any client that already integrates with the v0.2 API.
- The OpenEnv `reset` / `step` / `state` interface and the FastAPI server.
- The Docker + HF Space deployment pipeline.
- The static fixture tasks for the live UI demo. They're kept around so the deployed Space still has interactive episodes for visitors. They are **not** used for training or evaluation.

## What we wish we'd had time for

- Curriculum: harder episodes for later in training (we have the difficulty knob; we just didn't sweep it).
- A proper user study or domain-expert pass on the reward weights. The current weights are documented and auditable, but they're still our best judgment, not validated.
- A bigger procedural template pool. We have ~6 templates plus a finalize template; more variety would help generalization.

## Lessons we'd hammer into a future-self

- If you can't articulate what your environment's *world state* is, it isn't long-horizon. Bookkeeping isn't state.
- A static fixture is a dataset. The moment you train on it and evaluate on it, you've built a leaderboard for your own homework.
- Reward floors hide bugs. Ours did.
- "It works" on three hand-written examples does not generalize. Even fixing it to "it works on 100 procedural examples" is a different kind of "works".

---

## Links

- **Live environment**: [HuggingFace Space](https://huggingface.co/spaces/srivtx/openenv-conflict-resolver-v2)
- **Training notebook**: `notebooks/train_grpo_colab.ipynb`
- **Full docs**: see `docs/` (9 chapters from RL basics through the rebuild)

*Built with Unsloth, TRL, and a willingness to delete our own dishonest numbers.*
