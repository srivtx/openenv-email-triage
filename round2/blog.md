# teaching a 3B model to plan through a chaotic afternoon — with RL

*OpenEnv Hackathon 2026 | Team Agent (1)*

---

okay so — every AI assistant out there can set a timer or read you the weather. cool. but hand one a real afternoon — board review overlapping school pickup, visa deadline with a missing attachment, insurance payment failing, hotel cancel window closing — and it just... folds. they handle one request at a time. they don't *plan*.

so we built an RL environment that teaches models to do exactly that.

## the env, in one paragraph

`PersonalAssistantConflictEnv` is a multi-step OpenEnv environment with a mutable world state. the agent sees one conflict at a time (plus the visible calendar), emits a structured JSON action — intent, owner, priority, optional time slot, optional clarification flag, message — and the env grades it, mutates state, and decides what to surface next. clarifications take two steps (ask now, get info next step, then act). reschedules can spawn cascade conflicts ("you pushed the call past 18:00 and now nobody can do school pickup"). the model has to plan, not just classify.

## procedural episodes

`conflict_generator.generate_episode(seed, difficulty)` builds episodes from parameterized templates with random variation in times, owners, urgencies, event names. difficulty levels add structure on top:

- **easy**: 3 conflicts, no clarifications, no cascades.
- **medium**: 5 conflicts, ~1 clarification, no cascades.
- **hard**: 7 conflicts, ~2 clarifications, ~1 cascade rule.

pools are disjoint *by construction*:

- **train**: seeds `1000-1999` (1000 episodes) → SFT data + GRPO rollouts.
- **holdout**: seeds `9000-9099` (100 episodes) → generalization eval.
- **adversarial**: seeds `5000-5009` (10 episodes) → probes that the grader does what it claims.

`assert_split_disjoint()` runs at module import. the split is reproducible from any commit.

## actual world state, carried across steps

`WorldState` lives across the whole episode and tracks:

- `calendar`: events with start/end/owner/locked. **mutated** when the agent reschedules or proposes a plan.
- `pending_clarifications`: queued info reveals waiting to come back.
- `revealed_info`: `conflict_id → revealed text`.
- `cascade_queue`: follow-on conflicts the agent's own actions have spawned.

the conflict queue is dynamic, not a fixed array index. conflicts can be re-presented (after a clarification reveal), and new conflicts can be appended mid-episode (when a cascade rule fires).

## two-step partial observability

when a conflict has a `ClarificationSpec` and the agent picks `ask_clarification`:

1. **step N**: env scores the ask, records the revealed info on the case, and **keeps the same conflict at the front of the queue.**
2. **step N+1**: env presents the same conflict back, with `revealed_info` attached to the summary, and grades the *post-reveal action* (the actual decision).
3. **step N+2**: queue advances to the next conflict.

picking `ask_clarification` when no clarification is warranted triggers a `clarification_spam` penalty. the "ask everything to be safe" path is closed.

## real cascades

every procedural template can carry a `CascadeRule`. hard-difficulty episodes get a reschedule case where pushing the slot past 18:00 with `owner=work` spawns a follow-on "school pickup uncovered" conflict. the new conflict goes onto the queue and gets resolved like any other.

translation: **the agent's own decision generates the next problem.** that's the whole point of long-horizon. without that, it's just batched classification.

unit tests assert a cascade fires under perfect-play oracles for `seed=42` (`test_cascade_appends_followup_conflict`). it's a tested invariant, not a hopeful claim.

## the grader

- **slot**: regex-strict 24h `HH:MM` parsing. time-distance scoring (exact = 1.0, ≤30 min off = 0.7, ≤60 min = 0.4, ≤120 min = 0.2, else 0.0). substring shortcuts return 0.0.
- **message**: length + on-topic verb + word-diversity. keyword stuffing scores ~0.5; a real on-topic sentence scores 1.0.
- **weights**: `intent 0.40 / owner 0.20 / slot 0.20 / priority 0.10 / clarification 0.05 / message 0.05`. derivation documented in the module docstring.
- reward band is `[0.0, 1.0]`. no floor.

## anti-hacking penalties

| penalty               | size   | when it fires                                                  |
|-----------------------|--------|-----------------------------------------------------------------|
| `repetitive_intent`   | −0.10  | same intent 3+ steps in a row                                  |
| `premature_finalize`  | −0.15  | `finalize_itinerary` with multiple conflicts still pending      |
| `terminal_early`      | −0.05  | any other terminal-style intent used to short-circuit           |
| `missing_slot`        | −0.05  | `require_slot=True` but no slot supplied                        |
| `clarification_spam`  | −0.05  | asking when the case has no clarification spec or info revealed |
| `short_message`       | −0.04  | message under 16 chars                                          |

## inference path

`inference.py` defaults to **the trained** Qwen 2.5 3B Instruct + LoRA adapter via `transformers` and `peft` when `MODEL_PATH` is set. an HF Router 72B path is available as a baseline. `EVAL_POOL=holdout` runs the holdout pool. `EVAL_POOL=adversarial` runs the probe pool.

if neither token nor model path is set, it falls back to a heuristic baseline so the script still does something visible.

## how we trained it

**model**: Qwen 2.5 3B Instruct (4-bit quantized via Unsloth — fits a free Colab T4).

**approach**: SFT → GRPO. same recipe as ChatGPT, just smaller scale and on a weirder task.

1. **SFT** on procedurally generated train episodes. every example is unique. each clarification case generates *two* SFT examples — the initial ask, and the post-reveal action — so the model learns to operate in the partial-obs regime.
2. **GRPO** on top, using **the env's actual reward** as the GRPO reward. the reward function in the notebook replays the env up to each step, applies the sampled completion, and reads `result.reward`. no placeholder rewards.

LoRA only — `r=16`, ~1% of params trained. the rest stays frozen.

**why GRPO?** no critic network needed (saves VRAM), works natively with TRL, and the "generate multiple completions → score them all against the real env → reinforce the best ones" loop maps perfectly to a deterministic grader.

## results

| pool                | untrained 3B | after SFT | after SFT + GRPO |
|---------------------|--------------|-----------|------------------|
| holdout (n=40)      | **0.5454**   | **0.9877**| **0.9876**       |
| adversarial (n=10)  | —            | —         | **0.9885**       |

what these numbers actually say:

- **+0.4423** absolute lift from untrained → SFT on holdout. the model learns the format and intent routing on episodes it has never seen.
- **GRPO ≈ SFT** here (0.9876 vs 0.9877). on hard procedural data the SFT stage is doing most of the lifting; GRPO holds the score steady. honest finding, reported as-is.
- **0.9885 on adversarial.** the model holds up on probe episodes specifically designed to defeat the grader's anti-shortcut behavior.

### grader sanity probes (from the same run)

deterministic checks on `_slot_score` and `_message_score`:

```
slot-score probe (shortcuts -> 0.0; honest matches -> 1.0):
  hint='after 20:30'   slot='later today'              -> 0.00
  hint='after 20:30'   slot='3pm tomorrow'             -> 0.00
  hint='20:30'         slot='reschedule to 20:30'      -> 1.00
  hint='20:30'         slot='21:00'                    -> 0.70   (30 min off, partial credit)
  hint='20:30'         slot='08:00'                    -> 0.00

message-score probe:
  STUFFED : 0.50   ('reschedule, work, urgent.')
  REAL    : 1.00   ('Reschedule the incident review to 20:30 with owner confirmation and follow up note.')
```

substring slot shortcuts return `0.0`. an honest `HH:MM` match returns `1.0` with proportional partial credit when the time is off. an honest message scores `1.0`; a keyword-stuffed one scores `0.5`. the grader does what it says.

## what stayed the same (so we didn't break the API)

- the action schema (6 intents, 6 owners, 4 priorities, slot, needs_clarification, message). backward-compatible for any existing client.
- the OpenEnv `reset` / `step` / `state` interface and the FastAPI server.
- the Docker + HF Space deployment pipeline.
- a small set of static fixture tasks for the live UI demo. they're kept around so the deployed Space still has interactive episodes for visitors. they are **not** used for training or evaluation.

## what we wish we'd had time for

- **curriculum**: harder episodes for later in training. we have the difficulty knob, we just didn't sweep it.
- **a proper user study or domain-expert pass on the reward weights.** the current weights are documented and auditable, but they're still our best judgment.
- **a bigger procedural template pool.** ~6 templates plus a finalize template; more variety would help generalization.
- **a learned reward model.** right now it's deterministic. a learned one would let us scale to fuzzier judgment calls.

## lessons we'd hammer into a future-self

- if you can't articulate what your environment's *world state* is, it isn't long-horizon. step counters and reward histories don't count.
- a static fixture is a dataset. the moment you train on it AND evaluate on it, you've built a leaderboard for your own homework.
- reward floors hide bugs. don't have them.
- "it works on three hand-written examples" doesn't generalize. fixing it to "it works on 100 procedural examples" is a different *kind* of works.

---

## links

- **live environment**: [HuggingFace Space](https://huggingface.co/spaces/srivtx/openenv-conflict-resolver-v2)
- **training notebook**: `notebooks/train_grpo_colab.ipynb`
- **full docs**: see `docs/` (9-chapter walkthrough)

*built with Unsloth and TRL.*
