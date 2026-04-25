# teaching a 3B model to plan through a chaotic afternoon — with RL

*OpenEnv Hackathon 2026 | Team Agent (1)*

---

okay so — every AI assistant out there can set a timer or read you the weather. cool. but hand one a real afternoon — board review overlapping school pickup, visa deadline with a missing attachment, insurance payment failing, hotel cancel window closing — and it just... folds. they handle one request at a time. they don't *plan*.

so we built an RL environment that teaches models to do exactly that.

then a brutal external code review came in and basically said: "your v0.2 is a deployed dataset with a scoring function — not an RL environment." and they were right. so we tore it down and rebuilt it. this is the story of v0.3, which is the version actually worth reading about.

## what was wrong with v0.2 (the real talk)

short list of stuff we got called out on, and we earned every single one:

- **same conflicts for SFT and eval.** 15 static fixtures. train on them. eval on them. declare victory at 1.0. that's not learning, that's memorization with extra steps.
- **"long-horizon" was actually independent classification.** every step was decoupled. state didn't carry over. no decision had downstream effects. literally just `argmax` on each item.
- **"partial observability" was pre-labeled.** the right answer for a missing-info case was already `ask_clarification` in the JSON. the model just had to keyword-match.
- **"cascading" never cascaded.** the hard task had three "cascade" conflicts that didn't depend on each other at all. it was just three independent items in a list.
- **the grader had escape hatches.** substring slot scoring meant `"3pm tomorrow"` got 0.6 even when the expected answer was `20:30`. keyword stuffing in the message field got full marks. there was a 0.10 reward floor — meaning the model never saw a 0 even if it got everything wrong.
- **the notebook had a duplicate cell, `sft_data * 15` over 15 unique examples, and the inference script defaulted to Qwen 72B over an HF Router endpoint.** so the "1.0" we were reporting was either pure memorization or it was running on a model 24x bigger than the one we said we trained.

fair. all true. extremely valid. we deleted those numbers and started over.

## what v0.3 actually does

### procedural episodes with disjoint train / holdout / adversarial pools

`conflict_generator.generate_episode(seed, difficulty)` builds a `TaskDefinition` from parameterized templates with random variation in times, owners, urgencies, and event names. difficulty levels add structure on top:

- **easy**: 3 conflicts, no clarifications, no cascades.
- **medium**: 5 conflicts, ~1 clarification, no cascades.
- **hard**: 7 conflicts, ~2 clarifications, ~1 cascade rule.

pools are disjoint *by construction*:

- **train**: seeds `1000-1999` (1000 episodes) → SFT data + GRPO rollouts.
- **holdout**: seeds `9000-9099` (100 episodes) → honest generalization eval.
- **adversarial**: seeds `5000-5009` (10 episodes) → verify the old shortcuts are dead.

`assert_split_disjoint()` runs at module import. so the split is reproducible from any commit. no overlap. no fudging.

### actual world state, carried across steps

`WorldState` lives across the whole episode and tracks:

- `calendar`: events with start/end/owner/locked. **mutated** when the agent reschedules or proposes a plan.
- `pending_clarifications`: queued info reveals waiting to come back.
- `revealed_info`: `conflict_id → revealed text`.
- `cascade_queue`: follow-on conflicts the agent's own actions have spawned.

the conflict queue is dynamic now, not a fixed array index. conflicts can be re-presented (after a clarification reveal), and new conflicts can be appended mid-episode (when a cascade rule fires).

### two-step partial observability (for real this time)

when a conflict has a `ClarificationSpec` and the agent picks `ask_clarification`:

1. **step N**: env scores the ask, records the revealed info on the case, and **keeps the same conflict at the front of the queue.**
2. **step N+1**: env presents the same conflict back, with `revealed_info` attached to the summary, and grades the *post-reveal action* (which is the actual decision).
3. **step N+2**: queue advances to the next conflict.

picking `ask_clarification` when no clarification is warranted now triggers a `clarification_spam` penalty. the "ask everything to be safe" shortcut is closed.

### real cascades

every procedural template can carry a `CascadeRule`. hard-difficulty episodes get a reschedule case where pushing the slot past 18:00 with `owner=work` spawns a follow-on "school pickup uncovered" conflict. the new conflict goes onto the queue and gets resolved like any other.

translation: **the agent's own decision generates the next problem.** that's the whole point of long-horizon. without that, it's just batched classification.

the unit tests assert a cascade fires under perfect-play oracles for `seed=42` (`test_cascade_appends_followup_conflict`). so this isn't a hopeful claim — it's a tested invariant.

### rebuilt grader (no more cheap tricks)

- **slot**: regex-strict 24h `HH:MM` parsing. time-distance scoring (exact match = 1.0, ≤30 min off = 0.7, ≤60 min = 0.4, ≤120 min = 0.2, else 0.0). the substring shortcut is dead — `_slot_score("after 20:30", "later today", require_slot=True)` returns `0.0`. tested.
- **message**: length + on-topic verb + word-diversity check. keyword stuffing (`"reschedule, work, urgent."`) now scores 0.5 instead of 1.0; a real on-topic sentence still scores 1.0. tested.
- **weights**: `intent 0.40 / owner 0.20 / slot 0.20 / priority 0.10 / clarification 0.05 / message 0.05`. derivation documented in the module docstring (basically: most consequential decision gets the most weight, structural sanity checks get the least).
- **reward floor of 0.10? gone.** a fully wrong action scores 0. like it should.

### strengthened anti-hacking penalties

- `repetitive_intent` triggers on **3 in a row** (was 2; the old check was easy to bypass by alternating two intents forever).
- `premature_finalize` (`finalize_itinerary` with > 1 case left) jumps from 0.08 → 0.15.
- `terminal_early` is a new −0.05 for using *any* terminal-style intent to short-circuit out of an episode.
- `clarification_spam` is now aware of `pending_clarifications` and `revealed_info` — it actually checks whether the case had something to clarify.

### inference path (not Qwen 72B anymore)

`inference.py` was rewritten to default to the **trained** Qwen 2.5 3B Instruct + LoRA adapter via `transformers` and `peft` when `MODEL_PATH` is set. the HF Router 72B path is still there as a baseline option, but it's not the default. `EVAL_POOL=holdout` runs the honest holdout. `EVAL_POOL=adversarial` runs the probe pool.

if neither token nor model path is set, it falls back to a heuristic baseline so the script still does something visible. but yeah — the "1.0 was secretly a 72B model" criticism is dead.

## reward design (rebuilt, no fudging)

| signal           | weight | what it checks                                                                  |
|------------------|--------|----------------------------------------------------------------------------------|
| intent           | 0.40   | exact match — most consequential decision                                       |
| owner            | 0.20   | exact match — who handles it                                                    |
| slot             | 0.20   | regex-strict 24h `HH:MM`, time-distance scoring                                 |
| priority         | 0.10   | partial credit (0.5) for off-by-one — adjacent priorities are defensible        |
| clarification    | 0.05   | boolean alignment with `block_if_missing_context`                               |
| message          | 0.05   | length + on-topic verb + word-diversity                                         |

then the penalties on top, with no floor:

| penalty               | size   | when it fires                                                  |
|-----------------------|--------|-----------------------------------------------------------------|
| `repetitive_intent`   | −0.10  | same intent 3+ steps in a row                                  |
| `premature_finalize`  | −0.15  | `finalize_itinerary` with multiple conflicts still pending      |
| `terminal_early`      | −0.05  | any other terminal-style intent used to short-circuit           |
| `missing_slot`        | −0.05  | `require_slot=True` but no slot supplied                        |
| `clarification_spam`  | −0.05  | asking when the case has no clarification spec or info revealed |
| `short_message`       | −0.04  | message under 16 chars                                          |

reward band is `[0.0, 1.0]`. for real this time.

## how we trained it

**model**: Qwen 2.5 3B Instruct (4-bit quantized via Unsloth — fits a free Colab T4).

**approach**: SFT → GRPO (two-stage). same recipe as ChatGPT, just way smaller scale and on a way weirder task.

1. **SFT** on procedurally generated train episodes. no `* 15` duplication. every example is a unique procedural episode with varied times, owners, urgencies. each clarification case generates *two* SFT examples — the initial ask, and the post-reveal action — so the model learns to operate in the partial-obs regime.
2. **GRPO** on top, using **the env's actual reward** as the GRPO reward. the reward function in the notebook replays the env up to each step, applies the sampled completion, and reads `result.reward`. no placeholder rewards. no synthetic scoring.

LoRA only — `r=16`, ~1% of params trained. the rest stays frozen. tiny adapter, real signal.

**why GRPO?** no critic network needed (saves VRAM), works natively with TRL, and the "generate multiple completions → score them all against the real env → reinforce the best ones" loop maps perfectly to our deterministic grader.

## results

> **honesty section.** the numbers below will be filled in after rerunning the notebook on Colab T4. we are NOT reporting v0.2's `1.0 / 1.0 / 1.0` because those numbers came from a memorization regime that no longer exists in this codebase. dropping fake numbers > keeping them.

| pool                | untrained 3B | after SFT | after SFT + GRPO |
|---------------------|--------------|-----------|------------------|
| holdout (n=100)     | TBD          | TBD       | TBD              |
| adversarial (n=10)  | TBD          | TBD       | TBD              |

we expect holdout numbers to be *lower* than v0.2's `1.0` — that's the whole point. they'll reflect actual generalization to procedurally-novel episodes the model has never seen, with a grader that doesn't fold to substring shortcuts. that's the metric that means something.

## what stayed the same (so we didn't break the API)

- the action schema (6 intents, 6 owners, 4 priorities, slot, needs_clarification, message). backward-compatible for any client that already integrates with the v0.2 API.
- the OpenEnv `reset` / `step` / `state` interface and the FastAPI server.
- the Docker + HF Space deployment pipeline.
- the static fixture tasks for the live UI demo. they're kept around so the deployed Space still has interactive episodes for visitors. they are **not** used for training or evaluation. that's marked clearly in the code and the README.

## what we wish we'd had time for

- **curriculum**: harder episodes for later in training (we have the difficulty knob, we just didn't sweep it).
- **a proper user study or domain-expert pass on the reward weights.** the current weights are documented and auditable, but they're still our best judgment, not validated.
- **a bigger procedural template pool.** we have ~6 templates plus a finalize template; more variety would help generalization.
- **a learned reward model.** right now it's deterministic. a learned one would let us scale to fuzzier judgment calls.

## lessons we'd hammer into a future-self

- if you can't articulate what your environment's *world state* is, it isn't long-horizon. bookkeeping isn't state. step counters and reward histories don't count.
- a static fixture is a dataset. the moment you train on it AND evaluate on it, you've built a leaderboard for your own homework. and then you put that leaderboard on a slide.
- reward floors hide bugs. ours did. we never noticed because the model was always >0.10 — but we never noticed *because of the floor*. classic.
- "it works on three hand-written examples" doesn't generalize. fixing it to "it works on 100 procedural examples" is a different *kind* of works.
- when someone roasts your code with a numbered list, read the list before you defend yourself. we did, eventually, and the rebuild is way better for it.

---

## links

- **live environment**: [HuggingFace Space](https://huggingface.co/spaces/srivtx/openenv-conflict-resolver-v2)
- **training notebook**: `notebooks/train_grpo_colab.ipynb`
- **full docs**: see `docs/` (9 chapters from RL basics through the v0.3 rebuild)

*built with Unsloth, TRL, and a willingness to delete our own dishonest numbers.*
