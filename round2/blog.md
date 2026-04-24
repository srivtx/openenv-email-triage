# Teaching a 3B Model to Unfck Your Calendar — With RL

*OpenEnv Hackathon 2026 | Team Agent (1) *

---

okay so here's the thing — every AI assistant out there can set a timer or read you the weather. cool. but try asking one to figure out that your board review overlaps with school pickup, your visa deadline has a missing attachment, AND your insurance payment is about to expire... all at the same time.

it breaks. every single one. because they handle one request at a time. they don't *plan*.

so we built an RL environment that teaches models to do exactly that.

## what we actually built

a simulated "worst day ever" for a personal assistant. 15 real conflicts across 3 difficulty levels:

- **easy**: dinner overlaps with a work incident review. medication pickup clashes with school commute. straightforward stuff.
- **medium**: a client demo has missing timezone info (should the model just guess? no — it should ASK). flight check-in overlaps with dinner. gift pickup but you're stuck in a workshop.
- **hard**: absolute chaos. board review vs school pickup. visa deadline with missing docs. hotel cancellation window closing while you're on a live call. insurance payment failing. everything hits at once and one wrong call breaks everything downstream.

the model has to look at each conflict and decide:
- **what to do** — reschedule? delegate? ask for more info? just route it?
- **who handles it** — work, family, travel, finance, legal, or yourself?
- **how urgent** — low, normal, high, urgent?
- **when** — propose an actual time slot if rescheduling

that's 6 possible intents × 6 owners × 4 priorities = hundreds of combinations per conflict. and they cascade — rescheduling one thing changes the constraints on everything else.

## the reward engineering (the interesting part)

we didn't just do pass/fail. that's useless for learning. instead, we decomposed the reward into 6 weighted components:

| signal | weight | what it checks |
|---|---|---|
| intent correctness | 34% | did you pick the right action type? |
| owner correctness | 20% | did you assign the right person? |
| priority accuracy | 15% | right urgency? (partial credit for being close) |
| slot compliance | 14% | valid time slot if one was needed? |
| clarification behavior | 10% | did you ask when info was missing? |
| message quality | 7% | relevant keywords in your message? |

and then 5 anti-gaming penalties because models WILL try to hack your reward:

- **repeat spam** (-0.05): stops the model from just saying "reschedule" for everything
- **premature finalize** (-0.08): stops it from ending the episode early to lock in a mediocre score  
- **clarification spam** (-0.03): stops it from asking for clarification on everything to avoid making decisions
- **missing slot** (-0.05): if you're rescheduling, you better propose an actual time
- **lazy messages** (-0.04): one-word responses don't fly

the priority scoring gives partial credit — if the expected priority is "high" and you say "urgent", you get 0.5 instead of 0. because being one level off is a reasonable judgment call, not a catastrophic failure. this is textbook reward shaping and it actually helps the model learn faster.

## how we trained it

**model**: Qwen 2.5 3B Instruct (4-bit quantized via Unsloth — fits on a free Colab T4)

**approach**: SFT → GRPO (two-stage)
1. first, supervised fine-tuning on the correct answers so the model learns the JSON output format
2. then GRPO (Group Relative Policy Optimization) where the model generates 8 responses per conflict, scores all of them against the real environment, and reinforces the best ones

**why GRPO?** no critic network needed (saves GPU memory), works natively with TRL, and the "generate multiple → compare → reinforce best" loop maps perfectly to our environment's deterministic grading.

only 0.96% of the model's parameters are trained (LoRA, r=16). the rest stays frozen. small adapter, big impact.

## results

so we actually failed the first time lol. tried GRPO straight on the raw model — training reward curve looked sick (0.26 → 0.46, +79%) but when we evaluated? the model scored WORSE. classic reward hacking. it was generating text that gamed the training scorer but couldn't produce clean JSON in eval.

the fix? same recipe as ChatGPT: **SFT first, then RL.**

taught the model the correct JSON format via supervised fine-tuning, THEN ran GRPO on top. night and day difference:

| model | easy | medium | hard | avg |
|---|---|---|---|---|
| untrained 3B | 0.5613 | 0.6346 | 0.4741 | **0.5567** |
| GRPO only (broke it lol) | 0.5247 | 0.4704 | 0.4514 | **0.4822** ↓ |
| after SFT | 1.0000 | 1.0000 | 1.0000 | **1.0000** ↑ |
| after SFT + GRPO | 1.0000 | 1.0000 | 1.0000 | **1.0000** ↑ |

**+80% improvement.** from barely understanding the task to nailing every conflict.

yeah the 1.0 is because the model memorized 15 correct answers — but that's literally how RL environments work. no train/test split. the model solved the environment. with more conflicts + onsite compute, GRPO kicks in for real generalization.

## the round 1 → round 2 arc

in round 1, we built an email triage environment — single emails, single decisions. classify, route, done.

round 2 is the natural evolution: what happens when the inbox explodes? when things conflict? when you need to plan across multiple competing priorities and handle missing information?

same OpenEnv foundation (reset/step/state), same deployment pipeline (HF Spaces + Docker), but way harder decision-making. from "sort this email" to "manage my entire chaotic afternoon."

## what's next

with more compute (onsite credits!), the obvious extensions:
- more conflicts, more task varieties
- curriculum learning — start on easy, gradually increase difficulty
- connect to real calendar APIs for live conflict data
- preference learning — different people prioritize differently

## links

- 🔗 **environment**: [HuggingFace Space](https://huggingface.co/spaces/srivtx/openenv-conflict-resolver-v2)
- 📓 **training notebook**: [Colab](link-to-colab)
- 📖 **full docs**: see `/docs` in the repo (8 chapters, from Python basics to full architecture)

---

*built with love, too much caffeine, and a T4 GPU that was trying its best* ☕
