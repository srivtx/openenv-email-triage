# From Email Triage to Conflict Resolution: Our OpenEnv Journey

## The Real-World Problem We're Solving

Every knowledge worker drowns in the same chaos: overlapping meetings, conflicting deadlines, personal obligations colliding with work emergencies, and travel logistics that change at the last minute. Today's AI assistants can answer questions and set reminders — but they **cannot plan across multiple competing priorities**, handle missing information, or make tradeoff decisions when everything is urgent.

This project builds an RL training environment that teaches AI models to do exactly that.

---

## Round 1: Email Triage — Where We Started

### What It Did

In Round 1, we built **email-triage-openenv** — an environment where an AI agent processes an inbox of emails, one at a time, and makes a routing decision for each.

### The Task

An email arrives. The agent must decide:

| Decision | Options | Example |
|---|---|---|
| **Category** | billing, support, abuse, legal, engineering | "This billing dispute goes to..." |
| **Priority** | low, normal, high, urgent | "This is urgent because..." |
| **Team** | billing, support, engineering, legal, sales | "Route to the legal team" |
| **Spam?** | true / false | "This is a phishing attempt" |
| **Response** | Free-text template | "We've received your billing inquiry..." |

### How It Scored

The grader checked 5 things against the expected answer:
1. **Category match** — did you pick the right category?
2. **Priority match** — did you set the right urgency level?
3. **Team match** — did you route to the correct team?
4. **Spam detection** — did you correctly flag/unflag spam?
5. **Keyword quality** — did your response contain relevant terms?

### Penalties

- Short responses (< 12 chars) → -0.05
- Repetitive routing (same category + team 3x in a row) → -0.05
- Aggressive escalation (marking everything "urgent") → -0.02

### The 3 Tasks

| Task | Difficulty | What It Tests |
|---|---|---|
| easy_priority_routing | Easy | Clear-cut billing, account, and spam emails |
| medium_mixed_inbox | Medium | Mix of engineering, legal, and sales — less obvious routing |
| hard_ambiguous_escalations | Hard | Ambiguous cases where priority and ownership are debatable |

### What It Proved

Round 1 showed we could:
- ✅ Build a fully OpenEnv-compliant environment (reset/step/state)
- ✅ Design deterministic graders that output 0.0 → 1.0
- ✅ Deploy to HuggingFace Spaces
- ✅ Run baseline inference with structured logging

### Its Limitation

Email triage is a **single-step, single-decision problem**. Each email is independent. There's no memory, no dependencies between decisions, no need to plan ahead. A simple rule-based system can score well because the problem doesn't require reasoning across time.

---

## Round 2: Personal Assistant Conflict Resolver — Where We Are Now

### Why We Built This

Real personal assistant tasks aren't single-step. When your afternoon falls apart — a board review gets scheduled during school pickup, a visa deadline has a missing attachment, and your insurance payment is about to expire — you can't handle each problem in isolation. **Decisions cascade.** Rescheduling one meeting changes the constraints on everything else.

Round 2 builds an environment that captures this complexity.

### What It Does

The agent receives a queue of **conflicting commitments** — calendar overlaps, missing information, urgent deadlines, delegation needs — and must resolve them sequentially, where each decision can affect downstream conflicts.

### The Action Space (6 intents, not just routing)

| Intent | When To Use | Example |
|---|---|---|
| `route_message` | Forward to the right person/system | "Send insurance renewal alert to finance team" |
| `propose_plan` | Suggest a concrete alternative | "Move the demo to 11:30 so engineering lead can attend" |
| `reschedule_event` | Move a time-bound commitment | "Push dinner reservation to after 21:00 for flight check-in" |
| `delegate_task` | Hand off to someone else | "Ask family member to handle gift pickup since I'm in meetings" |
| `ask_clarification` | Block execution until info arrives | "Cannot confirm demo — timezone and participant list missing" |
| `finalize_itinerary` | Lock in the consolidated plan | "Final timeline: board review 2-4pm, pickup delegated, dinner 8:30pm" |

### The Decision Model

For every conflict, the agent must choose:

| Field | What it means |
|---|---|
| **intent** | What action to take (6 options above) |
| **owner** | Who owns this — self / work / family / travel / finance / legal |
| **priority** | How urgent — low / normal / high / urgent |
| **proposed_slot** | If rescheduling, the new time (e.g., "after 20:30") |
| **needs_clarification** | Should we block until we get more info? |
| **message_template** | The actual instruction/communication to send |

### How It Scores (6 components — up from 5 in Round 1)

| Component | Weight | What It Checks |
|---|---|---|
| Intent correctness | 34% | Did you pick the right action type? |
| Owner correctness | 20% | Did you assign the right responsible party? |
| Priority correctness | 15% | Did you set the right urgency? (distance-aware: off-by-one = partial credit) |
| Slot compliance | 14% | If a time slot was required, did you provide a valid one? |
| Clarification behavior | 10% | Did you ask for clarification when (and only when) context was missing? |
| Keyword quality | 7% | Does your message contain the expected action terms? |

### Penalties (5 types — up from 3 in Round 1)

| Penalty | Trigger | Cost | Why It Exists |
|---|---|---|---|
| Short message | Response < 16 chars | -0.04 | Prevents lazy one-word answers |
| Repetitive intent | Same intent 3 times in a row | -0.05 | Prevents spamming one action type |
| Premature finalize | Finalizing when conflicts remain | -0.08 | Prevents ending early to game reward |
| Missing slot | Required time slot not provided | -0.05 | Prevents vague rescheduling without a plan |
| Clarification spam | Asking for clarification when context is clear | -0.03 | Prevents stalling instead of deciding |

### The 3 Tasks (15 total conflicts)

#### Easy: Evening Conflict Cleanup (3 conflicts, 5 max steps)
- Family dinner overlaps with incident review → **reschedule**
- Medication pickup overlaps with school commute → **delegate**
- All blockers cleared, make final plan → **finalize**

*Tests: Basic routing + delegation with clear ownership*

#### Medium: Multi-Party Negotiation (5 conflicts, 8 max steps)
- Client demo has missing timezone → **ask clarification** (must block!)
- Need alternative demo slot → **propose plan** with time
- Flight check-in overlaps dinner → **reschedule** with new slot
- Gift pickup but user in workshop → **delegate** to family
- Make consolidated final plan → **finalize**

*Tests: Handling missing information + cross-party coordination*

#### Hard: Cascade Replanning (7 conflicts, 12 max steps)
- Board review overlaps school pickup → **reschedule** (urgent)
- Visa document missing attachment + unclear deadline → **ask clarification** (legal, must block!)
- Hotel cancel window closing during live call → **delegate** (time-critical)
- Investor dinner overlaps family exam prep → **propose plan** (split time)
- Insurance payment failing, grace period ending → **route** to finance (urgent)
- Driver availability changed, dinner commute delayed → **reschedule** transport
- Generate final itinerary with risk notes → **finalize** (urgent)

*Tests: Cascading disruptions where one wrong decision breaks everything downstream*

---

## How Round 1 → Round 2 Is Connected

### Same Foundation

| Aspect | Round 1 | Round 2 |
|---|---|---|
| OpenEnv contract | reset/step/state | reset/step/state ✅ same |
| Reward range | 0.0 → 1.0 | 0.0 → 1.0 ✅ same |
| Graders | Deterministic | Deterministic ✅ same |
| Deployment | HF Spaces Docker | HF Spaces Docker ✅ same |
| Inference logs | [START]/[STEP]/[END] | [START]/[STEP]/[END] ✅ same |

### Evolved Complexity

| Dimension | Round 1 (Email Triage) | Round 2 (Conflict Resolver) |
|---|---|---|
| **Domain** | Email inbox | Personal schedule + calendar + travel + finance |
| **Decision type** | Classify & route | Plan, negotiate, delegate, clarify, finalize |
| **Action space** | 5 fields | 6 fields with richer semantics |
| **Steps per task** | 3-7 (independent) | 3-12 (dependent — decisions cascade) |
| **Reward signals** | 5 components | 6 components (added slot compliance) |
| **Penalties** | 3 types | 5 types (added premature finalize + missing slot + clarification spam) |
| **Context dependency** | None — each email is standalone | High — resolving conflict A changes constraints on conflict B |
| **Clarification** | Not applicable | Agent must know when to ASK vs when to ACT |
| **Time reasoning** | Not applicable | Agent must propose valid time slots |
| **Multi-party** | Single team routing | Multiple owners (self/work/family/travel/finance/legal) |

### The Narrative

> We started by teaching AI to sort an inbox. Now we're teaching it to manage a chaotic day — balancing work deadlines, family obligations, travel logistics, and financial emergencies — all while handling missing information and cascading constraints.

---

## Why This Matters (The Bigger Picture)

### The Problem Today

Current AI assistants (Siri, Google Assistant, Alexa) are **reactive**: they respond to one command at a time. Ask them to "reschedule my 3pm meeting" and they'll do it. But they won't:

- Notice that rescheduling breaks your 4pm commitment
- Realize someone else needs to pick up your kid now
- Flag that your visa deadline is tomorrow and the attachment is missing
- Suggest delegating the hotel cancellation before the refund window closes

### What Our Environment Trains

Our environment teaches models to think like a **proactive human assistant** who:

1. **Sees the full picture** — all conflicts at once, not one at a time
2. **Makes tradeoff decisions** — when two urgent things conflict, which one moves?
3. **Knows when to ask** — if information is missing, block execution instead of guessing
4. **Delegates intelligently** — assign tasks to the right person (family for personal, travel agent for logistics)
5. **Plans ahead** — don't finalize until all conflicts are resolved
6. **Handles pressure** — time-critical decisions (hotel cancel window, insurance grace period)

### Why RL Is the Right Approach

You can't write rules for every possible calendar conflict. But you CAN:

1. Define what a **good decision looks like** (our 6-component reward function)
2. Define what **bad behavior looks like** (our 5 penalty types)
3. Let the model **learn through trial and error** (GRPO training)
4. Gradually **increase difficulty** (easy → medium → hard curriculum)

This is exactly what reinforcement learning is built for — learning complex decision-making through interaction with an environment.

---

## Training Pipeline

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    GRPO Training Loop                    │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │ Qwen 3B  │───▶│ Generate │───▶│ Environment      │  │
│  │ (LoRA)   │    │ JSON     │    │ Grades Decision   │  │
│  │          │◀───│ Action   │◀───│ Returns Reward    │  │
│  │          │    │          │    │ (0.0 → 1.0)       │  │
│  └──────────┘    └──────────┘    └──────────────────┘  │
│       │                                                  │
│       ▼                                                  │
│  Policy updates using reward signal (GRPO)              │
│  Model learns: high reward actions → more likely        │
│                low reward actions → less likely          │
└─────────────────────────────────────────────────────────┘
```

### Stack

| Component | Tool | Why |
|---|---|---|
| Base model | Qwen 2.5 3B Instruct | Small enough for free Colab T4 |
| Quantization | 4-bit (BnB) | Fits in 16GB VRAM |
| LoRA | r=16, alpha=16 | Train <1% of parameters efficiently |
| Acceleration | Unsloth | 2x faster fine-tuning |
| RL algorithm | GRPO (via TRL) | Group Relative Policy Optimization — no critic needed |
| Reward | Environment-backed | Every completion is scored by the real environment |

### What We Measure

| Metric | Meaning |
|---|---|
| Reward per step | Is each individual decision getting better? |
| Final task score | Does the full sequence of decisions lead to a good outcome? |
| Success rate | % of tasks where score ≥ 0.72 |
| Per-component scores | Which aspects (intent, owner, priority) improve most? |
| Penalty frequency | Is the model learning to avoid bad behaviors? |

---

## Deployment

| Platform | URL | Purpose |
|---|---|---|
| HuggingFace Space | [srivtx/openenv-conflict-resolver-v2](https://huggingface.co/spaces/srivtx/openenv-conflict-resolver-v2) | Live environment for judges and training |
| Round 1 Space | [srivtx/openenv-email-triage](https://huggingface.co/spaces/srivtx/openenv-email-triage) | Original email triage (Phase 1) |

Both environments pass `openenv validate` and expose the standard `/health`, `/reset`, `/step`, `/state` endpoints.

---

## What's Next

With more compute and training time, the natural extensions are:

1. **Real calendar integration** — connect to Google Calendar API for real conflicts
2. **Multi-day planning** — handle a full week of cascading commitments
3. **Preference learning** — adapt to individual user priorities (some people prioritize family, others work)
4. **Natural language interface** — let users describe conflicts in plain text instead of structured JSON
5. **Multi-agent delegation** — model actually sends messages to other agents/people and handles responses
