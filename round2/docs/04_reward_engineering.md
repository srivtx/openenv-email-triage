# Chapter 4: Reward Engineering — The Heart of the System

The reward function is the most important part of any RL system. A bad reward = a model that learns the wrong thing. This chapter explains every scoring decision in our graders.

> **note:** the weights and scoring described below illustrate a teaching version of the grader. the shipped grader uses regex-strict 24h `HH:MM` parsing, a length+verb+diversity message proxy, no reward floor, and documented weight derivation. see the module docstring in `src/assistant_conflict_env/graders.py` for the full design.

---

## 4.1 What the Grader Does

The grader takes two things:
1. **Expected answer** (from `conflict_cases.json` — the answer key)
2. **Agent's action** (what the model decided)

And returns a score between 0.0 and 1.0.

```python
# Simplified version
def score_action(expected, action) -> float:
    intent_score = 1.0 if action.intent == expected.intent else 0.0
    owner_score = 1.0 if action.owner == expected.owner else 0.0
    # ... more components ...
    
    total = 0.34 * intent_score + 0.20 * owner_score + ...
    return total
```

---

## 4.2 The 6 Components — Why These? Why These Weights?

### Component 1: Intent (weight: 0.34 — highest)

**What it checks**: Did the agent pick the right action type?

```python
intent_score = 1.0 if action.intent == expected.intent else 0.0
```

**Why it's heaviest (34%)**: If you delegate when you should reschedule, the entire decision is wrong. Intent is the most fundamental choice.

**Example**:
- Conflict: "Flight check-in overlaps with dinner"
- Expected: `reschedule_event`
- Agent says `delegate_task` → intent_score = 0.0 (wrong approach entirely)
- Agent says `reschedule_event` → intent_score = 1.0 (right approach)

**Why binary (0 or 1)?** There's no "close" intent. Rescheduling and delegating are completely different actions. You can't give partial credit for picking the wrong action type.

---

### Component 2: Owner (weight: 0.20)

**What it checks**: Did the agent assign the right responsible party?

```python
owner_score = 1.0 if action.owner == expected.owner else 0.0
```

**Why 20%**: Knowing WHO should handle something is important but secondary to knowing WHAT to do.

**Example**:
- Conflict: "Insurance payment failed"
- Expected owner: `finance`
- Agent says `work` → 0.0 (finance team handles payments, not general work)
- Agent says `finance` → 1.0

---

### Component 3: Priority (weight: 0.15 — with partial credit!)

**What it checks**: Did the agent set the right urgency level?

```python
PRIORITY_INDEX = {"low": 0, "normal": 1, "high": 2, "urgent": 3}

def priority_score(expected, actual):
    distance = abs(PRIORITY_INDEX[expected] - PRIORITY_INDEX[actual])
    if distance == 0: return 1.0   # Exact
    if distance == 1: return 0.5   # Close
    return 0.0                      # Way off
```

**Why partial credit here but not for intent?** Priorities are on a SCALE. "High" is close to "urgent" — off by one level is a reasonable mistake. But "reschedule" is not close to "delegate" — they're categorically different.

**Example**:
- Expected: `high`
- Agent says `urgent` → 0.5 (one level off, not terrible)
- Agent says `low` → 0.0 (two levels off, bad misjudgment)

---

### Component 4: Slot Compliance (weight: 0.14)

**What it checks**: If a time slot was required, did the agent propose a valid one?

```python
def slot_score(expected_hint, proposed_slot, require_slot):
    if not require_slot:          # Slot not needed for this conflict
        return 1.0                # Free points!
    
    if not proposed_slot.strip(): # Slot required but not provided
        return 0.0                # Failed to provide crucial info
    
    hint = expected_hint.lower()
    slot = proposed_slot.lower()
    
    if hint and hint in slot:     # Contains the expected hint
        return 1.0                # Perfect — "after 20:30" matches
    
    if any(t in slot for t in ["am", "pm", "today", "after", ":"]):
        return 0.6                # At least it's a valid time expression
    
    return 0.2                    # Something was provided but not useful
```

**Why the tiered scoring?**

| Score | What It Means | Example |
|---|---|---|
| 1.0 | Perfect slot | Expected "after 20:30", got "reschedule to after 20:30" |
| 0.6 | Valid time, wrong value | Expected "after 20:30", got "move to 3pm" |
| 0.2 | Something provided | Expected "after 20:30", got "later today" |
| 0.0 | Nothing provided | Required slot but left blank |

**Why not binary?** A wrong time is better than no time. "Move to 3pm" shows the model understands it needs to propose a time — it just got the wrong one. That deserves partial credit.

---

### Component 5: Clarification Behavior (weight: 0.10)

**What it checks**: Did the agent ask for clarification when (and only when) information was missing?

```python
def clarification_score(block_if_missing, needs_clarification):
    if block_if_missing:                    # Context IS missing
        return 1.0 if needs_clarification else 0.0  # MUST ask
    return 0.75 if needs_clarification else 1.0      # Shouldn't ask (penalty if does)
```

**Two cases**:

| Context Missing? | Agent Asks? | Score | Why |
|---|---|---|---|
| Yes | Yes | 1.0 | Correct! Don't act without info |
| Yes | No | 0.0 | Dangerous! Acting on incomplete info |
| No | No | 1.0 | Correct! Info is complete, just act |
| No | Yes | 0.75 | Unnecessary delay, mild penalty |

**Why this matters**: In the visa deadline conflict, the attachment is missing and the timezone is unclear. If the agent just says "submit the form" without asking for the missing attachment, that's a real-world disaster (wrong form submitted, legal penalties).

---

### Component 6: Keyword Quality (weight: 0.07 — lowest)

**What it checks**: Does the agent's message contain relevant terms?

```python
def keyword_score(required_keywords, message):
    if not required_keywords:
        return 1.0
    
    text = message.lower()
    hits = sum(1 for word in required_keywords if word in text)
    return hits / len(required_keywords)
```

**Example**:
- Required keywords: `["reschedule", "incident"]`
- Message: "Reschedule the incident review to after 8:30pm" → 2/2 = 1.0
- Message: "Move the meeting" → 0/2 = 0.0 (no keywords matched)
- Message: "Reschedule the event" → 1/2 = 0.5 (partial)

**Why lowest weight?** The message text is least important — what matters is the DECISION (intent + owner + priority). The message is just communication polish.

---

## 4.3 Weight Distribution — The Design Decision

```
Intent:         ████████████████████████████████████  34%
Owner:          ████████████████████                  20%
Priority:       ███████████████                       15%
Slot:           ██████████████                        14%
Clarification:  ██████████                            10%
Keyword:        ███████                                7%
                                                     ────
                                                     100%
```

**Design philosophy**: What to do > Who does it > How urgent > When > Should we wait? > What to say

---

## 4.4 The Final Reward Formula

```python
# In environment.py, line 80
reward = max(0.0, min(1.0, 0.10 + 0.90 * grader_score - penalty_total))
```

Let's break this down piece by piece:

| Part | What | Why |
|---|---|---|
| `0.10` | Reward floor | Even the worst action gets 0.10 — prevents "all zero" gradient death |
| `0.90 * grader_score` | Scaled grader score | The remaining 0.90 comes from actual performance |
| `- penalty_total` | Subtract penalties | Bad behaviors reduce the reward |
| `max(0.0, min(1.0, ...))` | Clamp to [0, 1] | Rewards must stay in valid range |

**Walk-through example**:
```
Agent picks the right intent (1.0), right owner (1.0), close priority (0.5),
provides a time slot (0.6), doesn't ask clarification when shouldn't (1.0),
message has one keyword (0.5).

grader_score = 0.34×1.0 + 0.20×1.0 + 0.15×0.5 + 0.14×0.6 + 0.10×1.0 + 0.07×0.5
             = 0.34 + 0.20 + 0.075 + 0.084 + 0.10 + 0.035
             = 0.834

penalties = 0.0 (no penalties triggered)

reward = max(0.0, min(1.0, 0.10 + 0.90 × 0.834 - 0.0))
       = max(0.0, min(1.0, 0.10 + 0.7506))
       = max(0.0, min(1.0, 0.8506))
       = 0.8506
```

---

## 4.5 The Final Grading — Episode Score

After all conflicts are resolved, `grade_task_decisions()` calculates the final score:

```python
def grade_task_decisions(task, decisions):
    scores = []
    for conflict in task.conflicts:
        decision = find_matching_decision(conflict, decisions)
        if decision is None:
            scores.append(0.0)  # Missed this conflict entirely
        else:
            scores.append(score_action(conflict.expected, decision).score)
    
    final_score = sum(scores) / len(task.conflicts)  # Average across all conflicts
    return final_score
```

**Why average?** If the agent nails 2 out of 3 conflicts but completely fails the third, it gets ~0.66 — not 1.0. Every conflict matters.

---

## What's Next?

Chapter 5 explains the **training pipeline** — how GRPO takes these reward signals and actually updates the model's weights to make better decisions.
