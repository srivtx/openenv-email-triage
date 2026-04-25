# Chapter 3: Building the Environment from Scratch

We'll build `PersonalAssistantConflictEnv` step by step. Each version adds one new concept. By the end, you'll understand every line of the real code.

> **note:** this chapter walks through a teaching ladder that arrives at a simple version of the environment for clarity. the shipped code goes further: it adds a `WorldState` carried across steps, a procedural conflict generator, two-step partial observability via clarification reveals, and cascade rules that spawn follow-on conflicts in response to the agent's actions. see `src/assistant_conflict_env/conflict_generator.py` for the full architecture.

---

## 3.1 Version 1 — The Absolute Minimum

The simplest possible environment: one conflict, one step, always returns 0.5.

```python
class ConflictEnvV1:
    """Version 1: Bare minimum — just reset and step."""
    
    def __init__(self):
        self.done = False
    
    async def reset(self):
        self.done = False
        return {"conflict": "Meeting A overlaps with Meeting B"}
    
    async def step(self, action):
        self.done = True
        return {
            "reward": 0.5,    # Always 0.5 — not useful yet
            "done": True,
            "observation": "Episode over"
        }
```

**What this does**: Shows the absolute skeleton. Reset starts a new game, step takes an action and ends it.

**What's wrong**: 
- No real grading (always 0.5)
- No typed data (just raw dicts)
- Only one conflict
- No state tracking

---

## 3.2 Version 2 — Add Typed Actions

Instead of accepting any random input, define exactly what an action looks like.

```python
# First, define what the agent can DO
class ActionIntent:
    RESCHEDULE = "reschedule_event"
    DELEGATE = "delegate_task"
    CLARIFY = "ask_clarification"

# Define the action structure
class ConflictAction:
    def __init__(self, intent: str, owner: str, priority: str, message: str):
        self.intent = intent      # What to do
        self.owner = owner        # Who owns it
        self.priority = priority  # How urgent
        self.message = message    # What to say


class ConflictEnvV2:
    def __init__(self):
        self.done = False
        self.expected_intent = "reschedule_event"  # The "correct" answer
    
    async def reset(self):
        self.done = False
        return {"conflict": "Meeting A overlaps with Meeting B"}
    
    async def step(self, action: ConflictAction):
        # Now we can actually GRADE the action!
        if action.intent == self.expected_intent:
            reward = 1.0  # Correct!
        else:
            reward = 0.0  # Wrong!
        
        self.done = True
        return {"reward": reward, "done": True}
```

**What we added**: Typed actions + simple grading (correct vs wrong).

**What's still wrong**:
- Binary reward (1.0 or 0.0) — too sparse, hard to learn from
- Only checks intent, ignores owner/priority
- No partial credit
- Still only one conflict

---

## 3.3 Version 3 — Multi-Component Rewards

Instead of binary right/wrong, check EACH part of the action separately.

```python
class ConflictEnvV3:
    def __init__(self):
        self.done = False
        # The answer key
        self.expected = {
            "intent": "reschedule_event",
            "owner": "work",
            "priority": "high"
        }
    
    async def step(self, action: ConflictAction):
        # Grade each component separately
        intent_score = 1.0 if action.intent == self.expected["intent"] else 0.0
        owner_score = 1.0 if action.owner == self.expected["owner"] else 0.0
        priority_score = 1.0 if action.priority == self.expected["priority"] else 0.0
        
        # Weighted combination
        reward = (
            0.50 * intent_score +    # Intent matters most
            0.30 * owner_score +     # Owner matters
            0.20 * priority_score    # Priority matters least
        )
        
        self.done = True
        return {
            "reward": reward,
            "done": True,
            "components": {
                "intent": intent_score,
                "owner": owner_score,
                "priority": priority_score
            }
        }
```

**What we added**: Multi-component reward. Now if the agent gets intent right but owner wrong, it gets 0.5 instead of 0.0. The agent learns WHICH parts to fix.

**Why this matters**: 
- Binary reward: "You're wrong" → agent has no idea what to fix
- Multi-component: "Intent is right, owner is wrong" → agent knows exactly what to fix

**What's still wrong**:
- Priority is still binary (high vs not-high = 0 or 1)
- No penalty for bad behaviors
- Still only one conflict per episode

---

## 3.4 Version 4 — Partial Credit (Reward Shaping)

Instead of "exactly right = 1.0, anything else = 0.0" for priority, give partial credit for close answers.

```python
# Priority ordering
PRIORITY_INDEX = {"low": 0, "normal": 1, "high": 2, "urgent": 3}

def priority_score(expected: str, actual: str) -> float:
    distance = abs(PRIORITY_INDEX[expected] - PRIORITY_INDEX[actual])
    if distance == 0:    # Exact match
        return 1.0
    elif distance == 1:  # Off by one level
        return 0.5
    else:                # Way off
        return 0.0

# Examples:
# expected="high", actual="high"   → 1.0 (perfect)
# expected="high", actual="urgent" → 0.5 (close, one level off)
# expected="high", actual="low"    → 0.0 (way off, two levels)
```

**Why partial credit?** Without it, the agent treats "off by 1" the same as "completely wrong." With partial credit, it learns to get closer over time:
- Round 1: picks "low" → reward 0.0 → learns this is bad
- Round 5: picks "normal" → reward 0.5 → learns this is better
- Round 10: picks "high" → reward 1.0 → learns this is best

This is called **reward shaping** — giving gradient information so the model can improve gradually instead of searching blindly.

---

## 3.5 Version 5 — Multiple Conflicts Per Episode

Real tasks have multiple conflicts, not just one. The agent processes them one by one.

```python
class ConflictEnvV5:
    def __init__(self):
        self.conflicts = [
            {"summary": "Dinner overlaps with review", "expected_intent": "reschedule_event"},
            {"summary": "Pickup overlaps with commute", "expected_intent": "delegate_task"},
            {"summary": "Make final plan", "expected_intent": "finalize_itinerary"},
        ]
        self.step_index = 0
        self.done = False
        self.rewards = []
    
    async def reset(self):
        self.step_index = 0
        self.done = False
        self.rewards = []
        return {"current_conflict": self.conflicts[0]}
    
    async def step(self, action: ConflictAction):
        # Grade current conflict
        current = self.conflicts[self.step_index]
        reward = 1.0 if action.intent == current["expected_intent"] else 0.0
        self.rewards.append(reward)
        
        # Move to next conflict
        self.step_index += 1
        self.done = self.step_index >= len(self.conflicts)
        
        if self.done:
            final_score = sum(self.rewards) / len(self.rewards)
            return {"reward": reward, "done": True, "final_score": final_score}
        else:
            next_conflict = self.conflicts[self.step_index]
            return {"reward": reward, "done": False, "current_conflict": next_conflict}
```

**What we added**: The agent now handles a SEQUENCE of conflicts. This is what makes it "long-horizon" — the agent must make good decisions across multiple steps.

**Key insight**: `step_index` tracks where we are in the sequence. Each `step()` call advances to the next conflict. When we run out of conflicts, `done = True`.

---

## 3.6 Version 6 — Add Penalties (Anti-Hacking)

Without penalties, the agent can "hack" the reward by doing stupid things that technically don't lose points.

```python
def compute_penalties(self, action, current_conflict):
    penalties = {}
    
    # Penalty 1: Don't send lazy one-word messages
    if len(action.message) < 16:
        penalties["short_message"] = 0.04
    
    # Penalty 2: Don't spam the same action type
    if len(self.decisions) >= 2:
        last_two = self.decisions[-2:]
        if all(d.intent == action.intent for d in last_two):
            penalties["repetitive_intent"] = 0.05
    
    # Penalty 3: Don't finalize early when conflicts remain
    remaining = len(self.conflicts) - self.step_index
    if action.intent == "finalize_itinerary" and remaining > 1:
        penalties["premature_finalize"] = 0.08
    
    return penalties
```

**Why we need each penalty**:

| Without Penalty | What Agent Does | Why It's Bad |
|---|---|---|
| No short_message | Sends "ok" as message | Gives no useful information |
| No repetitive_intent | Sends "reschedule" for everything | Doesn't actually solve different problems |
| No premature_finalize | Ends episode immediately | Gets average reward without trying hard cases |

**How penalties apply**:
```python
raw_reward = 0.10 + 0.90 * grader_score  # Base reward from grading
penalty_total = sum(penalties.values())    # Sum all penalties
final_reward = max(0.0, min(1.0, raw_reward - penalty_total))  # Subtract and clamp
```

---

## 3.7 Version 7 — Pydantic Models (What We Actually Use)

Now let's upgrade from raw dicts to proper Pydantic models. This is what the real code uses.

```python
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

# Define allowed values as enums
class ActionIntent(str, Enum):
    ROUTE_MESSAGE = "route_message"
    PROPOSE_PLAN = "propose_plan"
    RESCHEDULE_EVENT = "reschedule_event"
    DELEGATE_TASK = "delegate_task"
    ASK_CLARIFICATION = "ask_clarification"
    FINALIZE_ITINERARY = "finalize_itinerary"

class Owner(str, Enum):
    SELF = "self"
    WORK = "work"
    FAMILY = "family"
    TRAVEL = "travel"
    FINANCE = "finance"
    LEGAL = "legal"

class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

# The action the agent sends
class ConflictAction(BaseModel):
    intent: ActionIntent
    owner: Owner
    priority: Priority
    proposed_slot: str = ""
    needs_clarification: bool = False
    message_template: str = ""

# What the agent sees
class ConflictObservation(BaseModel):
    task_id: str
    step_index: int
    max_steps: int
    remaining_count: int
    current_conflict: Optional[dict] = None   # The current problem to solve
    history: List[dict] = []                   # Last 3 decisions
    open_risks: List[str] = []                 # Current constraints
    last_feedback: str = ""                    # Hint from last step

# What step() returns
class ConflictStepResult(BaseModel):
    observation: ConflictObservation
    reward: float
    done: bool
    info: dict = {}
```

**Why Pydantic instead of raw dicts?**

```python
# With raw dict — anything goes, no validation
action = {"intent": "fly_to_moon", "owner": 42, "priority": "ASAP"}
# No error... but everything breaks later

# With Pydantic — instant validation
action = ConflictAction(intent="fly_to_moon", owner=42, priority="ASAP")
# ❌ ValidationError: 'fly_to_moon' is not a valid ActionIntent
# Caught immediately! You know exactly what's wrong
```

---

## 3.8 The Final Version — Putting It All Together

Here's how V1 → V7 built up to the real environment:

| Version | What It Added | Why |
|---|---|---|
| V1 | reset() + step() | Bare minimum contract |
| V2 | Typed actions | Agent can't send garbage |
| V3 | Multi-component reward | Agent knows WHAT's wrong |
| V4 | Partial credit | Agent can improve gradually |
| V5 | Multiple conflicts | Long-horizon episodes |
| V6 | Penalties | Agent can't hack rewards |
| V7 | Pydantic models | Data validation everywhere |

The real `environment.py` is exactly V7 with some extra bookkeeping (state tracking, feedback messages, final grading). Every line exists for a reason we've now explained.

---

## 3.9 The Environment Lifecycle

```
User/Trainer                    Environment
    │                               │
    │──── reset("easy_task") ──────▶│  Creates fresh state
    │◀─── observation + reward=0 ───│  Shows first conflict
    │                               │
    │──── step(action_1) ──────────▶│  Grades action, applies penalties
    │◀─── observation + reward=0.8 ─│  Shows second conflict
    │                               │
    │──── step(action_2) ──────────▶│  Grades action
    │◀─── observation + reward=0.6 ─│  Shows third conflict
    │                               │
    │──── step(action_3) ──────────▶│  Grades action, computes final score
    │◀─── done=True + reward=0.9 ───│  Episode complete!
    │                               │
    │──── state() ─────────────────▶│
    │◀─── full state with scores ───│  Final score = average of all steps
```

---

## What's Next?

Chapter 4 dives into the **grader** — the code that actually decides whether an action is good or bad. We'll look at each of the 6 scoring components and understand why they're weighted the way they are.
