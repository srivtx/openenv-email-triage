# Chapter 2: What is Reinforcement Learning?

No math. No jargon. Just the core idea, explained like you're learning it for the first time.

---

## 2.1 The Core Idea — Learning by Doing

Imagine teaching a puppy to sit:

1. You say "sit"
2. The puppy does something random (jumps, lies down, spins)
3. If it sits → you give a treat (reward = good)
4. If it doesn't → no treat (reward = nothing)
5. After many tries, the puppy learns: "sit" → sit → treat

**That's reinforcement learning.** An agent (the puppy) takes actions in an environment (your living room) and learns from rewards (treats).

```
    ┌─────────────────────────────────┐
    │         ENVIRONMENT             │
    │    (your living room)           │
    │                                 │
    │  State: "human said sit"        │
    │  Reward: +1 if correct          │
    └───────────┬─────────────────────┘
                │ shows state
                │ gives reward
                ▼
    ┌─────────────────────────────────┐
    │           AGENT                 │
    │       (the puppy)               │
    │                                 │
    │  Picks action: sit / jump / etc │
    └───────────┬─────────────────────┘
                │ takes action
                │
                └──────────► back to environment
```

---

## 2.2 The Key Terms (You'll See These Everywhere)

### Agent
The thing that learns. In our project: an LLM (like Qwen 3B).

### Environment
The world the agent interacts with. In our project: `PersonalAssistantConflictEnv`.

### State / Observation
What the agent can see right now. In our project: the current conflict description + constraints.

### Action
What the agent decides to do. In our project: `{intent: "reschedule", owner: "work", priority: "high", ...}`

### Reward
A number (0.0 to 1.0) saying how good the action was. In our project: calculated by the grader.

### Episode
One complete run from start to finish. In our project: resolving all conflicts in one task.

### Step
One action within an episode. In our project: resolving one conflict.

### Policy
The agent's strategy — "when I see X, I do Y." This is what training improves.

---

## 2.3 Why Not Just Write Rules?

You might think: "Why not just write `if conflict has missing info → ask clarification`?"

You CAN — and we did (that's our heuristic baseline). But rules break down because:

```
Rule-based approach:
✅ "Missing timezone" → ask clarification     (works!)
✅ "Overlap with dinner" → reschedule          (works!)
❌ "Board review overlaps school pickup, but visa 
    deadline is tonight and insurance expires at 
    midnight" → ??? (which one first? delegate what?)
```

When conflicts interact with each other, the number of possible combinations explodes. Writing rules for every case is impossible. But RL can learn patterns from experience.

---

## 2.4 How RL Training Works (The Loop)

```
Round 1:  Agent sees conflict → picks random action → reward = 0.2 (bad)
Round 2:  Agent sees conflict → picks different action → reward = 0.6 (better)
Round 3:  Agent sees conflict → picks another action → reward = 0.9 (great!)
...
Round 100: Agent sees conflict → consistently picks good actions → reward = 0.85 (learned!)
```

The agent doesn't memorize answers — it learns a **policy** (strategy) that works for new conflicts too.

---

## 2.5 RL vs Other Types of AI

| Type | How It Learns | Example |
|---|---|---|
| **Supervised Learning** | "Here's the question AND the answer" | Studying from a textbook |
| **Unsupervised Learning** | "Here's data, find patterns" | Organizing a messy room |
| **Reinforcement Learning** | "Try stuff, I'll tell you if it's good" | Learning to ride a bike |

**Why RL for our project?** We don't have millions of labeled "correct conflict resolutions" (supervised would need that). Instead, we have a reward function that can JUDGE any decision. RL is perfect for this.

---

## 2.6 The Exploration-Exploitation Tradeoff

Should the agent:
- **Explore** — try new actions it hasn't tried before?
- **Exploit** — keep doing what worked last time?

```
Example:
  Agent knows "reschedule" works okay (reward = 0.6)
  Should it try "delegate" which it's never tried?
  
  If it always exploits: gets stuck at 0.6 forever
  If it always explores: never learns, keeps trying random stuff
  
  Balance: mostly exploit, sometimes explore → finds 0.9 actions!
```

In GRPO training, this balance is handled automatically by generating multiple responses and comparing them.

---

## 2.7 What Makes OUR RL Special?

Traditional RL (like training a robot):
- Agent is a neural network with 1000 parameters
- Actions are simple: move left, move right, jump
- State is numbers: position, velocity, angle

Our RL (LLM-based):
- Agent is a language model with **3 BILLION parameters**
- Actions are structured JSON: `{intent, owner, priority, slot, ...}`
- State is natural language: "Board review overlaps with school pickup"

This is called **RLVR (Reinforcement Learning with Verifiable Rewards)** — a new approach where:
1. The LLM generates text (a JSON action)
2. The environment verifies if it's correct (deterministic grading)
3. The reward feeds back to update the LLM's weights

---

## 2.8 The OpenEnv Standard

OpenEnv is a standard interface for RL environments. Every OpenEnv environment must have:

```python
class MyEnvironment:
    async def reset(task_name) -> StepResult:
        """Start a new episode. Return initial observation."""
    
    async def step(action) -> StepResult:
        """Take an action. Return observation + reward + done."""
    
    async def state() -> State:
        """Return the full internal state."""
```

**Why a standard?** So any RL training framework can work with any environment. Like how USB works with any device — you don't need a special cable for each phone.

```
Any Training Framework    ←→    Any OpenEnv Environment
    (TRL, GRPO)                   (email triage, conflict resolver, wordle)
```

---

## 2.9 Mapping RL Terms to Our Project

| RL Term | In Our Project | Concrete Example |
|---|---|---|
| Agent | Qwen 3B LLM | The model that generates JSON actions |
| Environment | PersonalAssistantConflictEnv | The Python class that manages conflicts |
| State | ConflictObservation | Current conflict + constraints + history |
| Action | ConflictAction | `{intent: "reschedule", owner: "work", ...}` |
| Reward | 0.0 - 1.0 float | Calculated by graders.py |
| Episode | One full task | Resolving all 3/5/7 conflicts in a task |
| Step | One conflict resolution | Processing one conflict |
| Policy | LLM weights | The 3B parameters that decide what action to take |
| Training | GRPO with TRL | Updating weights so rewards go up |

---

## What's Next?

Now you understand the RL concepts. Chapter 3 shows how we **build the environment from scratch** — starting with the simplest possible version and adding complexity step by step.
