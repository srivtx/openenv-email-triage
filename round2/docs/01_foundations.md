# Chapter 1: Python Foundations You Need

Before touching any RL or AI code, let's make sure the Python building blocks are crystal clear. Every concept here is used in our project.

---

## 1.1 Classes — The Blueprint Pattern

A class is a blueprint for creating objects. Think of it like a cookie cutter — the class is the cutter, objects are the cookies.

```python
# The simplest class
class Dog:
    def __init__(self, name):
        self.name = name
    
    def bark(self):
        return f"{self.name} says Woof!"

buddy = Dog("Buddy")
print(buddy.bark())  # "Buddy says Woof!"
```

**Why we need this**: Our entire environment is a class called `PersonalAssistantConflictEnv`. It holds the game state and has methods like `reset()`, `step()`, and `state()`.

**What happens if we don't use classes?** We'd have to pass dozens of variables between functions manually. Classes keep related data and behavior together.

---

## 1.2 Type Hints — Telling Python What to Expect

Python doesn't force you to declare types, but type hints make code readable and catch bugs early.

```python
# Without type hints — confusing
def add(a, b):
    return a + b

# With type hints — clear
def add(a: int, b: int) -> int:
    return a + b

# With complex types
from typing import List, Optional, Dict

def process(items: List[str], limit: Optional[int] = None) -> Dict[str, int]:
    result = {}
    for item in items[:limit]:
        result[item] = len(item)
    return result
```

**Why we need this**: Our environment uses types everywhere — `ConflictAction`, `ConflictObservation`, `Priority`. This makes it impossible to accidentally pass wrong data.

**What happens without types?** You'd get mysterious errors like "AttributeError: 'str' has no attribute 'intent'" deep in the code instead of at the point where you made the mistake.

---

## 1.3 Enums — Named Constants

An enum is a set of named values. Instead of using strings like `"high"` or `"urgent"` (easy to misspell), we use enums.

```python
from enum import Enum

# Bad — using raw strings
priority = "hihg"  # Typo! No error until runtime

# Good — using enums
class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

priority = Priority.HIGH  # Can't misspell — IDE catches it
print(priority.value)     # "high"
```

**Why `str, Enum`?** By inheriting from both `str` and `Enum`, our priority values work as regular strings in JSON. `Priority.HIGH == "high"` returns `True`.

**In our project**: We have 3 enums:
- `ActionIntent` — 6 possible actions (reschedule, delegate, etc.)
- `Owner` — 6 possible owners (self, work, family, etc.)
- `Priority` — 4 levels (low, normal, high, urgent)

---

## 1.4 Pydantic — Data Validation on Steroids

Pydantic is a library that validates data automatically. Instead of writing manual checks, you declare what your data should look like.

```python
# Without Pydantic — manual validation hell
def create_user(data):
    if "name" not in data:
        raise ValueError("name required")
    if not isinstance(data["name"], str):
        raise TypeError("name must be string")
    if "age" not in data:
        raise ValueError("age required")
    if not isinstance(data["age"], int):
        raise TypeError("age must be int")
    if data["age"] < 0:
        raise ValueError("age must be positive")
    # ... exhausting

# With Pydantic — declare once, validate automatically
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str
    age: int = Field(ge=0)  # ge = greater than or equal to 0

user = User(name="Sribatsha", age=22)     # ✅ Works
user = User(name="Sribatsha", age=-1)     # ❌ ValidationError automatically
user = User(name=123, age="hello")        # ❌ ValidationError automatically
```

**Why we need Pydantic**: Every piece of data flowing through our environment is a Pydantic model:
- `ConflictAction` — what the AI agent sends
- `ConflictObservation` — what the environment shows
- `ConflictState` — the internal game state
- `ConflictStepResult` — what the environment returns

**What happens without it?** Invalid data sneaks through. An agent sends `priority: "super_urgent"` (not a valid value) and instead of a clean error, you get a wrong reward calculation 10 steps later. Debugging nightmare.

**Install**: `pip install pydantic`

---

## 1.5 Async/Await — Doing Things Without Blocking

Normal Python runs one thing at a time. Async lets you start something, go do other work, and come back when it's done.

```python
import asyncio

# Normal function — blocks everything
def fetch_data():
    import time
    time.sleep(2)  # Nothing else can happen for 2 seconds
    return "data"

# Async function — doesn't block
async def fetch_data():
    await asyncio.sleep(2)  # Other code can run during these 2 seconds
    return "data"

# Calling async functions
async def main():
    result = await fetch_data()  # "await" means "wait for this to finish"
    print(result)

asyncio.run(main())
```

**Why we need async**: OpenEnv requires all environment methods to be async:
```python
async def reset(self, task_name=None) -> ConflictStepResult:
async def step(self, action: ConflictAction) -> ConflictStepResult:
async def state(self) -> ConflictState:
```

This is because in production, the environment runs as a web server (FastAPI), and async lets it handle multiple requests simultaneously.

**What happens without async?** If two training runs call your environment at the same time, the second one has to wait until the first finishes. With async, both can run together.

**The `await` keyword**: Every time you call an async function, you must use `await`:
```python
# Wrong — this gives you a coroutine object, not the result
result = env.reset()  # <coroutine object at 0x...>

# Right — this actually runs the function
result = await env.reset()  # ConflictStepResult(...)
```

---

## 1.6 Dataclasses — Simple Data Containers

Dataclasses are like Pydantic models but lighter — no validation, just convenient data storage.

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

p = Point(x=1.0, y=2.0)
print(p)  # Point(x=1.0, y=2.0)

# frozen=True means you can't change values after creation
@dataclass(frozen=True)
class ScoreBreakdown:
    score: float
    components: dict

breakdown = ScoreBreakdown(score=0.85, components={"intent": 1.0, "owner": 0.8})
breakdown.score = 0.9  # ❌ Error! Frozen dataclass
```

**In our project**: We use `@dataclass(frozen=True)` for `ScoreBreakdown` and `GradeReport` in the graders. Frozen means "read-only" — once a score is calculated, nobody can accidentally change it.

---

## 1.7 Dictionaries — The Everywhere Data Structure

```python
# Basic dict
scores = {"intent": 1.0, "owner": 0.8, "priority": 0.5}

# Accessing
print(scores["intent"])       # 1.0
print(scores.get("missing", 0.0))  # 0.0 (default if key missing)

# Iterating
for key, value in scores.items():
    print(f"{key}: {value}")

# Dict comprehension
penalties = {name: 0.0 for name in ["short_message", "repetitive_intent"]}
```

**In our project**: Reward components, penalties, and info dicts are all dictionaries:
```python
penalties = {
    "short_message": 0.0,
    "repetitive_intent": 0.0,
    "premature_finalize": 0.0,
    "missing_slot": 0.0,
    "clarification_spam": 0.0,
}
```

---

## 1.8 List Slicing — Getting Parts of Lists

```python
decisions = ["reschedule", "delegate", "clarify", "propose", "finalize"]

decisions[-3:]   # Last 3: ["clarify", "propose", "finalize"]
decisions[:2]    # First 2: ["reschedule", "delegate"]
decisions[-2:]   # Last 2: ["propose", "finalize"]
```

**In our project**: `self._state.decisions[-3:]` — we show the agent its last 3 decisions as context. This is like giving it "memory" of what it recently did.

---

## 1.9 Max, Min, Clamp — Keeping Numbers in Range

```python
# Clamp a value between 0.0 and 1.0
reward = max(0.0, min(1.0, some_value))

# If some_value = 1.5 → min(1.0, 1.5) = 1.0 → max(0.0, 1.0) = 1.0
# If some_value = -0.3 → min(1.0, -0.3) = -0.3 → max(0.0, -0.3) = 0.0
# If some_value = 0.7 → min(1.0, 0.7) = 0.7 → max(0.0, 0.7) = 0.7
```

**In our project**: Rewards must always be between 0.0 and 1.0. After adding bonuses and subtracting penalties, we clamp:
```python
reward = max(0.0, min(1.0, 0.10 + 0.90 * score - penalties))
```

---

## What's Next?

Now that you know the Python building blocks, Chapter 2 explains **what Reinforcement Learning is** — in simple terms, no math required.

Every concept from this chapter will show up again when we build the actual environment.
