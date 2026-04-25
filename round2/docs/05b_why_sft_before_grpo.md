# Chapter 5B: Why SFT Before GRPO — The Full Story with Code

This chapter explains SFT (Supervised Fine-Tuning) from absolute scratch — what it is, why we need it, how it works in code, and what happens when you skip it.

> **note:** the code snippets below illustrate the SFT-then-GRPO pipeline using a small teaching dataset for clarity. the shipped notebook (`notebooks/train_grpo_colab.ipynb`) sources SFT data from a procedural train pool of unique episodes, uses the env's real reward as the GRPO reward, and evaluates on a disjoint procedural holdout pool. for the actual measured numbers, see `README.md`.

---

## 5B.1 What is SFT? (The Basics)

SFT = Supervised Fine-Tuning. It's the simplest form of training:

```
"Here's the question. Here's the correct answer. Learn this."
```

That's it. You show the model input-output pairs and it learns to reproduce them.

```python
# The simplest possible SFT example
training_data = [
    {"input": "What is 2+2?", "output": "4"},
    {"input": "What is 3+3?", "output": "6"},
    {"input": "What is 5+5?", "output": "10"},
]

# The model sees "What is 2+2?" and learns to say "4"
# After training, if you ask "What is 2+2?" it confidently says "4"
```

**SFT vs GRPO:**

| | SFT | GRPO |
|---|---|---|
| What you give it | Question + correct answer | Question + reward function |
| How it learns | "Copy this answer" | "Try stuff, I'll score you" |
| Speed | Fast (direct learning) | Slow (trial and error) |
| Needs correct answers? | Yes | No (just needs a scorer) |
| Can discover new answers? | No (only learns what you show) | Yes (explores on its own) |

---

## 5B.2 What is GRPO? (Refresher with Code)

GRPO = Group Relative Policy Optimization. For each question:

```python
# Pseudocode of what GRPO does internally
def grpo_one_step(model, prompt, reward_function):
    # Step 1: Generate 8 different responses
    responses = [model.generate(prompt) for _ in range(8)]
    
    # Step 2: Score all 8
    scores = [reward_function(r) for r in responses]
    # Example: [0.2, 0.8, 0.1, 0.6, 0.3, 0.9, 0.4, 0.5]
    
    # Step 3: Calculate average
    avg = sum(scores) / len(scores)  # 0.475
    
    # Step 4: Calculate advantage (how much better/worse than average)
    advantages = [s - avg for s in scores]
    # [−0.275, +0.325, −0.375, +0.125, −0.175, +0.425, −0.075, +0.025]
    
    # Step 5: Update model
    # Make high-advantage responses MORE likely
    # Make low-advantage responses LESS likely
    for response, advantage in zip(responses, advantages):
        if advantage > 0:
            model.increase_probability(response)  # "do more of this"
        else:
            model.decrease_probability(response)  # "do less of this"
```

**The key insight:** GRPO doesn't need correct answers. It just needs a way to SCORE responses. Our environment provides that scoring.

---

## 5B.3 What Went Wrong: GRPO Without SFT

Here's what actually happened when we ran GRPO on the untrained Qwen 3B:

### The Training Logs (Looked Great!)

```
Step  5:  reward = 0.257   reward_std = 0.244
Step 10:  reward = 0.319   reward_std = 0.240
Step 15:  reward = 0.382   reward_std = 0.199
Step 20:  reward = 0.416   reward_std = 0.214
Step 25:  reward = 0.429   reward_std = 0.191
Step 30:  reward = 0.418   reward_std = 0.197
Step 35:  reward = 0.441   reward_std = 0.181
Step 40:  reward = 0.446   reward_std = 0.175
Step 45:  reward = 0.461   reward_std = 0.169
Step 50:  reward = 0.459   reward_std = 0.173
```

Reward going up! Standard deviation going down! Everything looks perfect!

### The Evaluation (Revealed the Problem)

```
BEFORE training:
  easy:   0.6567
  medium: 0.6234
  hard:   0.6373
  avg:    0.6391

AFTER GRPO training:
  easy:   0.5247  ← WORSE
  medium: 0.4704  ← WORSE
  hard:   0.4514  ← WORSE
  avg:    0.4822  ← 15% WORSE!
```

### Why Did This Happen? Let's Look at the Model's Output

**Before training (untrained Qwen 3B):**
```json
{"intent": "reschedule_event", "owner": "work", "priority": "high",
 "proposed_slot": "after 18:00", "needs_clarification": false,
 "message_template": "I suggest rescheduling the review to after 18:00"}
```
Clean JSON. Reasonable decision. Score: ~0.65

**After GRPO-only training:**
```
Based on my analysis of this scheduling conflict, I believe the best
course of action would be to reschedule the event. Here is my recommendation:

{"intent": "reschedule_event", "owner": "work", "priority":
```
JSON buried in text, cut off at 96 tokens. Parse fails. Falls back to heuristic. Score: ~0.48

### The Problem in Detail

```python
# What GRPO was seeing during training:
prompt = "Conflict: dinner overlaps with review..."

# Model generates 8 responses. Most are bad:
response_1 = '{"intent": "resc...'           # truncated → reward = 0.0
response_2 = 'I think we should...'          # no JSON → reward = 0.0
response_3 = '{"intent": "route_message"...' # wrong intent → reward = 0.3
response_4 = 'Here is my analysis...'        # no JSON → reward = 0.0
response_5 = '{"intent": "reschedule"...'    # partial match → reward = 0.4
response_6 = 'The conflict requires...'      # no JSON → reward = 0.0
response_7 = '{"intent": "delegate"...'      # wrong intent → reward = 0.2
response_8 = '{"intent": "reschedule"...'    # partial match → reward = 0.4

# GRPO picks response_5 as "best" (0.4) and reinforces it
# But 0.4 is still a BAD response!
# The model learns to produce slightly-less-bad responses
# instead of learning to produce GOOD responses
```

This is like grading a class where everyone fails — the "best" student gets a 40%, and you tell everyone to be more like them. That doesn't produce good students.

---

## 5B.4 The Fix: SFT First

### Step 1: Build the SFT Training Data

We take the correct answers directly from `conflict_cases.json`:

```python
import json

sft_data = []

for task_id in ["easy_evening_planner", "medium_multi_party_negotiation", "hard_cascade_replanning"]:
    env = PersonalAssistantConflictEnv()
    result = await env.reset(task_name=task_id)
    
    while not result.done and result.observation.current_conflict:
        conflict = result.observation.current_conflict
        expected = conflict.expected  # The answer key!
        
        # Build the prompt (same prompt we use everywhere)
        prompt = (
            "Return ONLY a JSON object, nothing else.\n"
            '{"intent": "...", "owner": "...", "priority": "...", '
            '"proposed_slot": "...", "needs_clarification": false, '
            '"message_template": "..."}\n\n'
            f"Conflict: {conflict.summary}\n"
            f"Constraints: {', '.join(conflict.constraints)}\n"
            "Intents: route_message/propose_plan/reschedule_event/"
            "delegate_task/ask_clarification/finalize_itinerary\n"
            "Owners: self/work/family/travel/finance/legal\n"
            "Priorities: low/normal/high/urgent\nJSON:"
        )
        
        # Build the CORRECT answer from the expected values
        correct_answer = json.dumps({
            "intent": expected.intent.value,       # e.g., "reschedule_event"
            "owner": expected.owner.value,          # e.g., "work"
            "priority": expected.priority.value,    # e.g., "high"
            "proposed_slot": expected.expected_slot_hint or "",
            "needs_clarification": expected.block_if_missing_context,
            "message_template": f"{expected.intent.value.replace('_', ' ')}: "
                               f"{', '.join(expected.required_keywords)}. Handle with care."
        })
        
        # Format as a chat conversation
        # The model sees: User asks about conflict → Assistant responds with correct JSON
        conversation = tokenizer.apply_chat_template([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": correct_answer}
        ], tokenize=False)
        
        sft_data.append({"text": conversation})
        
        # Advance to next conflict
        result = await env.step(heuristic_action(result.observation))
```

**What does one training example look like?**

```
<|im_start|>user
Return ONLY a JSON object, nothing else.
Conflict: Family dinner reservation at 19:30 overlaps with incident review call
Constraints: dinner cannot move earlier, review is mandatory
JSON:<|im_end|>
<|im_start|>assistant
{"intent": "reschedule_event", "owner": "work", "priority": "high",
 "proposed_slot": "after 20:30", "needs_clarification": false,
 "message_template": "reschedule event: reschedule, incident, review. Handle with care."}<|im_end|>
```

The model learns: "When a user asks about this conflict, output this exact JSON."

### Step 2: Why We Repeat the Data

```python
# We only have 15 unique examples (one per conflict)
# That's too few for training — the model needs more repetitions
sft_data = sft_data * 15  # 15 × 15 = 225 training examples
```

**Why 15 repetitions?**
- Too few repetitions (1-3): Model doesn't fully memorize the format
- Good range (10-20): Model reliably reproduces correct JSON
- Too many (50+): Wastes training time, no benefit

Think of it like flashcards — you need to see each card multiple times to remember it.

### Step 3: Configure SFT Training

```python
from trl import SFTConfig, SFTTrainer
from datasets import Dataset

sft_dataset = Dataset.from_list(sft_data)

sft_trainer = SFTTrainer(
    model=model,                    # Our Qwen 3B with LoRA
    train_dataset=sft_dataset,      # The 225 examples
    args=SFTConfig(
        output_dir="./sft_output",
        per_device_train_batch_size=4,    # Process 4 examples at once
        gradient_accumulation_steps=2,     # Accumulate 2 batches before updating
        num_train_epochs=3,               # Go through all data 3 times
        learning_rate=2e-4,               # How fast to learn (higher than GRPO!)
        max_seq_length=608,               # Max tokens per example
        logging_steps=10,                 # Print progress every 10 steps
        warmup_steps=5,                   # Start with tiny learning rate, ramp up
        seed=42,                          # Reproducibility
    ),
    processing_class=tokenizer,
)
```

**Why different settings than GRPO?**

| Setting | SFT | GRPO | Why Different |
|---|---|---|---|
| `learning_rate` | 2e-4 (high) | 1e-5 (low) | SFT can learn aggressively — correct answers are guaranteed |
| `batch_size` | 4 | 1 | SFT is simpler, can handle bigger batches |
| `epochs` | 3 | 1 | SFT data is small, needs more passes |
| `max_seq_length` | 608 | 512 prompt + 96 completion | SFT sees full conversation, GRPO separates prompt and completion |

### Step 4: Run SFT Training

```python
print("SFT training (teaching correct answers)...")
sft_trainer.train()
print("SFT complete!")
```

**What you see during SFT training:**

```
Step  10: loss = 2.3456    ← model is learning
Step  20: loss = 0.8901    ← getting better
Step  30: loss = 0.3456    ← almost there
Step  40: loss = 0.1234    ← converged!
Step  50: loss = 0.0567    ← memorized the answers
```

**What "loss" means in SFT:**
- High loss (2+): Model's predictions are very different from correct answers
- Medium loss (0.5-1): Model is getting close
- Low loss (< 0.2): Model reliably produces correct answers

Unlike GRPO's "reward" (higher = better), SFT's "loss" goes DOWN (lower = better).

### Step 5: Verify SFT Worked

```python
# Switch model to inference mode
FastLanguageModel.for_inference(model)

# Test on one conflict
env = PersonalAssistantConflictEnv()
r = await env.reset(task_name="easy_evening_planner")
conflict = r.observation.current_conflict

prompt = f"Return ONLY a JSON object...\nConflict: {conflict.summary}\n..."
msgs = [{"role": "user", "content": prompt}]
inp = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
out = model.generate(inp, max_new_tokens=200, temperature=0.05, do_sample=True)
text = tokenizer.decode(out[0][inp.shape[1]:], skip_special_tokens=True)

print(text)
# BEFORE SFT: "Based on analysis, I recommend..." (garbage)
# AFTER SFT:  '{"intent": "reschedule_event", "owner": "work", ...}' (clean JSON!)
```

---

## 5B.5 Stage 2: GRPO on Top of SFT

Now that the model knows how to produce valid JSON, GRPO can actually work:

```python
from trl import GRPOConfig, GRPOTrainer

grpo_trainer = GRPOTrainer(
    model=model,           # Same model, now with SFT knowledge
    reward_funcs=reward_fn, # Environment-backed reward function
    args=GRPOConfig(
        output_dir="./grpo_out",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,        # VERY low — don't break what SFT taught
        num_train_epochs=1,         # Just 1 epoch — gentle refinement
        max_prompt_length=512,
        max_completion_length=96,
        logging_steps=5,
        seed=42,
    ),
    train_dataset=grpo_dataset,
    processing_class=tokenizer,
)
```

**Why conservative settings?**

| Setting | Why |
|---|---|
| `learning_rate=1e-5` | 20x lower than SFT! We don't want to undo SFT's work |
| `num_train_epochs=1` | One pass is enough — model already knows the format |
| `max_completion_length=96` | JSON only needs ~80 tokens |

**What GRPO does differently AFTER SFT:**

```python
# BEFORE SFT: model generates 8 responses, most are garbage
responses = [
    '{"intent": "resc...',           # truncated
    'I think we should...',          # no JSON
    '{"intent": "route_message"...',  # wrong
    'Here is my analysis...',        # no JSON
    # ... mostly unusable
]
# Best response scores 0.3-0.4. Not helpful.

# AFTER SFT: model generates 8 responses, ALL are valid JSON!
responses = [
    '{"intent": "reschedule_event", "owner": "work", ...}',      # score: 0.95
    '{"intent": "reschedule_event", "owner": "travel", ...}',    # score: 0.75
    '{"intent": "delegate_task", "owner": "work", ...}',         # score: 0.45
    '{"intent": "reschedule_event", "owner": "work", ...}',      # score: 0.92
    '{"intent": "reschedule_event", "owner": "family", ...}',    # score: 0.70
    '{"intent": "propose_plan", "owner": "work", ...}',          # score: 0.55
    '{"intent": "reschedule_event", "owner": "work", ...}',      # score: 0.90
    '{"intent": "reschedule_event", "owner": "self", ...}',      # score: 0.60
]
# GRPO reinforces the 0.95 response. NOW it's learning real decisions!
```

---

## 5B.6 The Complete Two-Stage Code

Here's the entire pipeline in one place:

```python
# =====================================================
# STAGE 1: SFT — Teach the format
# =====================================================

# 1. Build training data from correct answers
sft_data = []
for tid in TASK_IDS:
    env = PersonalAssistantConflictEnv()
    r = await env.reset(task_name=tid)
    while not r.done and r.observation.current_conflict:
        c = r.observation.current_conflict
        prompt = build_prompt(c)                    # conflict → prompt string
        answer = build_correct_answer(c.expected)   # expected → JSON string
        text = tokenizer.apply_chat_template([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ], tokenize=False)
        sft_data.append({"text": text})
        r = await env.step(heuristic_action(r.observation))

sft_data = sft_data * 3  # Light repetition; main signal comes from dataset diversity

# 2. Train SFT
sft_trainer = SFTTrainer(
    model=model,
    train_dataset=Dataset.from_list(sft_data),
    args=SFTConfig(
        per_device_train_batch_size=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        max_seq_length=608,
    ),
    processing_class=tokenizer,
)
sft_trainer.train()

# 3. Verify: model should now produce clean JSON
FastLanguageModel.for_inference(model)
sft_scores = await eval_model(model, tokenizer, "SFT")
# Expected: ~1.0 on all tasks

# =====================================================
# STAGE 2: GRPO — Optimize decisions
# =====================================================

# 4. Collect prompts
prompt_rows = await collect_prompts(episodes=30)

# 5. Train GRPO (conservative!)
grpo_trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_fn,
    args=GRPOConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,      # 20x lower than SFT!
        num_train_epochs=1,       # Just 1 pass
        max_completion_length=96,
    ),
    train_dataset=Dataset.from_list(prompt_rows),
    processing_class=tokenizer,
)
grpo_trainer.train()

# 6. Final evaluation
FastLanguageModel.for_inference(model)
final_scores = await eval_model(model, tokenizer, "SFT+GRPO")
# Expected: ~1.0 maintained
```

---

## 5B.7 Understanding the Libraries

### TRL (Transformer Reinforcement Learning)

```python
pip install trl
```

TRL provides three trainers we care about:

| Trainer | What It Does | When To Use |
|---|---|---|
| `SFTTrainer` | Supervised fine-tuning on text pairs | Teaching format + correct answers |
| `GRPOTrainer` | RL with group relative policy optimization | Optimizing decisions with rewards |
| `DPOTrainer` | Direct preference optimization | When you have "better vs worse" pairs |

**We use SFTTrainer then GRPOTrainer.** DPO would require pairs of good/bad responses which we don't have.

```python
# SFTTrainer needs: model + dataset with "text" column
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,  # Must have "text" column
    args=SFTConfig(...),
    processing_class=tokenizer,
)
trainer.train()

# GRPOTrainer needs: model + reward function + dataset with "prompt" column
from trl import GRPOTrainer, GRPOConfig

trainer = GRPOTrainer(
    model=model,
    reward_funcs=my_reward_function,  # Function that returns list of floats
    train_dataset=dataset,             # Must have "prompt" column
    args=GRPOConfig(...),
    processing_class=tokenizer,
)
trainer.train()
```

### Unsloth

```python
pip install unsloth
```

Unsloth makes everything 2x faster and use 60% less memory. Here's what it does:

```python
from unsloth import FastLanguageModel

# Without Unsloth:
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
# Takes 12GB VRAM, slow training

# With Unsloth:
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",  # Pre-quantized!
    max_seq_length=608,
    load_in_4bit=True,      # 4-bit = 4x less memory
)
# Takes 4GB VRAM, 2x faster training
```

**What Unsloth actually does under the hood:**
1. **4-bit quantization**: Stores each weight as 4 bits instead of 16 bits. 3B params × 4 bits = 1.5GB vs 6GB
2. **Fused kernels**: Combines multiple GPU operations into one, reducing overhead
3. **Smart gradient offloading**: Moves gradients to CPU when GPU is full
4. **Optimized LoRA**: Faster adapter computation

### LoRA (Low-Rank Adaptation)

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,              # Rank of the adapter matrices
    lora_alpha=16,     # Scaling factor
    target_modules=[   # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",   # Attention
        "gate_proj", "up_proj", "down_proj"         # Feed-forward
    ],
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)
```

**What LoRA does:**

```
Original layer: y = Wx          (W is 3072 × 3072 = 9.4M params)

LoRA layer:     y = Wx + BAx    (B is 3072 × 16, A is 16 × 3072 = 98K params)
                                 ↑           ↑
                              tiny!       tiny!
```

Instead of updating the full 9.4M parameter matrix W, LoRA adds two tiny matrices B (3072×16) and A (16×3072). Only B and A are trained. This is 96x fewer parameters!

**What `r=16` means:**
- `r=4`: Very small adapter. Fast but limited capacity.
- `r=16`: Good balance. Our choice. 29M trainable params.
- `r=64`: Large adapter. More capacity but slower, more memory.

**What `target_modules` means:**

A transformer has two main blocks per layer:

```
Attention block:
  q_proj → creates "query" (what am I looking for?)
  k_proj → creates "key" (what do I contain?)
  v_proj → creates "value" (what's my actual content?)
  o_proj → combines attention outputs

Feed-forward block:
  gate_proj → controls information flow
  up_proj   → expands dimension
  down_proj → compresses back down
```

We add LoRA to ALL of them. More adapters = more learning capacity.

### Datasets (HuggingFace)

```python
pip install datasets
```

Simple library to create training datasets:

```python
from datasets import Dataset

# From a list of dicts
data = [
    {"text": "example 1"},
    {"text": "example 2"},
]
dataset = Dataset.from_list(data)

# Access
print(dataset[0])          # {"text": "example 1"}
print(len(dataset))         # 2
print(dataset.column_names) # ["text"]
```

**Why not just use a Python list?** The `Dataset` class handles:
- Efficient memory mapping (doesn't load everything into RAM)
- Automatic batching and shuffling
- Integration with TRL trainers
- Caching and checkpointing

---

## 5B.8 The Full Results Comparison

For the measured numbers from a Colab T4 run on procedurally-novel holdout episodes, see the **Training results** section in `README.md`. Headline:

| pool                | untrained 3B | after SFT | after SFT + GRPO |
|---------------------|--------------|-----------|------------------|
| holdout (n=40)      | 0.5454       | 0.9877    | 0.9876           |
| adversarial (n=10)  | —            | —         | 0.9885           |

**Key observations:**
1. GRPO alone (without SFT) tends to collapse output format on this task — SFT is needed first to teach the JSON schema.
2. SFT lifts holdout from 0.5454 to 0.9877 (+0.4423 absolute) on episodes the model has not seen during training.
3. GRPO ≈ SFT here (0.9876 vs 0.9877). On hard procedural data the SFT stage is doing most of the lifting; GRPO holds the score steady.
4. The model also holds 0.9885 on the adversarial probe pool, where the slot grader is regex-strict and message scoring penalizes keyword stuffing.

---

## 5B.9 When Does GRPO Actually Matter?

On a small, fixed task set with strong demonstrations, SFT can do most of the lifting. GRPO becomes essential when:

**Scenario 1: A wide procedural pool**
```python
# Procedural train pool: 1000+ unique episodes (seeds 1000-1999)
# Holdout pool:          100 unique episodes (seeds 9000-9099, never seen)

# SFT teaches the format and intent routing on diverse demonstrations.
# GRPO learns STRATEGIES against the env's real reward — useful when the
# correct action is a function of state, not a lookup.
# GRPO generalizes when no demonstration covers the exact case.
```

**Scenario 2: Dynamic conflicts (production)**
```python
# Currently: same conflicts every time
# Production: real calendar data, new conflicts daily

# SFT: useless — never seen these specific conflicts
# GRPO: useful — learned general conflict resolution strategies
```

**Scenario 3: Bigger model + more compute**
```python
# Currently: 3B model on T4 (limited)
# Onsite: 7B or 14B on A100 (powerful)

# Bigger models benefit MORE from GRPO
# because they can learn more complex strategies
# that go beyond simple memorization
```

---

## 5B.10 The ChatGPT/InstructGPT Parallel

Our two-stage pipeline mirrors how the most successful AI systems are built:

```
InstructGPT (2022):
────────────────────
1. Start with GPT-3 (base model)
2. SFT on human-written demonstrations (13K examples)
3. Train reward model on human preferences
4. RLHF (PPO) to optimize against reward model
Result: ChatGPT — the model that changed everything

Our Pipeline (2026):
────────────────────
1. Start with Qwen 3B (base model)
2. SFT on correct conflict resolutions (15 examples × 15 repeats)
3. Reward model = our environment's grader (deterministic!)
4. GRPO to optimize against environment rewards
Result: A model that resolves scheduling conflicts

Same recipe:
  SFT teaches FORMAT → RL optimizes QUALITY
```

**Why this recipe works:**
1. Base models know language but not your task
2. SFT teaches them your specific task format
3. RL explores beyond the SFT examples to find even better strategies

**The insight: RL doesn't replace supervised learning — it builds on top of it.**

---

## 5B.11 Key Takeaways

| # | Lesson | Detail |
|---|---|---|
| 1 | **Never skip SFT** | Model needs to know the output format before RL can optimize |
| 2 | **Training reward ≠ eval score** | Rising GRPO reward doesn't guarantee better real performance |
| 3 | **SFT + RL > either alone** | SFT = foundation, RL = optimization |
| 4 | **Conservative GRPO after SFT** | Low learning rate (1e-5), few epochs — don't break SFT |
| 5 | **"Overfitting" in RL ≠ bad** | Solving the environment perfectly IS the goal |
| 6 | **SFT carries small environments** | When demonstrations cover most of the state space, SFT is enough |
| 7 | **GRPO matters at scale** | Procedural data, unseen episodes, fuzzier rewards |
| 8 | **Same recipe as ChatGPT** | SFT → RL is the universal LLM training pipeline |

---

## What's Next?

You now understand the complete training pipeline — why SFT comes first, why GRPO builds on top, and how the code works line by line. Chapter 6 covers the inference script: how we run the trained model and log results.
