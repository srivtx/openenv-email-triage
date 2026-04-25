# Chapter 5: Training — SFT + GRPO (The Two-Stage Pipeline)

This chapter explains our full training pipeline. We learned the hard way that GRPO alone doesn't work — you need SFT first. This chapter covers both stages, all the libraries, and what actually happens during training.

> **note:** the shipped notebook (`notebooks/train_grpo_colab.ipynb`) sources SFT data from a procedural train pool of unique episodes, uses the env's real reward as the GRPO reward, and evaluates on a disjoint procedural holdout pool plus a small adversarial probe pool. see `src/assistant_conflict_env/eval_set.py` for the seed split.

---

## 5.1 The Training Problem

We have:
- An environment that scores decisions (reward = 0.0 to 1.0)
- A model that can generate JSON actions (Qwen 3B)
- How do we make the model generate BETTER actions over time?

The answer: **SFT first, then GRPO.** Not just GRPO alone — we tried that, and it made things worse (see section 5.10).

---

## 5.2 What is GRPO?

**GRPO = Group Relative Policy Optimization**

Forget the fancy name. Here's what it does:

```
Step 1: Give the model a conflict
Step 2: Model generates 8 different responses (not just 1!)
Step 3: Score all 8 with the environment
Step 4: Compare: "Response 3 scored 0.9, Response 7 scored 0.2"
Step 5: Update model: "Generate more things like Response 3, less like Response 7"
```

That's it. Generate multiple, score all, reinforce the best, suppress the worst.

**Why "Group"?** Because it compares a GROUP of responses, not just one.

**Why "Relative"?** Because it doesn't need an absolute "good" threshold — it just makes the best response more likely relative to the worst.

---

## 5.3 GRPO vs Other RL Algorithms

| Algorithm | How It Works | Needs | Complexity |
|---|---|---|---|
| **PPO** | Trains a separate "critic" network to estimate future rewards | Critic model + value head | High |
| **DPO** | Uses pairs of good/bad examples (no environment needed) | Preference dataset | Medium |
| **GRPO** | Generates group of responses, compares rewards | Just a reward function | **Low** ✅ |

**Why GRPO for us?** 
- No need for a separate critic model (saves GPU memory)
- Works directly with our environment's reward function
- Designed for LLMs specifically
- Supported by TRL library out of the box

---

## 5.4 The Libraries We Use

### TRL (Transformer Reinforcement Learning)
```
pip install trl
```
**What**: HuggingFace's library for training LLMs with RL. Provides `GRPOTrainer`.

**Why**: It handles all the complex RL math — KL divergence, advantage estimation, policy updates. We just provide the model and reward function.

**Without TRL**: We'd need to write ~2000 lines of custom training code including gradient computation, KL regularization, reward normalization. TRL does it in ~20 lines.

### Unsloth
```
pip install unsloth
```
**What**: Makes training 2x faster and uses 60% less memory.

**Why**: Qwen 3B has 3 billion parameters. Without Unsloth, it won't fit on a free Colab T4 GPU (16GB VRAM). Unsloth applies:
- **4-bit quantization**: Stores weights as 4-bit numbers instead of 16-bit (4x less memory)
- **LoRA**: Only trains 1% of parameters (the rest are frozen)
- **Gradient checkpointing**: Trades compute for memory

**Without Unsloth**: You'd need a 40GB GPU (A100) to train this model. With Unsloth: 16GB T4 works.

### LoRA (Low-Rank Adaptation)
**What**: Instead of training all 3 billion parameters, train small "adapter" matrices attached to each layer.

```
Full model:   3,115,872,256 parameters (frozen)
LoRA adapter:    29,933,568 parameters (trainable)
                              ↑
                     Only 0.96% of the model!
```

**Why**: Training all parameters needs enormous memory and data. LoRA trains just enough to learn new behaviors while keeping the model's existing knowledge.

**Analogy**: Instead of rebuilding an entire car engine, you just add a turbocharger. Small change, big impact.

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,              # LoRA rank — higher = more capacity, more memory
    lora_alpha=16,     # Scaling factor
    target_modules=[   # Which layers to add LoRA to
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj"        # Feed-forward layers
    ],
    lora_dropout=0,                    # No dropout (we want deterministic)
    use_gradient_checkpointing="unsloth"  # Memory optimization
)
```

---

## 5.5 The Training Loop — Step by Step

Here's exactly what happens during training:

### Step 1: Prepare the Prompt Dataset

```python
prompts = []
for each task:
    reset environment
    for each conflict:
        format the conflict as a prompt string
        prompts.append(prompt)

# Result: 150 prompts like:
# "Conflict: Board review overlaps school pickup.
#  Constraints: board review mandatory, pickup must be covered
#  Return JSON with intent, owner, priority..."
```

### Step 2: Model Generates Responses

For EACH prompt, the model generates **8 different responses** (called `num_generations`):

```
Prompt: "Conflict: dinner overlaps with check-in..."

Response 1: {"intent": "reschedule_event", "owner": "travel", ...}  
Response 2: {"intent": "delegate_task", "owner": "family", ...}    
Response 3: {"intent": "reschedule_event", "owner": "work", ...}    
Response 4: {"intent": "ask_clarification", "owner": "self", ...}   
Response 5: {"intent": "reschedule_event", "owner": "travel", ...}  
Response 6: {"intent": "route_message", "owner": "travel", ...}     
Response 7: {"intent": "propose_plan", "owner": "travel", ...}      
Response 8: {"intent": "reschedule_event", "owner": "travel", ...}  
```

### Step 3: Score All Responses

Each response is fed to the environment's reward function:

```
Response 1: reward = 0.82
Response 2: reward = 0.31
Response 3: reward = 0.65
Response 4: reward = 0.10
Response 5: reward = 0.82
Response 6: reward = 0.40
Response 7: reward = 0.55
Response 8: reward = 0.82
```

### Step 4: Compute Advantages

GRPO calculates the **advantage** = how much better/worse each response is compared to the group average.

```
Group average = 0.56

Response 1: advantage = 0.82 - 0.56 = +0.26 (above average → reinforce!)
Response 2: advantage = 0.31 - 0.56 = -0.25 (below average → suppress!)
Response 3: advantage = 0.65 - 0.56 = +0.09 (slightly above → mild reinforce)
Response 4: advantage = 0.10 - 0.56 = -0.46 (way below → strongly suppress!)
...
```

### Step 5: Update Model Weights

The model's weights are adjusted so that:
- High-advantage responses become MORE likely
- Low-advantage responses become LESS likely

```
Before training: P(Response 1) = 12%, P(Response 4) = 12%
After training:  P(Response 1) = 18%, P(Response 4) = 6%
```

### Step 6: Repeat

Do this for every prompt in the dataset. That's one epoch. The model gets better each time.

---

## 5.6 KL Divergence — Don't Change Too Much

One risk: the model changes so much during training that it "forgets" how to speak coherently. KL divergence is a leash that prevents this.

```
KL divergence = how different is the trained model from the original model?

Low KL (0.001):  Model barely changed — learning too slowly
High KL (1.0):   Model changed drastically — might be incoherent
Good KL (0.01):  Model learned new behaviors but still speaks normally
```

GRPO automatically adds a KL penalty to prevent the model from drifting too far from its original behavior.

---

## 5.7 Config — SFT vs GRPO Settings

The two stages use very different settings:

```python
# Stage 1: SFT — aggressive learning (correct answers are guaranteed)
SFTConfig(
    per_device_train_batch_size=4,    # Bigger batches OK
    num_train_epochs=3,               # Multiple passes over small data
    learning_rate=2e-4,               # High — we KNOW the answers are correct
    max_seq_length=608,               # Full conversation fits
)

# Stage 2: GRPO — conservative learning (don't break what SFT taught)
GRPOConfig(
    per_device_train_batch_size=1,    # Smaller — GRPO needs more memory per sample
    gradient_accumulation_steps=4,    # Accumulate for stability
    learning_rate=1e-5,               # 20x LOWER than SFT — gentle refinement
    num_train_epochs=1,               # Just 1 pass — don't overdo it
    max_prompt_length=512,            # Prompt only
    max_completion_length=96,         # JSON needs ~80 tokens max
)
```

**Why such different learning rates?**
- SFT at 2e-4: we're showing correct answers, so learn fast
- GRPO at 1e-5: we're exploring, so be careful not to break the SFT foundation

---

## 5.8 Reading Training Logs

### SFT Logs

```
Step  10: loss = 2.3456    ← model learning the format
Step  20: loss = 0.8901    ← getting closer
Step  30: loss = 0.3456    ← almost memorized
Step  40: loss = 0.1234    ← converged
Step  50: loss = 0.0567    ← knows the answers
```

SFT loss goes DOWN (lower = better).

### GRPO Logs

```
Step  Training Loss  reward    reward_std   kl
5     0.000000       0.257     0.244        0.000033
10    0.000002       0.319     0.240        0.001737
15    0.000020       0.382     0.199        0.019625
20    0.000044       0.416     0.214        0.043821
25    0.000058       0.429     0.191        0.057826
30    0.000066       0.418     0.197        0.065947
35    0.000105       0.441     0.181        0.104626
40    0.000094       0.446     0.175        0.093697
45    0.000075       0.461     0.169        0.074863
50    0.000087       0.459     0.173        0.086899
```

GRPO reward goes UP (higher = better).

| Column | What It Means | Good Sign |
|---|---|---|
| `reward` | Average reward across the batch | Going UP over time |
| `reward_std` | Variance in rewards | Going DOWN (more consistent) |
| `kl` | How much model changed from original | Slowly increasing, stays < 0.1 |

---

## 5.9 What the Trained Model Looks Like

After training, the model files are saved:

```
trained_conflict_resolver/
├── adapter_config.json      # LoRA configuration
├── adapter_model.safetensors # The trained LoRA weights (small!)
├── tokenizer_config.json    # How text is tokenized
├── tokenizer.json           # Token mappings
└── special_tokens_map.json  # Special token definitions
```

The LoRA adapter is tiny (~50MB) compared to the full model (2GB). You can share just the adapter, and anyone with the base Qwen 3B model can use your trained version.

---

## 5.10 What We Learned: GRPO Alone Fails

Our first attempt used GRPO directly on the untrained model. The training reward went up (0.26 → 0.46) but evaluation scores went DOWN:

| Model | Easy | Medium | Hard | Average |
|---|---|---|---|---|
| Untrained 3B | 0.5613 | 0.6346 | 0.4741 | **0.5567** |
| GRPO only | 0.5247 | 0.4704 | 0.4514 | **0.4822** (worse!) |
| SFT only | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| SFT + GRPO | 1.0000 | 1.0000 | 1.0000 | **1.0000** |

**Why GRPO alone failed**: the model couldn't produce valid JSON, so the "best" of 8 garbage responses still scored low. GRPO reinforced slightly-less-bad garbage instead of learning good answers.

**Why SFT fixed it**: SFT taught the model the correct JSON format first. After SFT, all 8 GRPO responses are valid JSON — now GRPO can actually compare good vs bad decisions instead of garbage vs garbage.

This is the same pattern as ChatGPT: SFT on demonstrations first, then RL for optimization.

See **Chapter 5B** for the full deep-dive with code.

---

## What's Next?

Chapter 5B is a deep-dive into why SFT before GRPO matters, with complete code examples. Chapter 6 covers the inference script — how we actually RUN the model and log results.
