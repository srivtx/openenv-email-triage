# Chapter 6: Inference & Logging — Running the Model

Inference is when you actually USE the model against the environment and record the results. This chapter explains the inference script, how it talks to the LLM, and why the logging format matters.

---

## 6.1 What Inference Means

**Training** = model learns by trying many actions and getting rewards
**Inference** = model runs once on each task to produce a final score

Think of it like:
- Training = practicing basketball shots for hours
- Inference = the actual game

---

## 6.2 How Our Inference Works

```python
# Simplified version of inference.py

# 1. Connect to a remote LLM via API
API_URL = "https://router.huggingface.co/v1"
MODEL = "Qwen/Qwen2.5-72B-Instruct"

# 2. For each task...
for task_id in ["easy", "medium", "hard"]:
    
    # 3. Reset the environment
    env = PersonalAssistantConflictEnv()
    result = await env.reset(task_name=task_id)
    
    # 4. While there are conflicts to resolve...
    while not result.done:
        conflict = result.observation.current_conflict
        
        # 5. Build a prompt describing the conflict
        prompt = f"Resolve this conflict: {conflict.summary}"
        
        # 6. Send to LLM, get response
        response = call_llm(prompt)  # Returns JSON string
        
        # 7. Parse the JSON into an action
        action = parse_json_to_action(response)
        
        # 8. Send action to environment, get reward
        result = await env.step(action)
        
        # 9. Log the step
        print(f"[STEP] step={step} reward={result.reward}")
```

---

## 6.3 The HuggingFace Router — Where the Model Lives

We don't run the model locally during inference. We call HuggingFace's API:

```python
import openai

client = openai.OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN")  # Your HuggingFace token
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-72B-Instruct",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=256,
    temperature=0.2
)

text = response.choices[0].message.content
```

**Why use the OpenAI library for HuggingFace?** HuggingFace Router uses the same API format as OpenAI. So we use the `openai` Python package but point it to HuggingFace's URL. Same interface, different backend.

**Why 72B for inference but 3B for training?**

| | Inference (72B) | Training (3B) |
|---|---|---|
| Purpose | Get best possible baseline scores | Model we can actually train |
| Runs on | HF cloud (free API) | Your GPU (Colab T4) |
| Memory needed | ~140GB (impossible locally) | ~4GB (with 4-bit) |
| Can we modify it? | No — read-only API | Yes — we update its weights |

---

## 6.4 Parsing the LLM Response

The model returns text. We need to extract JSON from it.

```python
def parse_json(text):
    # Try 1: Maybe the whole response is JSON
    try:
        return json.loads(text)
    except:
        pass
    
    # Try 2: Maybe JSON is embedded in other text
    left = text.find("{")
    right = text.rfind("}")
    if left >= 0 and right > left:
        try:
            return json.loads(text[left:right+1])
        except:
            pass
    
    # Failed to parse
    return None
```

**Why two attempts?** LLMs are unpredictable. Sometimes they return clean JSON:
```json
{"intent": "reschedule_event", "owner": "work"}
```

Sometimes they add explanation text around it:
```
Based on the conflict, I recommend:
{"intent": "reschedule_event", "owner": "work"}
This would resolve the overlap.
```

Our parser handles both cases.

---

## 6.5 The Required Log Format

The hackathon validator expects this EXACT format on stdout:

```
[START] task=easy_evening_planner env=personal_assistant_conflict_resolution model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=act(intent=reschedule_event,owner=work,priority=high,clarify=false) reward=0.82 done=false error=null
[STEP] step=2 action=act(intent=delegate_task,owner=family,priority=normal,clarify=false) reward=0.75 done=false error=null
[STEP] step=3 action=act(intent=finalize_itinerary,owner=self,priority=high,clarify=false) reward=0.90 done=true error=null
[END] success=true steps=3 score=0.82 rewards=0.82,0.75,0.90
```

**Why this format?**

| Tag | Purpose |
|---|---|
| `[START]` | Marks beginning of a task. Includes task name, env name, model. |
| `[STEP]` | Each action taken. Shows intent, owner, reward, and whether episode is done. |
| `[END]` | End of task. Shows total steps, average score, and list of all rewards. |

**What happens if format is wrong?** The validator can't parse your results → your submission fails → you get 0 points. This format is non-negotiable.

---

## 6.6 Error Handling

What if the model returns garbage?

```python
# Model returns: "I can't help with that"
payload = parse_json(response)

if payload is None:
    # Fallback: use a safe default action
    action = ConflictAction(
        intent="route_message",   # Safest default
        owner="self",
        priority="normal",
        message_template="Unable to parse model response."
    )
```

**Why not just crash?** Because the hackathon runs your inference script end-to-end. If it crashes on step 2 of the hard task, you lose ALL points for that task. A bad action with low reward is better than a crash.

---

## 6.7 Temperature — Controlling Randomness

```python
response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0.2  # Low = more deterministic
)
```

| Temperature | Behavior | When to Use |
|---|---|---|
| 0.0 | Always same response | Testing, evaluation |
| 0.2 | Mostly consistent, slight variation | Inference (our choice) |
| 0.7 | More creative, more variation | Exploration during training |
| 1.0 | Very random | Brainstorming, not for our case |

**For inference**: We use 0.2 because we want the BEST answer, not creative experiments.

**For training**: GRPO uses higher temperature internally to generate diverse responses for comparison.

---

## 6.8 The Full Inference Flow

```
┌──────────────────────────────────────────────────────────┐
│                    inference.py                          │
│                                                          │
│  ┌─────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │ Load     │    │ For each     │    │ Call HF Router │  │
│  │ tasks    │───▶│ task:        │───▶│ with prompt    │  │
│  │ from env │    │  reset env   │    │ parse JSON     │  │
│  └─────────┘    │  loop steps  │    │ create action  │  │
│                  └──────┬───────┘    └───────┬────────┘  │
│                         │                     │          │
│                         ▼                     │          │
│                  ┌──────────────┐             │          │
│                  │ env.step()   │◀────────────┘          │
│                  │ get reward   │                         │
│                  │ log [STEP]   │                         │
│                  └──────┬───────┘                         │
│                         │                                │
│                         ▼                                │
│                  ┌──────────────┐                         │
│                  │ All done?    │                         │
│                  │ Log [END]    │                         │
│                  └──────────────┘                         │
└──────────────────────────────────────────────────────────┘
```

---

## What's Next?

Chapter 7 covers **deployment** — how we package everything into a Docker container and push it to HuggingFace Spaces so the hackathon judges can access it.
