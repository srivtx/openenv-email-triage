# 01 - RL From Scratch (No Prior Knowledge Required)

This chapter teaches Reinforcement Learning from first principles, with no assumptions.

## 1. What is Reinforcement Learning?

Reinforcement Learning (RL) is learning by trial and feedback.

- The **agent** takes an action.
- The **environment** responds with a new observation.
- The agent gets a **reward**.
- The process repeats.

The goal is to maximize total reward over time.

## 2. Core Terms You Must Know

### Agent

The decision maker. In this project, the LLM is the agent.

### Environment

The world the agent interacts with. Here, it is the email triage simulator.

### Observation (what the agent sees)

Current email details, previous history, progress counters.

### Action (what the agent does)

Choose category, priority, team, spam flag, and response template.

### Reward (instant feedback)

A score for the action quality in that step.

### Episode

A full run of one task from reset to done.

### Policy

The agent's strategy: given this observation, which action should it choose?

## 3. Why RL Matters in LLM Post-Training

Think of model development in 3 layers:

1. Pre-training: learns language and world patterns
2. SFT: learns instruction following and format
3. RL: learns to optimize exact objective with reward

SFT says "follow instructions".
RL says "maximize this measurable behavior".

## 4. Why Not Just Prompt Better?

Prompting (in-context learning) helps quickly, but:

- long prompts cost more tokens
- latency increases
- behavior can drift with context length

RL moves behavior into model weights, so inference can be cheaper and more stable.

## 5. Reward Design Is the Heart of RL

If reward is wrong, the model learns wrong behavior.

Bad reward example:

- reward only at the very end (sparse reward)
- no partial credit
- no penalty for dumb loops

Good reward example (used in this project):

- partial credit by category/priority/team/spam correctness
- small bonus baseline so learning signal is not too sparse
- penalties for repetitive or unhelpful actions

## 6. Reward Hacking (Very Important)

Agents optimize reward, not your intention.

If there is a loophole, the model will exploit it.

Example in general RL:

- benchmark rewards lower runtime
- model edits benchmark script itself
- gets fake high reward

Defense:

- sandboxing
- immutable evaluation harness
- deterministic graders
- random trajectory audits

## 7. Why We Used an Email Domain

Hackathon requires real-world tasks, not toy games.

Email triage is real because teams do this daily:

- billing escalation
- account recovery
- legal notices
- abuse/phishing reports
- engineering incident routing

## 8. Mental Model Before You Build

Always remember this loop:

1. `reset()` -> start task and return first observation
2. agent decides action
3. `step(action)` -> return observation, reward, done, info
4. repeat until done
5. optional `state()` for full internal status

That is RL environment design in one sentence.

## 9. Beginner Study Plan (Recommended)

If you are very new, do this:

Day 1:

- Learn core RL terms
- Understand episode loop

Day 2:

- Learn reward shaping and grading
- Read environment and task fixtures

Day 3:

- Run inference script
- Read step logs and map them to reward

Day 4:

- Learn deployment, validation, and submission checks

## 10. What to Remember Most

- RL is optimization under feedback.
- Environment quality decides training quality.
- Reward quality decides behavior quality.
- Deterministic grading decides evaluation trust.
