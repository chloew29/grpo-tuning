# grpo-tuning — Teach a Small LLM to Reason, DeepSeek-R1 Style

Fine-tune `Qwen2.5-1.5B-Instruct` on GSM8K math problems with **GRPO**
(Group Relative Policy Optimization) — the reinforcement-learning method
behind DeepSeek-R1. Runs on a free Colab T4.

## The idea in one paragraph

SFT teaches a model to *imitate* answers. DPO teaches it to *prefer* better
answers. GRPO skips human-written reasoning entirely: for each question the
model samples a **group** of completions, each is scored by a reward function
(did it get the number right? is the format sane?), and the policy is pushed
toward completions that beat the **group average** — no critic model needed.
Reasoning emerges because only completions that actually solve the problem
get reinforced.

## Quickstart (Colab, free T4)

```python
# 1. runtime: T4 GPU
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -r requirements.txt
```

```bash
# 2. data (needs network once)
python src/data_prep.py --train_n 2000 --test_n 300

# 3. baseline: how dumb is the base model? (optional but honest)
python src/evaluate.py --n 100

# 4. train (~1-2h on T4 for 200 steps)
python src/train_grpo.py --steps 200

# 5. did it learn to reason?
python src/evaluate.py --n 100 --adapter outputs/grpo-qwen-gsm8k
```

## The reward is the whole game

`src/rewards.py` (pure Python, tested without a GPU):

- `correctness_reward` — 1.0 if the final number matches gold, else 0.0.
  Extracts from `<answer>` tags, `\boxed{}`, or the last number as fallback.
- `format_reward` — 0.5 for well-formed `<think>/<answer>` structure.
  Deliberately weighted lower: format is hygiene, not intelligence.

If the reward is wrong, GRPO learns the wrong thing fast — reward design,
not the optimizer, is where the skill lives.

## Files

```
src/
  rewards.py     # reward functions (+ unit-testable, no GPU needed)
  data_prep.py   # GSM8K -> {prompt, answer} jsonl
  train_grpo.py  # Unsloth + TRL GRPOTrainer (the actual training)
  evaluate.py    # greedy-decode accuracy, base vs tuned
```

## Honest limitations

- 1.5B + GSM8K is a teaching demo, not a frontier result. Expect modest gains.
- `format_reward` can be reward-hacked (pretty wrong answers); watch the
  *correctness* curve, not the total.
- GRPO is sensitive to `beta` (KL penalty) and group size — the defaults are
  a T4 starting point, not gospel.
- Training needs a GPU; everything else in this repo runs anywhere.
