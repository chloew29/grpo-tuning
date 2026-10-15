"""Reward functions for GRPO on GSM8K. Pure Python — testable without a GPU.

The whole trick of GRPO (DeepSeek-R1): no human-written reasoning traces needed.
For each question we sample a GROUP of completions, score each with these
rewards, and reinforce the ones that beat the group average. The model learns
to reason because only completions that actually get the right answer win.
"""
import re


def extract_number(text: str):
    """Pull the final numeric answer out of a completion.

    Tries, in order: <answer> tags, \\boxed{}, then the last standalone number.
    Returns float or None.
    """
    m = re.search(r"<answer>(.*?)</answer>", text, re.S)
    if m:
        text = m.group(1)
    else:
        m = re.search(r"\\boxed\{([^}]*)\}", text)
        if m:
            text = m.group(1)
    # last standalone number (handles commas, decimals, negatives)
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def correctness_reward(prompts, completions, answer, **kwargs) -> list:
    """1.0 if the extracted number matches gold, else 0.0.

    `answer` is the list of gold final answers (strings) aligned with completions.
    """
    rewards = []
    for completion, gold in zip(completions, answer):
        pred = extract_number(completion)
        try:
            gold_num = float(str(gold).replace(",", ""))
        except ValueError:
            gold_num = None
        rewards.append(1.0 if (pred is not None and gold_num is not None
                               and abs(pred - gold_num) < 1e-6) else 0.0)
    return rewards


def format_reward(completions, **kwargs) -> list:
    """0.5 for well-formed <think>/<answer> structure, else 0.0.

    Keeps the model honest about showing its work in the expected format.
    Weighted lower than correctness — format is hygiene, not intelligence.
    """
    rewards = []
    for completion in completions:
        has_think = bool(re.search(r"<think>.*?</think>", completion, re.S))
        has_answer = bool(re.search(r"<answer>.*?</answer>", completion, re.S))
        rewards.append(0.5 if (has_think and has_answer) else 0.0)
    return rewards


def combined_reward(prompts, completions, answer, **kwargs) -> list:
    """What GRPOTrainer actually optimizes: correctness + format."""
    c = correctness_reward(prompts, completions, answer, **kwargs)
    f = format_reward(completions, **kwargs)
    return [ci + fi for ci, fi in zip(c, f)]
