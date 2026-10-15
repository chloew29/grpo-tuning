"""Prepare GSM8K for GRPO training. Run wherever you have network (Colab).

Output: data/gsm8k_train.jsonl / data/gsm8k_test.jsonl
Each line: {"prompt": ..., "answer": "42"}  (answer = gold final number only)

GSM8K's `answer` field looks like:
    "Natalia sold 48 clips...\\n#### 48"
We take everything after #### as gold. The model never sees the reasoning —
that's the point of GRPO: it has to discover reasoning itself.
"""
import argparse
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SYSTEM = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> "
    "and <answer> </answer> tags, respectively."
)


def gold_number(answer_text: str) -> str:
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
    if not m:
        raise ValueError(f"no #### answer in: {answer_text[:80]}")
    return m.group(1).replace(",", "")


def build_prompt(question: str) -> str:
    return f"{SYSTEM}\n\nUser: {question}\nAssistant:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_n", type=int, default=2000,
                    help="cap train rows (T4-friendly default)")
    ap.add_argument("--test_n", type=int, default=300)
    args = ap.parse_args()

    from datasets import load_dataset  # import here: only needed on the GPU box
    ds = load_dataset("openai/gsm8k", "main")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for split, n in [("train", args.train_n), ("test", args.test_n)]:
        out = []
        for ex in ds[split].select(range(min(n, len(ds[split])))):
            out.append({"prompt": build_prompt(ex["question"]),
                        "answer": gold_number(ex["answer"])})
        p = DATA_DIR / f"gsm8k_{split}.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in out) + "\n")
        print(f"wrote {p} ({len(out)} rows)")


if __name__ == "__main__":
    main()
