"""Evaluate base vs GRPO-tuned model on held-out GSM8K questions.

    python src/evaluate.py --n 100                          # base model
    python src/evaluate.py --n 100 --adapter outputs/grpo-qwen-gsm8k

Greedy decoding, extract_number() from rewards.py, exact-match accuracy.
The honest comparison: same prompts, same decoder, only the weights differ.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (post-GRPO)")
    ap.add_argument("--data", default=str(ROOT / "data" / "gsm8k_test.jsonl"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    import torch
    from unsloth import FastLanguageModel
    from rewards import extract_number

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=1024, load_in_4bit=True)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"loaded adapter: {args.adapter}")
    FastLanguageModel.for_inference(model)

    rows = [json.loads(l) for l in open(args.data)][:args.n]
    correct = 0
    for i, row in enumerate(rows):
        inputs = tokenizer(row["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        pred = extract_number(text[len(row["prompt"]):])
        gold = float(row["answer"])
        hit = pred is not None and abs(pred - gold) < 1e-6
        correct += hit
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)} acc so far: {correct/(i+1):.1%}")
    print(f"\nFINAL accuracy: {correct}/{len(rows)} = {correct/len(rows):.1%}")


if __name__ == "__main__":
    main()
