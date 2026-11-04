"""Pick 32 ARC-Challenge questions with an equal share of answer keys (8 each of A, B, C, D).

Run once to write prompts_arc32_balanced.json, which the experiment notebook loads.
"""
import json
import random

from datasets import load_dataset

OUTPUT_FILE = "prompts_arc32_balanced.json"
KEYS = ["A", "B", "C", "D"]
PER_KEY = 8
SEED = 42


def make_prompt(item):
    lines = [item["question"]]
    for label, text in zip(item["choices"]["label"], item["choices"]["text"]):
        lines.append(f"{label}. {text}")
    lines.append("Answer with the letter and briefly explain why.")
    return "\n".join(lines)


def main():
    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rng = random.Random(SEED)

    
    pool = [item for item in arc if item["choices"]["label"] == KEYS]

    chosen = []
    for key in KEYS:
        chosen += rng.sample([item for item in pool if item["answerKey"] == key], PER_KEY)
    
    rng.shuffle(chosen)

    data = {
        "ids": [item["id"] for item in chosen],
        "prompts": [make_prompt(item) for item in chosen],
        "answer_keys": [item["answerKey"] for item in chosen],
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print("saved", len(chosen), "prompts to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
