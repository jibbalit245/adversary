"""
format_data.py
--------------
Reads all raw JSONL files from data/raw/ and converts them into
ChatML-formatted training samples for the adversary model.

Each sample is a multi-turn exchange in Qwen's native ChatML format:

    <|im_start|>system
    {SYSTEM_PROMPT}<|im_end|>
    <|im_start|>user
    {claim}<|im_end|>
    <|im_start|>assistant
    {challenge}<|im_end|>

Loss is computed ONLY on the assistant turn. The formatted dataset
is saved as data/processed/train.jsonl and data/processed/eval.jsonl.

Usage:
    python data/format_data.py [--raw-dir data/raw] [--out-dir data/processed]
"""

import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT = """You are a rigorous intellectual challenger. Your role is to stress-test reasoning, not to be agreeable. For every claim presented to you:

Identify the weakest link in the argument
Propose a specific counterexample or edge case
Ask: "What would have to be true for this to be wrong?"
If the response is vague, demand specifics
If the response is specific, probe the assumptions behind it
You are not hostile. You are demanding. You do not accept "it seems like" or "it could be argued." You accept evidence, logical derivation, and honest uncertainty. When the other model says "I don't know," that is acceptable. Push on WHERE they don't know and what they would need to find out."""

DEFAULT_RAW = Path(__file__).parent / "raw"
DEFAULT_OUT = Path(__file__).parent / "processed"
EVAL_FRACTION = 0.05
RANDOM_SEED = 42


def build_chatml(claim: str, challenge: str) -> str:
    """
    Return a single ChatML string with the system prompt, user claim,
    and assistant challenge. Qwen uses <|im_start|> / <|im_end|>.
    """
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{claim}<|im_end|>\n"
        f"<|im_start|>assistant\n{challenge}<|im_end|>"
    )


def load_raw_pairs(raw_dir: Path) -> list[dict]:
    """Load all JSONL files from raw_dir and return list of pair dicts."""
    pairs: list[dict] = []
    jsonl_files = sorted(raw_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(
            f"No JSONL files found in {raw_dir}. "
            "Run the ingest_*.py scripts first."
        )
    for path in jsonl_files:
        file_count = 0
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                claim = (obj.get("claim") or "").strip()
                challenge = (obj.get("challenge") or "").strip()
                if claim and challenge:
                    pairs.append(
                        {
                            "source": obj.get("source", "unknown"),
                            "claim": claim,
                            "challenge": challenge,
                            "text": build_chatml(claim, challenge),
                        }
                    )
                    file_count += 1
        print(f"  Loaded {file_count} pairs from {path.name}")
    return pairs


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main(raw_dir: Path = DEFAULT_RAW, out_dir: Path = DEFAULT_OUT) -> None:
    print(f"Loading raw pairs from {raw_dir} …")
    pairs = load_raw_pairs(raw_dir)
    print(f"Total pairs loaded: {len(pairs)}")

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(pairs)

    split_idx = max(1, int(len(pairs) * (1 - EVAL_FRACTION)))
    train_pairs = pairs[:split_idx]
    eval_pairs = pairs[split_idx:]

    train_path = out_dir / "train.jsonl"
    eval_path = out_dir / "eval.jsonl"

    print(f"Writing {len(train_pairs)} training samples → {train_path}")
    write_jsonl(train_pairs, train_path)

    print(f"Writing {len(eval_pairs)} eval samples    → {eval_path}")
    write_jsonl(eval_pairs, eval_path)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Format raw data into ChatML training samples."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW,
        help="Directory containing raw JSONL files (default: %(default)s)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory for train/eval JSONL (default: %(default)s)",
    )
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir)
