"""
ingest_ibm_debater.py
---------------------
Ingests the IBM Project Debater Argument Quality dataset from HuggingFace
(ibm/debate_speeches — also listed as ibm-research/debate_speeches).

Focuses on high-quality counter-arguments scored by human annotators.
Pairs the original motion/claim with the high-quality counter-argument.

Output: data/raw/ibm_debater.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("datasets is not installed. Run: pip install datasets")

# Try the canonical HF dataset ID; fall back to alternate slug
DATASET_IDS = [
    "ibm/debate_speeches",
    "ibm-research/debate_speeches",
    "orieg/ibm-debater-acl-2019-argument-quality-ranking",
]
DEFAULT_OUT = Path(__file__).parent / "raw" / "ibm_debater.jsonl"

# Minimum quality score to keep (scale varies by dataset version; normalised below)
MIN_QUALITY_SCORE = 0.6


def normalise_score(score) -> float:
    """Normalise score to [0, 1] regardless of original scale."""
    if score is None:
        return 0.0
    score = float(score)
    # Scores on a 1–4 scale → normalise to 0–1
    if score > 1.0:
        score = (score - 1.0) / 3.0
    return score


def iter_pairs(dataset):
    """Yield (motion/claim, counter-argument/challenge) dicts."""
    for split in ("train", "validation", "test"):
        if split not in dataset:
            continue
        for example in tqdm(dataset[split], desc=f"IBM Debater [{split}]"):
            # Field names depend on dataset version
            motion = (
                example.get("topic")
                or example.get("motion")
                or example.get("claim")
                or ""
            ).strip()

            argument = (
                example.get("argument")
                or example.get("speech")
                or example.get("text")
                or ""
            ).strip()

            if not motion or not argument:
                continue

            # Quality / stance filtering
            score_raw = (
                example.get("score")
                or example.get("quality")
                or example.get("mace_score")
                or example.get("ibm_score")
            )
            score = normalise_score(score_raw)
            if score_raw is not None and score < MIN_QUALITY_SCORE:
                continue

            # Prefer counter-arguments (stance = "con" / 0 / False)
            stance = example.get("stance") or example.get("label") or ""
            stance_str = str(stance).lower()
            # Include both stances but tag the record
            yield {
                "source": "ibm_debater",
                "claim": motion,
                "challenge": argument,
                "stance": stance_str,
                "quality_score": score,
            }


def load_any(dataset_ids: list[str]):
    """Try each dataset ID until one loads successfully."""
    last_exc = None
    for did in dataset_ids:
        try:
            print(f"  Trying '{did}' …")
            return load_dataset(did, trust_remote_code=True)
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(
        f"Could not load any of {dataset_ids}. Last error: {last_exc}"
    )


def main(out_path: Path = DEFAULT_OUT) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading IBM Debater dataset …")
    try:
        dataset = load_any(DATASET_IDS)
    except RuntimeError as exc:
        sys.exit(str(exc))

    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for pair in iter_pairs(dataset):
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} high-quality argument pairs → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest IBM Project Debater high-quality counter-arguments."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output JSONL path (default: %(default)s)",
    )
    args = parser.parse_args()
    main(args.out)
