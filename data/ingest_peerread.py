"""
ingest_peerread.py
------------------
Ingests the PeerRead dataset from HuggingFace (allenai/peer_read).
Focuses on review text that identifies specific weaknesses in papers.

For each review, we pair the paper abstract (the "claim") with the
weakness-identifying review text (the "challenge").

Output: data/raw/peerread.jsonl
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

DATASET_ID = "allenai/peer_read"
DEFAULT_OUT = Path(__file__).parent / "raw" / "peerread.jsonl"

# Minimum character length for a review to be useful
MIN_REVIEW_LEN = 100

# Keywords that indicate the review is pointing out a weakness
WEAKNESS_KEYWORDS = [
    "weakness",
    "limitation",
    "however",
    "unclear",
    "missing",
    "lacks",
    "fails",
    "insufficient",
    "not clear",
    "does not",
    "doesn't",
    "no evidence",
    "concern",
    "issue",
    "problem",
    "should clarify",
    "needs to",
]


def is_weakness_review(text: str) -> bool:
    """Return True if the review text focuses on identifying weaknesses."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in WEAKNESS_KEYWORDS)


def iter_pairs(dataset):
    """Yield (abstract/claim, review-weakness/challenge) dicts."""
    for split in ("train", "validation", "test"):
        if split not in dataset:
            continue
        for example in tqdm(dataset[split], desc=f"PeerRead [{split}]"):
            abstract = (example.get("abstract") or "").strip()
            if not abstract:
                continue

            reviews = example.get("reviews") or []
            if isinstance(reviews, str):
                # Some HF dataset versions flatten reviews to a string
                reviews = [{"comments": reviews}]

            for review in reviews:
                # Field names vary by dataset version
                text = (
                    review.get("comments")
                    or review.get("review")
                    or review.get("text")
                    or ""
                ).strip()

                if len(text) < MIN_REVIEW_LEN:
                    continue
                if not is_weakness_review(text):
                    continue

                yield {
                    "source": "peerread",
                    "claim": abstract,
                    "challenge": text,
                }


def main(out_path: Path = DEFAULT_OUT) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset '{DATASET_ID}' …")
    try:
        dataset = load_dataset(DATASET_ID, trust_remote_code=True)
    except Exception as exc:
        sys.exit(f"Failed to load dataset: {exc}")

    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for pair in iter_pairs(dataset):
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} weakness review pairs → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest PeerRead weakness-identifying reviews."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output JSONL path (default: %(default)s)",
    )
    args = parser.parse_args()
    main(args.out)
