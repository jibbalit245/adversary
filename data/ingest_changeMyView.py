"""
ingest_changeMyView.py
----------------------
Ingests the Cornell ChangeMyView corpus via ConvoKit.
Focuses on "delta" conversations — exchanges where a challenger
successfully changed the original poster's mind.

The challenger's turns that earned a delta are extracted as
(claim, challenge) pairs for adversary training.

Output: data/raw/changeMyView.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

try:
    from convokit import Corpus, download
except ImportError:
    sys.exit("ConvoKit is not installed. Run: pip install convokit")

CORPUS_NAME = "winning-args-corpus"
DEFAULT_OUT = Path(__file__).parent / "raw" / "changeMyView.jsonl"


def iter_delta_pairs(corpus: Corpus):
    """
    Yield dicts with keys: source, claim, challenge.

    A delta-earning reply is marked with meta["success"] == True in
    the winning-args-corpus. We pair the immediate parent utterance
    (the claim) with the successful challenger reply (the challenge).
    """
    for convo in tqdm(corpus.iter_conversations(), desc="Conversations"):
        utterances = {u.id: u for u in convo.iter_utterances()}

        for uid, utt in utterances.items():
            if not utt.meta.get("success", False):
                continue

            challenge_text = (utt.text or "").strip()
            if not challenge_text:
                continue

            parent_id = utt.reply_to
            if parent_id and parent_id in utterances:
                claim_text = (utterances[parent_id].text or "").strip()
            else:
                # Fall back to conversation root
                root_id = convo.meta.get("root", uid)
                root = utterances.get(root_id)
                claim_text = (root.text if root else "").strip()

            if not claim_text or not challenge_text:
                continue

            yield {
                "source": "changeMyView",
                "claim": claim_text,
                "challenge": challenge_text,
            }


def main(out_path: Path = DEFAULT_OUT) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading / loading corpus '{CORPUS_NAME}' …")
    try:
        corpus = Corpus(filename=download(CORPUS_NAME))
    except Exception as exc:
        sys.exit(f"Failed to load corpus: {exc}")

    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for pair in iter_delta_pairs(corpus):
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} delta pairs → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest Cornell ChangeMyView delta conversations."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output JSONL path (default: %(default)s)",
    )
    args = parser.parse_args()
    main(args.out)
