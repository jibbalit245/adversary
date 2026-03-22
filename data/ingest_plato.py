"""
ingest_plato.py
---------------
Downloads Plato's Meno, Euthyphro, and Theaetetus from Project Gutenberg
and extracts passages where Socrates challenges an interlocutor's position
through questioning (the Socratic elenchus).

Strategy:
- Download each dialogue as plain text from Gutenberg.
- Split into speaker-labelled turns using regex (e.g. "SOCRATES." / "MENO.").
- Extract consecutive (interlocutor-claim → Socrates-question) pairs.
- Filter to Socrates turns that end with a question mark (challenges).

Output: data/raw/plato.jsonl
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterator

import requests
from tqdm import tqdm

DEFAULT_OUT = Path(__file__).parent / "raw" / "plato.jsonl"

# Project Gutenberg plain-text URLs for each dialogue
DIALOGUES = {
    "Meno": "https://www.gutenberg.org/cache/epub/1744/pg1744.txt",
    "Euthyphro": "https://www.gutenberg.org/cache/epub/1642/pg1642.txt",
    "Theaetetus": "https://www.gutenberg.org/cache/epub/1726/pg1726.txt",
}

# Regex to detect a speaker label at the start of a line.
# Gutenberg Jowett translations use "SOCRATES.", "MENO.", "EUTHYPHRO.", etc.
SPEAKER_RE = re.compile(r"^([A-Z][A-Z\s]{1,20})\.\s*(.*)", re.MULTILINE)

# Minimum character length for a turn to be useful
MIN_TURN_LEN = 40

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; adversary-pipeline/1.0; "
        "+https://github.com/jibbalit245/adversary)"
    )
}


def fetch_text(url: str, retries: int = 3) -> str:
    """Fetch plain text from URL with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            print(f"  Retry {attempt}/{retries} for {url}: {exc}")
            time.sleep(2 ** attempt)
    return ""  # unreachable


def parse_turns(text: str) -> list[tuple[str, str]]:
    """
    Parse the dialogue text into a list of (speaker, text) turns.

    Gutenberg Jowett dialogues have lines like:
        SOCRATES. And do you think there is such a thing as knowledge?
        MENO.  Certainly.
    """
    turns: list[tuple[str, str]] = []
    current_speaker = ""
    current_lines: list[str] = []

    for line in text.splitlines():
        m = SPEAKER_RE.match(line.strip())
        if m:
            if current_speaker and current_lines:
                body = " ".join(current_lines).strip()
                if len(body) >= MIN_TURN_LEN:
                    turns.append((current_speaker, body))
            current_speaker = m.group(1).strip()
            rest = m.group(2).strip()
            current_lines = [rest] if rest else []
        else:
            stripped = line.strip()
            if stripped:
                current_lines.append(stripped)

    if current_speaker and current_lines:
        body = " ".join(current_lines).strip()
        if len(body) >= MIN_TURN_LEN:
            turns.append((current_speaker, body))

    return turns


def iter_socratic_pairs(
    turns: list[tuple[str, str]], dialogue_name: str
) -> Iterator[dict]:
    """
    Yield dicts where Socrates responds to an interlocutor claim
    with a probing question or challenge.
    """
    for i in range(1, len(turns)):
        prev_speaker, prev_text = turns[i - 1]
        curr_speaker, curr_text = turns[i]

        if curr_speaker != "SOCRATES":
            continue
        # Socrates must be posing a question or challenge
        if "?" not in curr_text:
            continue
        # Previous turn must be from someone else
        if prev_speaker == "SOCRATES":
            continue

        yield {
            "source": "plato",
            "dialogue": dialogue_name,
            "interlocutor": prev_speaker,
            "claim": prev_text,
            "challenge": curr_text,
        }


def main(out_path: Path = DEFAULT_OUT) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for name, url in tqdm(DIALOGUES.items(), desc="Dialogues"):
            print(f"\nFetching {name} from {url} …")
            try:
                text = fetch_text(url)
            except requests.RequestException as exc:
                print(f"  WARNING: Could not fetch {name}: {exc}. Skipping.")
                continue

            turns = parse_turns(text)
            print(f"  Parsed {len(turns)} speaker turns from {name}.")

            count = 0
            for pair in iter_socratic_pairs(turns, name):
                fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
                count += 1
            print(f"  Extracted {count} Socratic challenge pairs from {name}.")
            total += count

    print(f"\nTotal: {total} Socratic elenchus pairs → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Socratic challenge pairs from Plato (Project Gutenberg)."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output JSONL path (default: %(default)s)",
    )
    args = parser.parse_args()
    main(args.out)
