"""
evaluate_adversary.py
---------------------
Evaluates the fine-tuned adversary model against 20 sample claims and
scores challenge quality on three dimensions:

  1. Specificity  — Does the challenge reference concrete details?
  2. Relevance    — Is the challenge directly about the claim?
  3. Follow-up depth — Does the challenge ask a probing question?

Each dimension is scored 0–1 using heuristics. A final composite
score (mean of the three) is reported per claim and averaged overall.

Usage:
    python eval/evaluate_adversary.py \\
        --adapter output/adversary-lora \\
        [--base-model Qwen/Qwen2.5-14B-Instruct] \\
        [--output eval/results.json]
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# 20 sample claims covering a range of topics and argument types
# ---------------------------------------------------------------------------
SAMPLE_CLAIMS = [
    "Renewable energy will inevitably replace fossil fuels within the next 20 years.",
    "Large language models are simply pattern-matching engines with no genuine understanding.",
    "Democracy is the best form of government because it represents the will of the people.",
    "The evidence clearly shows that increasing the minimum wage reduces poverty.",
    "Artificial general intelligence will be achieved before 2040.",
    "Social media platforms are primarily responsible for the rise in teen mental health issues.",
    "Higher education is essential for economic success in the modern economy.",
    "Free markets always allocate resources more efficiently than planned economies.",
    "Vaccines are safe and effective — the science is settled.",
    "Climate change is the most pressing existential threat facing humanity today.",
    "Human memory works like a video recorder, storing experiences accurately.",
    "The trolley problem demonstrates that consequentialist ethics is correct.",
    "Most scientific breakthroughs come from individual genius rather than collective effort.",
    "Stricter gun control laws would significantly reduce violent crime.",
    "Learning a second language before age 10 is fundamentally easier than learning it as an adult.",
    "Economic growth and environmental sustainability are inherently at odds.",
    "The death penalty serves as an effective deterrent to violent crime.",
    "Consciousness is an emergent property of sufficiently complex information processing.",
    "Meritocracy ensures that the most capable individuals rise to positions of influence.",
    "Regular exercise is more effective than medication for treating mild-to-moderate depression.",
]

SYSTEM_PROMPT = """You are a rigorous intellectual challenger. Your role is to stress-test reasoning, not to be agreeable. For every claim presented to you:

Identify the weakest link in the argument
Propose a specific counterexample or edge case
Ask: "What would have to be true for this to be wrong?"
If the response is vague, demand specifics
If the response is specific, probe the assumptions behind it
You are not hostile. You are demanding. You do not accept "it seems like" or "it could be argued." You accept evidence, logical derivation, and honest uncertainty. When the other model says "I don't know," that is acceptable. Push on WHERE they don't know and what they would need to find out."""


# ---------------------------------------------------------------------------
# Scoring heuristics
# ---------------------------------------------------------------------------

# Words that indicate specificity (concrete references)
SPECIFIC_MARKERS = [
    r"\b\d{4}\b",               # years
    r"\b\d+[\.,]\d+",           # numbers with decimals
    r"\b\d+\s*%",               # percentages
    r"\bfor example\b",
    r"\bspecifically\b",
    r"\bsuch as\b",
    r"\bfor instance\b",
    r"\bconcretely\b",
    r"\bin particular\b",
    r"\bnamely\b",
    r"\bcounterexample\b",
    r"\bedge case\b",
]

# Probing / follow-up question patterns
PROBE_PATTERNS = [
    r"what would have to be true",
    r"how would you",
    r"can you define",
    r"what evidence",
    r"where exactly",
    r"which specific",
    r"what do you mean by",
    r"how do you know",
    r"what mechanism",
    r"under what conditions",
    r"what would falsify",
    r"have you considered",
    r"\bwhy\b.*\?",
    r"\bwhat\b.*\?",
    r"\bhow\b.*\?",
    r"\bwhere\b.*\?",
    r"\bwhen\b.*\?",
    r"\bwhich\b.*\?",
]


def score_specificity(text: str) -> float:
    """0–1: does the response contain concrete, specific references?"""
    matches = sum(
        1 for p in SPECIFIC_MARKERS if re.search(p, text, re.IGNORECASE)
    )
    return min(1.0, matches / 3.0)


def score_relevance(claim: str, challenge: str) -> float:
    """0–1: does the challenge share key words/concepts with the claim?"""
    # Simple unigram overlap on content words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "to", "of", "and", "or", "in", "on", "at", "for",
        "it", "its", "this", "that", "with", "as", "by", "from",
        "will", "would", "could", "should", "may", "might", "must",
        "not", "no", "do", "does", "did", "have", "has", "had",
    }
    claim_words = {
        w.lower().strip(".,!?;:\"'")
        for w in claim.split()
        if w.lower() not in stop_words and len(w) > 3
    }
    challenge_words = {
        w.lower().strip(".,!?;:\"'")
        for w in challenge.split()
        if w.lower() not in stop_words and len(w) > 3
    }
    if not claim_words:
        return 0.0
    overlap = len(claim_words & challenge_words) / len(claim_words)
    return min(1.0, overlap * 2)  # normalised: 50% overlap → 1.0


def score_follow_up_depth(text: str) -> float:
    """0–1: does the response ask probing follow-up questions?"""
    question_count = text.count("?")
    probe_count = sum(
        1 for p in PROBE_PATTERNS if re.search(p, text, re.IGNORECASE)
    )
    depth = min(1.0, (question_count * 0.2) + (probe_count * 0.3))
    return depth


def score_challenge(claim: str, challenge: str) -> dict:
    specificity = score_specificity(challenge)
    relevance = score_relevance(claim, challenge)
    depth = score_follow_up_depth(challenge)
    composite = (specificity + relevance + depth) / 3.0
    return {
        "specificity": round(specificity, 3),
        "relevance": round(relevance, 3),
        "follow_up_depth": round(depth, 3),
        "composite": round(composite, 3),
    }


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def load_model(adapter_path: str, base_model: str):
    """Load the base model with the LoRA adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"Loading tokeniser from '{adapter_path}' …")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model '{base_model}' in 4-bit …")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    print(f"Loading LoRA adapter from '{adapter_path}' …")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate_challenge(
    claim: str, model, tokenizer, max_new_tokens: int = 256
) -> str:
    """Generate an adversarial challenge for the given claim."""
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{claim}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens (not the prompt)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=False)
    # Strip trailing <|im_end|> if present
    text = text.split("<|im_end|>")[0].strip()
    return text


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate adversary LoRA model on 20 sample claims."
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to the LoRA adapter directory (output of training).",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-14B-Instruct",
        help="Base model ID or path (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results.json",
        help="Output JSON file for results (default: %(default)s)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max tokens to generate per challenge (default: %(default)s)",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter, args.base_model)

    results = []
    for i, claim in enumerate(SAMPLE_CLAIMS, 1):
        print(f"\n[{i:02d}/20] Claim: {claim[:80]}…" if len(claim) > 80 else f"\n[{i:02d}/20] Claim: {claim}")
        t0 = time.time()
        challenge = generate_challenge(claim, model, tokenizer, args.max_new_tokens)
        elapsed = time.time() - t0

        scores = score_challenge(claim, challenge)
        print(f"  Challenge ({elapsed:.1f}s): {challenge[:120]}…" if len(challenge) > 120 else f"  Challenge ({elapsed:.1f}s): {challenge}")
        print(
            f"  Scores — specificity: {scores['specificity']:.3f} | "
            f"relevance: {scores['relevance']:.3f} | "
            f"depth: {scores['follow_up_depth']:.3f} | "
            f"composite: {scores['composite']:.3f}"
        )

        results.append(
            {
                "claim": claim,
                "challenge": challenge,
                "scores": scores,
                "elapsed_s": round(elapsed, 2),
            }
        )

    # Summary statistics
    composites = [r["scores"]["composite"] for r in results]
    avg_composite = sum(composites) / len(composites)
    avg_specificity = sum(r["scores"]["specificity"] for r in results) / len(results)
    avg_relevance = sum(r["scores"]["relevance"] for r in results) / len(results)
    avg_depth = sum(r["scores"]["follow_up_depth"] for r in results) / len(results)

    summary = {
        "num_claims": len(results),
        "avg_specificity": round(avg_specificity, 3),
        "avg_relevance": round(avg_relevance, 3),
        "avg_follow_up_depth": round(avg_depth, 3),
        "avg_composite": round(avg_composite, 3),
    }

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    output = {"summary": summary, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    print(f"\nFull results saved → {args.output}")


if __name__ == "__main__":
    main()
