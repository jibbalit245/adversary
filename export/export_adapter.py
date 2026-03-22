"""
export_adapter.py
-----------------
Saves and exports the fine-tuned LoRA adapter for use by the main
training pipeline at inference time.

The adapter is exported in two forms:
  1. PEFT adapter format  — loadable with PeftModel.from_pretrained()
  2. Merged model (optional, --merge flag) — full float16 weights,
     suitable for GGUF conversion or direct inference without PEFT.

Usage:
    # Export adapter only (lightweight, recommended for inference loading)
    python export/export_adapter.py \\
        --adapter output/adversary-lora \\
        --out export/adversary-adapter

    # Export merged full model
    python export/export_adapter.py \\
        --adapter output/adversary-lora \\
        --out export/adversary-merged \\
        --merge
"""

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"


def load_adapter_model(adapter_path: str, base_model: str, merge: bool):
    """Load the base model, apply the LoRA adapter, and optionally merge."""
    print(f"Loading tokeniser from '{adapter_path}' …")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path, trust_remote_code=True
    )

    if merge:
        # For merging we need full fp16 weights (not 4-bit)
        print(f"Loading base model '{base_model}' in fp16 for merging …")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # For adapter-only export, load in 4-bit to verify the adapter loads
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
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

    if merge:
        print("Merging LoRA weights into base model …")
        model = model.merge_and_unload()

    return model, tokenizer


def export_adapter_only(adapter_path: str, out_path: Path) -> None:
    """
    Copy the PEFT adapter files to the output directory.
    This is the lightweight format for inference-time loading:

        PeftModel.from_pretrained(base_model, out_path)
    """
    out_path.mkdir(parents=True, exist_ok=True)
    src = Path(adapter_path)

    # Files that make up a PEFT adapter
    adapter_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",  # older peft versions use .bin
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]
    copied = 0
    for fname in adapter_files:
        src_file = src / fname
        if src_file.exists():
            shutil.copy2(src_file, out_path / fname)
            print(f"  Copied {fname}")
            copied += 1

    if copied == 0:
        raise FileNotFoundError(
            f"No adapter files found in '{adapter_path}'. "
            "Run training first."
        )

    print(f"Adapter exported to {out_path} ({copied} files).")


def export_merged(model, tokenizer, out_path: Path) -> None:
    """Save the fully merged model weights."""
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {out_path} …")
    model.save_pretrained(str(out_path), safe_serialization=True)
    tokenizer.save_pretrained(str(out_path))
    print("Merged model saved.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export adversary LoRA adapter for inference."
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to the trained LoRA adapter directory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "adversary-adapter",
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base model ID or path (default: %(default)s)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help=(
            "Merge LoRA weights into the base model and export full weights. "
            "Requires ~28GB VRAM. Without this flag, only the adapter is exported."
        ),
    )
    args = parser.parse_args()

    if args.merge:
        model, tokenizer = load_adapter_model(
            args.adapter, args.base_model, merge=True
        )
        export_merged(model, tokenizer, args.out)
    else:
        # Lightweight export: just copy the adapter files
        export_adapter_only(args.adapter, args.out)
        print(
            "\nTo load the adapter at inference time:\n"
            "    from peft import PeftModel\n"
            "    from transformers import AutoModelForCausalLM\n"
            f"    model = AutoModelForCausalLM.from_pretrained('{args.base_model}', ...)\n"
            f"    model = PeftModel.from_pretrained(model, '{args.out}')\n"
        )


if __name__ == "__main__":
    main()
