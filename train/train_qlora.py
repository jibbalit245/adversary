"""
train_qlora.py
--------------
QLoRA fine-tuning of Qwen 2.5 14B-Instruct as an adversarial challenger.

Configuration:
  - 4-bit NF4 quantization via bitsandbytes
  - LoRA rank 32, alpha 64, targeting all attention + MLP projection layers
  - Loss computed ONLY on assistant turns (ChatML <|im_start|>assistant tokens)
  - Multi-GPU via Accelerate (handled by the launcher — pass no extra flags)

Usage:
    # Direct (single-process):
    python train/train_qlora.py

    # Multi-GPU via accelerate (recommended):
    accelerate launch --config_file train/accelerate_config.yaml \\
        train/train_qlora.py
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Defaults — override via CLI flags
# ---------------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
TRAIN_DATA = Path(__file__).parent.parent / "data" / "processed" / "train.jsonl"
EVAL_DATA = Path(__file__).parent.parent / "data" / "processed" / "eval.jsonl"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "adversary-lora"

LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

MAX_SEQ_LEN = 2048
BATCH_SIZE = 2          # per-device; effective batch is multiplied by #GPUs × grad_accum
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.03

# Qwen2.5 ChatML token IDs (looked up at runtime from tokenizer)
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# LoRA target modules for Qwen2.5
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# ---------------------------------------------------------------------------
# Tokenisation — mask everything except the assistant turn
# ---------------------------------------------------------------------------

def tokenise_and_mask(example: dict, tokenizer, max_len: int) -> dict:
    """
    Tokenise a ChatML sample and set labels so loss is computed ONLY
    on the assistant turn tokens.

    The text looks like:
        <|im_start|>system\n...<|im_end|>\n
        <|im_start|>user\n...<|im_end|>\n
        <|im_start|>assistant\n{challenge}<|im_end|>

    We find the last occurrence of "<|im_start|>assistant\n" and mask
    all tokens before (and including) that header with -100.
    """
    text: str = example["text"]

    tokenised = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    input_ids: list[int] = tokenised["input_ids"]
    labels: list[int] = list(input_ids)

    # Find the boundary: everything up to and including the assistant header
    # is masked. We encode the assistant header separately to get its token IDs.
    assistant_header = f"{IM_START}assistant\n"
    header_ids = tokenizer.encode(assistant_header, add_special_tokens=False)
    header_len = len(header_ids)

    # Find the LAST occurrence of header_ids in input_ids
    mask_up_to = len(input_ids)  # default: mask everything (shouldn't happen)
    for start in range(len(input_ids) - header_len, -1, -1):
        if input_ids[start : start + header_len] == header_ids:
            # Mask up to and including the header
            mask_up_to = start + header_len
            break

    for i in range(mask_up_to):
        labels[i] = -100

    tokenised["labels"] = labels
    return tokenised


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str):
    """Load quantised base model and tokeniser, then apply LoRA."""

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading tokeniser from '{model_name}' …")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model '{model_name}' in 4-bit …")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_split(path: Path, tokenizer, max_len: int):
    """Load a JSONL file and tokenise it."""
    dataset = load_dataset("json", data_files=str(path), split="train")
    return dataset.map(
        lambda ex: tokenise_and_mask(ex, tokenizer, max_len),
        remove_columns=dataset.column_names,
        desc=f"Tokenising {path.name}",
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_training_args(
    output_dir: Path,
    num_epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    warmup_ratio: float,
) -> TrainingArguments:

    use_wandb = os.environ.get("WANDB_API_KEY") is not None
    report_to = "wandb" if use_wandb else "none"

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=lr,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        tf32=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to=report_to,
        ddp_find_unused_parameters=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA training for adversary model.")
    parser.add_argument("--model", default=BASE_MODEL, help="Base model ID or path")
    parser.add_argument("--train-data", type=Path, default=TRAIN_DATA)
    parser.add_argument("--eval-data", type=Path, default=EVAL_DATA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model)

    print("Loading and tokenising training data …")
    train_dataset = load_split(args.train_data, tokenizer, args.max_seq_len)
    eval_dataset = load_split(args.eval_data, tokenizer, args.max_seq_len)
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Eval  samples: {len(eval_dataset)}")

    training_args = build_training_args(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        warmup_ratio=WARMUP_RATIO,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting training …")
    trainer.train()

    print(f"Saving LoRA adapter to {args.output_dir} …")
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print("Training complete.")


if __name__ == "__main__":
    main()

