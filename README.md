# Adversary — LoRA Fine-Tuning Pipeline

A QLoRA fine-tuning pipeline for an adversarial challenger model built on
**Qwen 2.5 14B-Instruct**. The model's sole job is to stress-test reasoning
by finding weaknesses, proposing counterexamples, and asking probing
questions — a sparring partner for a larger 72B model during training.

---

## Repository Layout

```
adversary/
├── data/
│   ├── ingest_changeMyView.py   # Cornell ChangeMyView (ConvoKit)
│   ├── ingest_peerread.py       # PeerRead reviews (HuggingFace)
│   ├── ingest_ibm_debater.py    # IBM Debater quality corpus (HuggingFace)
│   ├── ingest_plato.py          # Plato's Socratic dialogues (Project Gutenberg)
│   └── format_data.py           # Convert all sources → ChatML training samples
├── train/
│   ├── train_qlora.py           # QLoRA training script (rank 32, alpha 64, 4-bit)
│   └── accelerate_config.yaml   # Accelerate multi-GPU config (auto-detects GPU count)
├── eval/
│   └── evaluate_adversary.py    # Evaluate on 20 sample claims (specificity/relevance/depth)
├── export/
│   └── export_adapter.py        # Save/export LoRA adapter for inference loading
├── scripts/
│   ├── launch_single_gpu.sh     # Single-GPU debug launch
│   └── launch_multi_gpu.sh      # Multi-GPU production launch (accelerate)
├── deploy.sh                    # Full deployment: venv → pip → GPU check → dry-run → train
└── requirements.txt
```

---

## Quick Start

### 1 — Deploy (recommended for RunPod / Lambda Labs)

```bash
bash deploy.sh              # full pipeline: install → verify → train
bash deploy.sh --dry-run-only  # install + GPU check + 5-step verification only
```

`deploy.sh` automatically:
- Creates a `venv/`
- Installs all dependencies
- Detects available GPUs (`torch.cuda.device_count()` — no hardcoded count)
- Runs a 5-step dry-run to catch config errors before downloading the full model
- Launches single-GPU or multi-GPU training as appropriate

### 2 — Manual setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## Data Ingestion

Run each ingestion script to populate `data/raw/`:

```bash
python data/ingest_changeMyView.py   # → data/raw/changeMyView.jsonl
python data/ingest_peerread.py       # → data/raw/peerread.jsonl
python data/ingest_ibm_debater.py    # → data/raw/ibm_debater.jsonl
python data/ingest_plato.py          # → data/raw/plato.jsonl
```

Then format all sources into ChatML training samples:

```bash
python data/format_data.py           # → data/processed/train.jsonl + eval.jsonl
```

---

## Training

### Single-GPU (debug / dry-run)

```bash
bash scripts/launch_single_gpu.sh           # 5-step dry-run (default)
bash scripts/launch_single_gpu.sh --full    # full training on GPU 0
```

### Multi-GPU (production)

```bash
bash scripts/launch_multi_gpu.sh
```

GPU count is auto-detected at runtime. Override with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/launch_multi_gpu.sh
```

### Direct Python invocation

```bash
python train/train_qlora.py --dry-run           # 5-step verification
python train/train_qlora.py                     # full training, single process
python train/train_qlora.py --help              # all options
```

---

## Evaluation

```bash
python eval/evaluate_adversary.py \
    --adapter output/adversary-lora \
    --output eval/results.json
```

Scores each of 20 built-in sample claims on:
- **Specificity** — concrete references, numbers, counterexamples
- **Relevance** — keyword overlap with the claim
- **Follow-up depth** — probing questions present

---

## Export

```bash
# Lightweight adapter export (recommended — load with PeftModel.from_pretrained)
python export/export_adapter.py \
    --adapter output/adversary-lora \
    --out export/adversary-adapter

# Full merged model (requires ~28GB VRAM)
python export/export_adapter.py \
    --adapter output/adversary-lora \
    --out export/adversary-merged \
    --merge
```

Loading the adapter at inference time:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct", ...)
model = PeftModel.from_pretrained(base, "export/adversary-adapter")
```

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `Qwen/Qwen2.5-14B-Instruct` |
| Quantization | 4-bit NF4 (bitsandbytes) |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA targets | q/k/v/o/gate/up/down projections |
| Max seq length | 2048 |
| Per-device batch | 2 |
| Gradient accumulation | 8 |
| Learning rate | 2e-4 |
| Scheduler | Cosine |
| Epochs | 3 |
| Token format | Qwen ChatML (`<|im_start|>` / `<|im_end|>`) |
| Loss mask | Assistant turns only |

---

## Requirements

- Python 3.11+
- CUDA 12.4+
- NVIDIA GPU (1× for debug; 4+ GPUs with 48GB+ VRAM recommended for <2h training)
- See `requirements.txt` for Python packages