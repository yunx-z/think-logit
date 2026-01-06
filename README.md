# ThinkLogit 🧠

> **Eliciting Long Chain-of-Thought Reasoning via Logit Arithmetic**

ThinkLogit steers a target language model toward long reasoning *without fine-tuning*, by combining delta logits from a guider model and a base model at decoding time:

```
final_logits = target_logits + α × (guider_logits − base_logits)
```

---

## ✨ Key Features

- **Training-free** — Enhance reasoning capabilities without modifying the target model
- **Cross-family support** — Combine models with different tokenizers (e.g., Llama target + Qwen guider)
- **Parallel inference** — Guider, target, and base model forward passes run concurrently for efficiency
- **Flexible evaluation** — Supports AIME, MATH, GPQA, and LiveCodeBench benchmarks

---

## 🚀 Quick Start

### 1. Installation

```bash
# Create conda environment
conda create -n thinklogit python=3.9 -y
conda activate thinklogit

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Experiments

#### ThinkLogit (Same-Family)

Use a small guider/base pair to steer a large target model:

```bash
python -m eval.gsm.run_eval \
    --data_dir data/eval/aime2024 \
    --save_dir results/aime2024/thinklogit \
    --base_model_name_or_path Qwen/Qwen2.5-32B \
    --expert_model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --anti_expert_model_name_or_path Qwen/Qwen2.5-Math-1.5B \
    --max_new_tokens 8192 \
    --do_sample True \
    --eval_batch_size 1
```

#### ThinkLogit (Cross-Family)

Steer a model from a different family (e.g., Llama) using Qwen guider/base:

```bash
python -m eval.gsm.run_eval \
    --data_dir data/eval/MATH500 \
    --save_dir results/MATH500/qwen-guide-llama \
    --base_model_name_or_path meta-llama/Llama-3.3-70B-Instruct \
    --expert_model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --anti_expert_model_name_or_path Qwen/Qwen2.5-Math-1.5B \
    --tokenizer_mapping_path Llama_to_Qwen_tokalign.json \
    --max_new_tokens 8192 \
    --do_sample False \
    --eval_batch_size 1
```

#### Target Model Only (Baseline)

Run inference with just the target model (no guider):

```bash
python -m eval.gsm.run_eval \
    --data_dir data/eval/aime2024 \
    --save_dir results/aime2024/baseline \
    --base_model_name_or_path Qwen/Qwen2.5-32B \
    --max_new_tokens 8192 \
    --do_sample True \
    --eval_batch_size 1
```

### 3. Compute Metrics

Aggregate results across all completed runs:

```bash
python -m eval.compute_metrics
```

---

## 📁 Project Structure

```
think-logit/
├── modeling/
│   ├── dexperts.py          # Core ThinkLogit implementation
│   └── utils.py             # Utility functions (top-k/p filtering)
├── eval/
│   ├── compute_metrics.py   # Aggregate evaluation results
│   ├── grader.py            # Answer grading logic
│   ├── math_equivalence.py  # Mathematical equivalence checking
│   └── gsm/
│       └── run_eval.py      # Main evaluation script
├── data/
│   └── eval/                # Benchmark datasets
│       ├── aime2024/
│       ├── aime2025/
│       ├── MATH500/
│       ├── gpqa/
│       └── lcb/             # LiveCodeBench
└── results/                 # Generated outputs and metrics
```

---

## ⚙️ Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_dir` | Path to dataset directory | Required |
| `--save_dir` | Output directory for results | Required |
| `--base_model_name_or_path` | HuggingFace model ID for target model | Required |
| `--expert_model_name_or_path` | HuggingFace model ID for guider | `None` |
| `--anti_expert_model_name_or_path` | HuggingFace model ID for base model | `None` |
| `--tokenizer_mapping_path` | Token mapping JSON for cross-family | `None` |
| `--max_new_tokens` | Maximum tokens to generate | `4096` |
| `--do_sample` | Use sampling (`True`) or greedy (`False`) | `False` |
| `--eval_batch_size` | Batch size for evaluation | `1` |
| `--budget_forcing` | Replace EOS with "Wait" for longer CoT | `False` |

---

## 📊 Supported Datasets

| Dataset | Description | Path |
|---------|-------------|------|
| AIME 2024 | AIME 2024 competition problems | `data/eval/aime2024` |
| AIME 2025 | AIME 2025 competition problems | `data/eval/aime2025` |
| MATH500 | MATH benchmark (500 problems) | `data/eval/MATH500` |
| MATH Level 5 | Hardest MATH problems | `data/eval/MATHL5` |
| GPQA | Graduate-level science QA | `data/eval/gpqa` |
| LiveCodeBench | Code generation benchmark | `data/eval/lcb` |

---

## 📊 Output Format

Results are saved incrementally to enable **automatic resumption** if interrupted.

```
results/<dataset>/<experiment_name>/
├── example_prompt.txt    # Sample input prompt
├── predictions.jsonl     # Model outputs (append-only)
├── metrics.json          # Accuracy metrics
└── logits.log            # Per-step logit debugging (optional)
```

### Resuming Interrupted Runs

If `predictions.jsonl` contains N predictions, rerunning will automatically resume from question N+1.

---

## 🔬 Cross-Family Mode

ThinkLogit supports combining models with **different tokenizers** via token mapping.

The token mapping file (e.g., `Gemma_to_Qwen_tokalign.json`) maps tokens between the target and guider vocabularies:

```json
{
  "▁the": "Ġthe",
  "▁of": "Ġof",
  ...
}
```

This enables logit transfer across model families (e.g., using Qwen guider/base to steer EXAONE, LLaMA, or OLMo target models).


