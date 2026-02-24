# Training Guide

Reference for submitting hosted RL training runs on Prime Intellect Lab.

Docs: https://docs.primeintellect.ai/hosted-training/advanced-configs

---

## Prerequisites

- Environment pushed to Hub: `prime env push`
- Distribution checks passed: `python scripts/check_reward_distribution.py`
- You are logged in: `prime login`

---

## Quick Submit (existing Hub env)

```bash
python scripts/submit_training.py --env your-username/my-env --yes
```

Or manually — all config goes in a TOML file, then:

```bash
prime rl run configs/rl/my-env.toml
```

There are no `--env`, `--model`, `--gpu` flags on `prime rl run`. Everything is in the TOML.

---

## Full Config Reference (TOML)

Below is the complete annotated config per the official docs.
Required fields are uncommented; optional fields shown as comments with their defaults.

```toml
# ============================================================
# Core (required)
# ============================================================
model = "Qwen/Qwen3-4B-Instruct-2507"   # Must be a supported model
max_steps = 500
batch_size = 512                          # Rollouts consumed per training step
rollouts_per_example = 16                 # Rollouts generated per dataset example

# ============================================================
# Training Hyperparameters (optional)
# ============================================================
# learning_rate = 1e-4                   # LoRA learning rate
# lora_alpha = 16                        # LoRA alpha scaling
# oversampling_factor = 1.0             # Generate this multiple of rollouts, select best batch_size
# max_async_level = 4                   # Async generation level
# env_files = ["secrets.env"]           # Secrets files for API keys (e.g. WANDB_API_KEY)

# ============================================================
# Sampling (required)
# ============================================================
[sampling]
max_tokens = 2048                         # Max tokens per model response (official default: 2048)

# ============================================================
# Environment (at least one required)
# ============================================================
[[env]]
id = "your-username/my-env"              # owner/name format
# args = { split = "train" }            # Passed to load_environment()

# Multi-environment: just add more [[env]] sections
# [[env]]
# id = "your-username/another-env"

# ============================================================
# W&B (optional)
# ============================================================
# [wandb]
# project = "my-project"
# name = "my-run-name"
# entity = "my-team"

# ============================================================
# Online Evaluation (optional)
# ============================================================
# [eval]
# interval = 100
# num_examples = -1                     # -1 = all examples
# rollouts_per_example = 1
# eval_base_model = true
#
# [[eval.env]]
# id = "your-username/my-env"
# args = { split = "test" }
# num_examples = 50
# rollouts_per_example = 4

# ============================================================
# Validation (optional, runs more frequently than eval)
# ============================================================
# [val]
# num_examples = 64
# rollouts_per_example = 1
# interval = 5

# ============================================================
# Difficulty Filtering (optional)
# ============================================================
# [buffer]
# online_difficulty_filtering = false
# easy_threshold = 0.8
# hard_threshold = 0.2
# easy_fraction = 0.0                   # 0.0 = drop all easy examples
# hard_fraction = 0.0
# env_ratios = [0.5, 0.5]              # For multi-env, ratio per env
# seed = 42

# ============================================================
# Checkpoints (optional)
# ============================================================
# [checkpoints]
# interval = 100
# keep_cloud = 5                        # -1 = keep all

# ============================================================
# Warm-start from checkpoint (optional, top-level)
# ============================================================
# checkpoint_id = "cp_abc123"
```

---

## Supported Models

Run `prime rl models` to see the current list. As of the last update:

- `Qwen/Qwen3-4B-Instruct-2507` / `Qwen/Qwen3-4B-Thinking-2507`
- `Qwen/Qwen3-30B-Instruct-2507` / `Qwen/Qwen3-30B-Thinking-2507`
- `Qwen/Qwen3-235B-Instruct-2507` / `Qwen/Qwen3-235B-Thinking-2507`
- `PrimeIntellect/INTELLECT-3`



---

## GRPO Hyperparameter Notes

**`rollouts_per_example` (group size G):** More rollouts = better advantage estimate = lower variance = more stable training. Default 8; use 16–32 for noisy reward signals. Higher uses more VRAM.

**`batch_size`:** Total rollouts consumed per gradient step. Must be divisible by `rollouts_per_example`. Keep ≥ 256 for stable updates; 512–1024 for more stability.

**`learning_rate`:** Default is `1e-4` for LoRA. If reward spikes then crashes, reduce to `1e-5`.

**`oversampling_factor`:** Generates this multiple of `batch_size` rollouts, then selects the most informative `batch_size` for the gradient update. Default `1.0` (no oversampling). Increasing to `2.0` improves sample quality but doubles generation cost. This is **not** gradient clipping — it controls rollout selection, not gradient magnitude.

**`max_steps`:** For GSM8K on a 4B model, 300–500 steps shows measurable improvement. Use `[eval]` with `interval=50` to track progress and pick the best checkpoint.

---

## Monitoring

```bash
prime rl list                           # list all runs and their status
prime rl logs <run-id>                  # stream logs for a run
prime rl metrics <run-id>              # training metrics
prime rl checkpoints <run-id>          # list checkpoints for a run
```

Watch reward metrics in W&B or dashboard logs:
- `reward_mean` should increase over training
- `reward_std` should stay > 0.05 (if it collapses, reward function is being hacked)
- `kl_div` should stay < 0.5 (higher = model drifting far from reference, risk of collapse)

---

## After Training

List checkpoints (download link appears in dashboard once training completes):
```bash
prime rl checkpoints <run-id>
# Full run details including checkpoint URLs:
prime rl get <run-id>
```
Visit https://app.primeintellect.ai/dashboard/training/<run-id> to download the adapter.

Evaluate the fine-tuned checkpoint by serving it locally (e.g. with vLLM) then pointing `prime eval run` at it:
```bash
# 1. Serve the merged checkpoint locally
vllm serve ./checkpoints/merged/ --port 8000

# 2. In another terminal, eval against it
prime eval run your-username/my-env \
  -m your-model-name \
  -b http://localhost:8000/v1 \
  -n 200
```
`-m` takes a model **name string** (not a local path). The `-b` flag points the eval at your local server, bypassing Prime Inference validation.

Merge LoRA and push to Hub:
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "Qwen/Qwen3-4B-Instruct-2507"
base = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(base, "./checkpoints/best/")
merged = model.merge_and_unload()
merged.save_pretrained("./checkpoints/merged/")
AutoTokenizer.from_pretrained(base_model).save_pretrained("./checkpoints/merged/")
```

```bash
huggingface-cli upload your-username/my-grpo-model ./checkpoints/merged/
```

---

## Common Training Failures

**Reward collapses to 0 after initial improvement:**
- KL divergence too high → reduce `learning_rate`
- Reward function being gamed → check reward std; if near 0, add diversity pressure

**No improvement after 200 steps:**
- Reward signal too sparse → if mean < 3%, add SFT warmup before RL
- `rollouts_per_example` too small → try 16

**OOM during generation:**
- Reduce `batch_size` or `[sampling].max_tokens`
- Enable `online_difficulty_filtering` to skip examples that always score 0 or 1
