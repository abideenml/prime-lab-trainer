# Prime Lab Trainer

A Claude Code skill that builds, validates, and submits RL training environments on [Prime Intellect Lab](https://app.primeintellect.ai) — from any HuggingFace dataset to a live GRPO training run, fully automated.

You describe the task. Claude writes the environment, validates it, pushes it to the Hub, and submits a hosted GRPO training run.

---

## What It Does

Given a prompt like:

> *"Build an environment for cais/mmlu abstract algebra, use Qwen/Qwen3-30B-Instruct-2507, 1000 steps, and submit."*

Claude autonomously:

1. Inspects the dataset schema and answer format
2. Writes the reward function and environment package
3. Unit tests the reward function against auto-generated cases
4. Runs a real evaluation via Prime Inference
5. Validates the reward distribution is learnable
6. Pushes the environment to the Prime Hub
7. Submits a hosted GRPO training run

No GPU setup. No infrastructure config. One prompt.

---

## Prerequisites

### Accounts
- [Prime Intellect](https://app.primeintellect.ai) account
- [Claude Code](https://claude.ai/code) (`npm install -g @anthropic-ai/claude-code`)

### Install
```bash
pip install prime verifiers datasets
```

### Login
```bash
prime login    # opens browser → authenticates everything
```

 `prime login` covers inference, the environment hub, and training submission.

> **HuggingFace auth is not required** for public datasets. Only needed if your dataset is private on HF, or you want to publish trained model checkpoints to HF Hub after training.

---

## Quickstart

```bash
git clone https://github.com/your-username/prime-lab-trainer
cd prime-lab-trainer
pip install prime verifiers datasets
prime login
claude
```

Then in Claude:

```
Build an RL training environment for the pubmed_qa dataset and submit training.
```

---

## How It Works

This project is a **Claude Code skill** — a structured instruction set in `SKILL.md` that Claude reads automatically when you open the project. Claude follows a strict 7-step workflow, validating at every stage before proceeding.

```
Step 0  Preflight Check        Verify environment, credentials, tools
Step 1  Inspect Dataset        Learn schema, columns, answer format
Step 2  Write Environment      Reward function + 3-file package
Step 3  Validate Reward        Unit tests against auto-generated cases
Step 4  Local Eval             Real model completions via Prime Inference
Step 5  Distribution Check     Statistical validation of reward signal
Step 6  Push to Hub            Publish environment to Prime Hub
Step 7  Submit Training        GRPO training run on hosted GPUs
```

**Claude never skips a step. If any step fails, it stops and fixes the issue before continuing.**

---

## The 7-Step Workflow

### Step 0 — Preflight
Verifies all prerequisites before touching any code.

```bash
python scripts/preflight.py
# or auto-install missing packages:
python scripts/preflight.py --install
```

Checks: Python ≥ 3.10, `datasets`, `verifiers`, `prime` CLI, Prime login.

---

### Step 1 — Inspect Dataset
Loads the dataset and reports its exact schema before writing any code.

```bash
python scripts/inspect_dataset.py pubmed_qa
python scripts/inspect_dataset.py openai/gsm8k main train
```

Output includes column names, types, sample rows, and an auto-detected answer format (GSM8K `####`, numeric, boolean, multi-choice, or free text).

---

### Step 2 — Write the Environment
Creates three files in `environments/my_env/`:

```
environments/my_env/
├── my_env.py        ← reward function + load_environment()
├── pyproject.toml   ← package metadata and dependencies
└── README.md        ← required for Hub display
```

Environments use the [`verifiers`](https://github.com/willccbb/verifiers) library. See `references/environment_guide.md` for complete patterns: classification, multi-choice, judge-scored free text, multi-turn, and more.

---

### Step 3 — Unit Test the Reward
Auto-generates test cases from the environment's own dataset and runs sanity checks.

```bash
python scripts/test_reward.py environments/my_env/my_env.py
python scripts/test_reward.py environments/my_env/my_env.py --verbose
```

Sanity checks:
- Gold completions score higher than wrong completions
- At least one gold reward > 0.5
- At least one wrong reward < 0.5
- Not all rewards identical

```
✓ All sanity checks passed. Reward function looks correct.
  Proceed to: prime eval run <env-name> -m  -n 50
```

---

### Step 4 — Local Evaluation
Runs real model completions against the environment using Prime Inference.

```bash
prime env install my-env --with pip
prime eval run my-env -m qwen/qwen3-vl-30b-a3b-instruct -n 50
```

Uses your `prime login` credentials — no additional API key needed. See `prime inference models` for the full list of available models.

---

### Step 5 — Reward Distribution Check
Statistically validates that the reward signal is learnable before committing to a training run.

```bash
python scripts/check_reward_distribution.py
```

| Check | Pass Condition | Failure Means |
|-------|---------------|---------------|
| Sample count | n ≥ 20 | Too few samples |
| Not all zero | mean > 0.05 | Reward function broken |
| Not all ones | mean < 0.95 | Trivially easy / reward hacked |
| Variance | std > 0.05 | Training signal too weak |
| Format reward | > 0.3 | Model can't follow format |

All 5 must pass before pushing to hub.

---

### Step 6 — Push to Hub
Publishes the environment to the Prime Environments Hub.

```bash
prime env push my-env
```

Verify at: `https://app.primeintellect.ai/dashboard/environments`

---

### Step 7 — Submit Training
Generates a TOML config and submits a hosted GRPO training run.

```bash
python scripts/submit_training.py --env your-username/my-env
# or with flags:
python scripts/submit_training.py --env your-username/my-env \
  --model Qwen/Qwen3-30B-Instruct-2507 \
  --steps 1000 \
  --yes
```

GPU allocation is fully managed — no instance config needed.

Monitor your run:
```bash
prime rl list
prime rl logs <run-id> -f
prime rl metrics <run-id>
```

Dashboard: `https://app.primeintellect.ai/dashboard/training`

---

## Fast Path — Existing Hub Environment

If the environment already exists on the Prime Hub, skip Steps 1–6.

**First, disambiguate** — both HF datasets and Prime environments use `owner/name` format:

```bash
prime env info owner/name
# exit 0  → Prime environment → use fast path
# exit 1  → HF dataset or not found → follow Steps 1–7
```

Then submit directly:

```bash
python scripts/submit_training.py --env zain/OpenMed_PubMedQA
```

---


## Project Structure

```
prime-lab-trainer/
│
├── SKILL.md                        ← Claude Code skill definition (agent instructions)
│
├── scripts/
│   ├── preflight.py                ← Step 0: verify setup
│   ├── inspect_dataset.py          ← Step 1: inspect HF dataset schema
│   ├── test_reward.py              ← Step 3: unit test reward functions
│   ├── check_reward_distribution.py← Step 5: validate reward distribution
│   └── submit_training.py          ← Step 7: generate TOML + submit
│
├── references/
│   ├── environment_guide.md        ← Full environment authoring reference
│   └── training_guide.md           ← Full training config reference
│
├── environments/                   ← Your environments go here
│   └── my_env/
│       ├── my_env.py
│       ├── pyproject.toml
│       └── README.md
│
└── configs/
    └── rl/                         ← Auto-generated TOML configs
        └── my-env.toml
```

---


---

## References

- [Prime Intellect Docs](https://docs.primeintellect.ai)
- [Verifiers Library](https://github.com/willccbb/verifiers)
- [Prime Environments Hub](https://app.primeintellect.ai/dashboard/environments)
- [Prime Training Dashboard](https://app.primeintellect.ai/dashboard/training)
- [Prime Inference Models](https://api.pinference.ai/api/v1/models)
