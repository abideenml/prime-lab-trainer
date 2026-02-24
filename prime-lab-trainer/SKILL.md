---
name: prime-lab-trainer
description: >
  Use when the user wants to build, validate, or submit RL training environments
  on Prime Intellect Lab using the verifiers library. Covers environment creation
  from HuggingFace datasets, reward function authoring and validation, pushing to
  the Environments Hub, and submitting hosted training runs. Invoke for: GRPO,
  verifiers, prime-rl, prime rl run, Environments Hub, RL environment, reward
  function, SingleTurnEnv, vf.Rubric, prime eval run, prime rl run.
---

# Prime Lab Trainer Skill

You are an agent that builds, validates, and submits RL training environments on Prime Intellect Lab.

## MANDATORY WORKFLOW — Follow This Order Every Time

**Never skip a step. Never submit training before all validations pass.**

```
Step 0: Preflight Check        → scripts/preflight.py
Step 1: Inspect Dataset        → scripts/inspect_dataset.py
Step 2: Write Environment      → references/environment_guide.md
Step 3: Validate Reward (Unit) → scripts/test_reward.py
Step 4: Local Eval             → prime eval run <env-name> -m <model>
Step 5: Check Distribution     → scripts/check_reward_distribution.py
Step 6: Push to Hub            → prime env push
Step 7: Submit Training        → references/training_guide.md
```

**If any step fails or produces unexpected output, STOP and fix before proceeding.**

---

## Step 0 — Preflight Check (Always First)
Install required tools:
```bash
pip install prime verifiers datasets
```

Verify that all tools, libraries, and credentials are in place **before** starting work:

```bash
python scripts/preflight.py
# Or auto-install missing Python packages:
python scripts/preflight.py --install
```

This checks:
- Python >= 3.10
- `datasets` and `verifiers` libraries installed
- `prime` CLI on PATH and authenticated (`prime login`)
**All checks must pass before proceeding.**

If not logged in: `prime login`

> **HuggingFace auth is not required** for public datasets or pushing environments.
> Only needed if: (a) your dataset is private on HF, or (b) you want to publish
> the trained model checkpoint to HF Hub after training.

---

## FAST PATH — Existing Hub Environment

Both HuggingFace datasets and Prime environments use `owner/name` format. **Always disambiguate first:**

```bash
prime env info owner/name
```

- **Exit 0** → it's a Prime environment → use the Fast Path below
- **Exit 1 / 404** → it's an HF dataset (or doesn't exist) → follow Steps 1–7 to build a new environment

**If confirmed as a Prime environment**, skip Steps 1–6 entirely. Go straight to training:

```bash
python scripts/submit_training.py --env owner/env-name
```

This script will:
1. Confirm the environment exists (`prime env info`)
2. Prompt for model and hyperparameters (or accept `--model`, `--batch`, `--rollouts` flags)
3. Generate the TOML config file at `configs/rl/<env-name>.toml`
4. Show the config for review before submitting
5. Submit via `prime rl run configs/rl/<env-name>.toml`

**Do NOT re-validate the reward function or re-inspect the dataset** — the environment is already finalized and live on the Hub. Trust the user.

If the user also says "just submit" or "don't ask", pass `--yes` to skip the review step and submit immediately.

---

## Step 1 — Dataset Inspection (Always First)

Before writing any code, inspect the dataset to understand its exact schema:

```bash
python scripts/inspect_dataset.py <dataset_name_or_hf_path>
# Example: python scripts/inspect_dataset.py gsm8k
# Example: python scripts/inspect_dataset.py openai/gsm8k
```

Read the output carefully:
- Column names and types
- Sample rows (prompt and answer fields)
- **Answer format** — the script auto-detects: GSM8K `####`, plain numeric, boolean, multi-choice, short label, or free text

**Do not assume column names or answer format.** The inspect script will tell you exactly what columns exist and recommend the right parsing strategy for your dataset.

---

## Step 2 — Write the Environment

Read **references/environment_guide.md** for the full environment authoring spec. It includes complete examples for GSM8K math, classification, and multi-choice tasks, plus a decision table for choosing the right reward pattern.

**Every environment requires three files. Do not skip any:**

```
environments/my_env/
├── my_env.py        ← reward function + load_environment()
├── pyproject.toml   ← package metadata and dependencies
└── README.md        ← REQUIRED — Hub will not display without it
```

Create the README.md alongside the code. See `references/environment_guide.md` for the README template. At minimum it must describe: the dataset, expected input/output format, reward breakdown, and usage example.

Quick template (GSM8K numeric — adapt for your dataset type):

```python
# environments/my_env/my_env.py
import re
import verifiers as vf
from datasets import load_dataset

SYSTEM_PROMPT = """Solve the math problem step by step.
Show your reasoning, then give your final numeric answer inside <answer> tags.

Format:
[step-by-step reasoning]
<answer>[number only]</answer>"""

def load_environment(num_examples: int = -1) -> vf.Environment:
    train_ds = load_dataset("openai/gsm8k", "main", split="train")
    eval_ds  = load_dataset("openai/gsm8k", "main", split="test")
    if num_examples > 0:
        train_ds = train_ds.select(range(num_examples))

    # GSM8K has a "question" column — SingleTurnEnv auto-wraps it into ChatMessage format.
    # No manual rename_column or .map() needed.
    # "answer" column is already named correctly.

    # Only list tags you need to parse — don't include "think"
    # (verifiers v0.1.10+ warns about think tags for Qwen3/DeepSeek models)
    parser = vf.XMLParser(["answer"])

    # Declare parser as argument — runtime injects it when parser= is passed to Rubric
    async def correct_answer(completion, answer, parser) -> float:
        # Parse ground truth from GSM8K "#### 42" format
        gt_match = re.search(r"####\s*([\d,\.\-]+)", str(answer))
        if not gt_match:
            return 0.0
        gt = gt_match.group(1).replace(",", "").strip()
        # Parse model answer from <answer> tag
        model_ans = parser.parse_answer(completion)
        if model_ans is None:
            return 0.0
        model_ans = re.sub(r"[^\d\.\-]", "", str(model_ans)).strip()
        return 1.0 if model_ans == gt else 0.0

    # Pass parser= to Rubric so it is injected into reward functions by name
    rubric = vf.Rubric(
        funcs=[correct_answer, parser.get_format_reward_func()],
        weights=[1.0, 0.2],
        parser=parser,
    )
    env = vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        rubric=rubric,
        system_prompt=SYSTEM_PROMPT,
    )
    return env
```

---

## Step 3 — Unit Test the Reward Function (Always Before Eval)

```bash
python scripts/test_reward.py environments/my_env/my_env.py
# Or with verbose output:
python scripts/test_reward.py environments/my_env/my_env.py --verbose
```

This script **auto-generates test cases from your dataset** (works for any answer type: numeric, boolean, multi-choice, text). It:
1. Loads your environment and accesses its dataset
2. Detects the answer format and expected output structure
3. Generates gold/wrong/malformed completions for 3 dataset examples
4. Runs your reward function(s) on all test cases
5. **Checks sanity**: gold rewards > wrong rewards, not all identical, etc.

**Expected output:**
```
Detected answer format: gsm8k
Detected output tags: ['answer']
Generated 9 test cases

────────────────────────────────────────────────────────────────
SANITY CHECKS

  [✓ PASS] Gold rewards > wrong rewards
  [✓ PASS] At least one gold > 0.5
  [✓ PASS] At least one wrong < 0.5
  [✓ PASS] Not all rewards identical

✓ All sanity checks passed. Reward function looks correct.
```

**Alternative modes:**
- `--preset gsm8k` — run 5 hardcoded GSM8K test cases with exact-match (backward compatible)
- `--custom tests.json` — provide your own test cases as JSON

If any sanity check fails, fix the reward function before proceeding. Do NOT proceed to eval.

---

## Step 4 — Local Evaluation

**Evaluations use Prime Inference by default** — no extra API key needed beyond `prime login`.

```bash
# Install the local environment package so prime eval run can import it.
# (prime eval run also auto-discovers ./environments/ without this, but
#  explicit install is more reliable and mirrors how Hub deployments work.)
prime env install my-env --with pip

# Run eval via Prime Inference (default) — uses prime login credentials
prime eval run my-env -m openai/gpt-5-nano -n 50
```

**Note:** `prime eval run` uploads results to Prime Evals Hub by default. Pass `--skip-upload` to keep results local:
```bash
prime eval run my-env -m openai/gpt-5-nano -n 50 --skip-upload
```

You need real model completions to check the reward distribution in Step 5.

---

## Step 5 — Reward Distribution Check (Always After Eval)

```bash
python scripts/check_reward_distribution.py --results outputs/evals/my-env/results.jsonl --verbose
```


This reads the last eval results from Prime's local store and checks:

| Check | Pass Condition | Failure Means |
|-------|---------------|---------------|
| Sample count | n ≥ 20 | Too few samples to trust distribution |
| Not all zero | mean reward > 0.05 | Reward function broken / parsing wrong |
| Not all ones | mean reward < 0.95 | Reward function trivially easy / hacked |
| Variance | std > 0.05 | Training signal too weak to learn |
| Format reward | > 0.3 | Model can't follow format at all |

**All checks must pass before pushing to hub.**

---

## Step 6 — Push to Hub

```bash
prime env push my-env
# Or push to a team:
# prime env push my-env --team <team-username>
```

Verify on https://app.primeintellect.ai/dashboard/environments

---

## Step 7 — Submit Training

Read **references/training_guide.md** for full training config options.

Quick start — all config goes in a TOML file, then:

```bash
prime rl run configs/rl/my-env.toml
```

Use `python scripts/submit_training.py --env your-username/my-env` to auto-generate
the TOML and submit in one step. See `references/training_guide.md` for the full
annotated config reference including optional fields (eval, validation, difficulty
filtering, checkpoints, W&B, secrets).

---

## Hardware Reference

`prime rl run` uses **Prime Intellect Hosted Training** — GPU allocation is fully managed. You do not configure GPU count, instance type, or infrastructure. Everything goes in the TOML config file.

For reference, the approximate scale used per model:

| Model | Approximate Scale |
|-------|-------------------|
| Qwen3-4B (LoRA) | ~2× A100 80GB |
| Qwen3-30B MoE (LoRA) | ~4–8× H100 |
| Qwen3-235B MoE (LoRA) | ~16× H100 |

Monitor cost and usage: https://app.primeintellect.ai/dashboard/training

---

## Common Failure Patterns

**Reward always 0:** Column mapping wrong, parser not matching model output format, regex failing on answer format.

**Reward always 1:** Reward function too lenient (substring match on short answers), wrong column passed as `answer`.

**Low variance:** Not enough examples, dataset too easy/hard for the model, tolerance too tight.

**Training diverges:** Reward signal too sparse (< 5% positive), increase format reward weight or check system prompt.
