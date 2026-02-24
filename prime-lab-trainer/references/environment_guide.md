# Environment Authoring Guide

Reference for writing verifiers environments for Prime Intellect Lab.

---

## Core Concepts

An environment has three parts:

1. **Dataset** — HuggingFace `Dataset` with `prompt` or `question` column (required), `answer` column (optional but typical), `info` column (optional dict for rich metadata)
2. **Rubric** — One or more reward functions that score model completions
3. **Harness** — `SingleTurnEnv` (one response per prompt) or `MultiTurnEnv` (tool loops, multi-step)

For GSM8K-style math, always use `SingleTurnEnv`.

---

## Reward Function Signature

The verifiers runtime injects kwargs **by name** from the rollout context. Name your arguments correctly:

```python
async def my_reward(completion, answer, prompt=None, state=None) -> float:
    ...
```

| Arg name | Type | What it is |
|----------|------|-----------|
| `completion` | `list[dict]` | Full message history. Last element is assistant's response. Access via `completion[-1]["content"]` |
| `answer` | `str` | The `answer` column value from the dataset row |
| `prompt` | `str` | The `prompt` column value |
| `info` | `str` | The `info` column value (JSON string or dict) |
| `state` | `dict` | Mutable shared state across reward functions in same rubric |
| `parser` | `vf.XMLParser` | Injected automatically when `vf.Rubric(parser=parser)` is used |
| `judge_client` | Prime Inference client | Injected if using `vf.JudgeRubric` |

**Always return a float between 0.0 and 1.0.**

---

## Dataset Patterns

### Option A — `question` Column (Auto-Wrapped)

If your dataset has a `question` string column, the environment wraps it in a user message automatically. No manual mapping needed:

```python
from datasets import Dataset

dataset = Dataset.from_list([
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "What is 3*5?", "answer": "15"},
])

# system_prompt prepends a system message automatically
env = vf.SingleTurnEnv(dataset=dataset, system_prompt="...", rubric=rubric)
# Results in: [{"role": "system", ...}, {"role": "user", "content": "What is 2+2?"}]
```

### Option B — `prompt` Column (Manual ChatMessage)

Use this when you need to compose multi-field prompts (e.g. question + context):

```python
dataset = dataset.map(lambda row: {
    "prompt": [{"role": "user", "content": f"{row['question']}\n\n{row['context']}"}],
    "answer": row["label"],
})
```

If the dataset already has a `prompt` column, `question` is ignored. A `system_prompt` passed to the environment will be prepended to prompts that don't already start with a system message.

### `info` Column for Metadata

Use `info` for structured metadata that reward functions may need but isn't the answer itself:

```python
dataset = dataset.map(lambda row: {
    "prompt": [...],
    "answer": row["label"],
    "info": json.dumps({"difficulty": row["difficulty"], "category": row["category"]}),
})

async def difficulty_bonus(completion, answer, info) -> float:
    meta = json.loads(info)
    base = 1.0 if model_correct else 0.0
    return base * (1.0 + 0.1 * meta["difficulty"])
```

Prefer JSON strings in `info` when rows may have different schemas.

### Separate Eval Dataset

Pass a separate eval split so `prime eval run` uses held-out data instead of training data.

**If the dataset has a dedicated test split:**
```python
def load_environment(num_examples: int = -1) -> vf.Environment:
    train_ds = load_dataset("openai/gsm8k", "main", split="train")
    eval_ds  = load_dataset("openai/gsm8k", "main", split="test")

    return vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,   # used by prime eval run by default
        rubric=rubric,
        system_prompt=SYSTEM_PROMPT,
    )
```

**If the dataset only has a `train` split**, carve out a held-out eval set manually:
```python
def load_environment(num_examples: int = -1) -> vf.Environment:
    full_ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    splits   = full_ds.train_test_split(test_size=0.1, seed=42)
    train_ds = splits["train"]
    eval_ds  = splits["test"]

    return vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        rubric=rubric,
        system_prompt=SYSTEM_PROMPT,
    )
```

Always check available splits before writing `load_environment()` — use `inspect_dataset.py` or check the dataset card on HuggingFace.

### Lazy Loading with DatasetBuilder

For large datasets or multiple environment replicas, defer loading using a callable:

```python
def load_environment() -> vf.Environment:
    def build_train():
        ds = load_dataset("my-dataset", split="train")
        return ds.shuffle(seed=42)

    def build_eval():
        return load_dataset("my-dataset", split="test")

    return vf.SingleTurnEnv(
        dataset=build_train,       # called on first access
        eval_dataset=build_eval,   # called on first access
        rubric=rubric,
    )
```

Useful when dataset loading is expensive or you want to parameterize creation without loading immediately.

---

## GSM8K Answer Parsing Pattern

GSM8K answers look like: `"She bought 3 apples. 3 * 2 = 6. #### 6"`

The ground truth number is after `####`. Parse it inline inside your reward function:

```python
gt_match = re.search(r"####\s*([\d,\.\-]+)", str(answer))
if not gt_match:
    return 0.0
gt_num = gt_match.group(1).replace(",", "").strip()
```

Model output parsing — use XMLParser and pass it to the Rubric for injection:

```python
# Only list tags you need to PARSE — don't include "think"
# (verifiers v0.1.10+ warns about think tags for Qwen3/DeepSeek models)
parser = vf.XMLParser(["answer"])

# Pass parser to Rubric — it is then injected into reward functions by name
rubric = vf.Rubric(
    funcs=[correct_answer, parser.get_format_reward_func()],
    weights=[1.0, 0.2],
    parser=parser,   # ← required for injection
)

# Declare `parser` as an argument — runtime injects it automatically
async def correct_answer(completion, answer, parser) -> float:
    model_ans = parser.parse_answer(completion)  # extracts <answer> tag content
    if model_ans is None:
        return 0.0
    ...
```

If you want to accept unstructured output as fallback:

```python
def parse_model_number(text: str) -> str | None:
    """Extract the last number from model output."""
    matches = re.findall(r"[\d,]+\.?\d*", text)
    if not matches:
        return None
    return matches[-1].replace(",", "").strip()
```

---

## Complete GSM8K Environment

```python
import re
import verifiers as vf
from datasets import load_dataset

SYSTEM_PROMPT = """Solve the math problem step by step.
Show your reasoning, then give your final numeric answer inside <answer> tags.

Format:
[step-by-step reasoning]
<answer>[number only]</answer>"""

def load_environment(
    split: str = "train",
    num_examples: int = -1,
    system_prompt: str = SYSTEM_PROMPT,
) -> vf.Environment:

    train_ds = load_dataset("openai/gsm8k", "main", split="train")
    eval_ds  = load_dataset("openai/gsm8k", "main", split="test")

    if num_examples > 0:
        train_ds = train_ds.select(range(num_examples))

    # GSM8K has "question" column — environment auto-wraps it into ChatMessage format.
    # "answer" column is already named correctly.

    # Only list tags you need to parse — don't include "think"
    # (verifiers v0.1.10+ warns about think tags for Qwen3/DeepSeek models)
    parser = vf.XMLParser(["answer"])

    async def correct_answer(completion, answer, parser) -> float:
        """Primary reward: is the final numeric answer correct?"""
        # Parse ground truth from #### format
        gt = re.search(r"####\s*([\d,\.\-]+)", str(answer))
        if gt is None:
            return 0.0
        gt_num = gt.group(1).replace(",", "").strip()

        # Parse model output from <answer> tag (injected via Rubric)
        model_ans = parser.parse_answer(completion)
        if model_ans is None:
            return 0.0

        # Strip everything except digits, decimal point, minus sign
        model_num = re.sub(r"[^\d\.\-]", "", str(model_ans)).strip()

        return 1.0 if model_num == gt_num else 0.0

    # Pass parser= to Rubric so it is injected into reward function arguments
    rubric = vf.Rubric(
        funcs=[correct_answer, parser.get_format_reward_func()],
        weights=[1.0, 0.2],
        parser=parser,
    )

    env = vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        rubric=rubric,
        system_prompt=system_prompt,
    )
    return env
```

---

## Using vf.MathRubric for Symbolic Math

For MATH-style datasets that use `\boxed{}` answers, use the built-in `vf.MathRubric` instead of writing manual regex. It uses the `math-verify` library for symbolic equivalence (so `1/2`, `0.5`, and `\frac{1}{2}` all match):

```python
import verifiers as vf
from datasets import load_dataset

SYSTEM_PROMPT = """Solve the problem step by step.
Put your final answer in \\boxed{}.

Format:
[step-by-step reasoning]
\\boxed{[answer]}"""

def load_environment(split: str = "train") -> vf.Environment:
    dataset = load_dataset("lighteval/MATH", split=split)
    # dataset has "problem" and "solution" columns

    # MathRubric includes correct_answer reward that parses \boxed{} with symbolic equivalence
    rubric = vf.MathRubric()

    return vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        system_prompt=SYSTEM_PROMPT,
    )
```

Use `vf.MathRubric` for: MATH, AMC, AIME, competition math, or any task where answers need symbolic equivalence.
Use manual regex only for: GSM8K (uses `####` format, plain integers).

---

## Choosing Your Reward Pattern

| Answer type | Recommended approach | Example datasets |
|---|---|---|
| Numeric with `####` | Manual regex + exact match | GSM8K |
| `\boxed{}` math | `vf.MathRubric()` | MATH, AMC, AIME |
| Plain numeric | `strip()` + exact match | AQuA, numerical QA |
| Exact text / label | `.strip().lower()` == | Classification, sentiment, boolean |
| Multi-choice (A-D) | Single-char extraction + match | MMLU, ARC, HellaSwag |
| Free text | `vf.JudgeRubric` | Open QA, summarization |

---

## Complete Classification Environment

```python
import verifiers as vf
from datasets import load_dataset

SYSTEM_PROMPT = """Classify the sentiment of the given text.
Respond with exactly one word: positive, negative, or neutral.

Show your reasoning, then give your classification in <answer> tags.

Format:
[your reasoning]
<answer>[positive/negative/neutral]</answer>"""

def load_environment(
    split: str = "train",
    num_examples: int = -1,
    system_prompt: str = SYSTEM_PROMPT,
) -> vf.Environment:

    train_ds = load_dataset("stanfordnlp/sst2", split="train")
    eval_ds  = load_dataset("stanfordnlp/sst2", split="validation")

    if num_examples > 0:
        train_ds = train_ds.select(range(num_examples))

    # Map SST-2 columns to verifiers standard
    label_map = {0: "negative", 1: "positive"}

    def format_row(row):
        return {
            "prompt": [{"role": "user", "content": row["sentence"]}],
            "answer": label_map[row["label"]],
        }

    train_ds = train_ds.map(format_row).remove_columns(["sentence", "label", "idx"])
    eval_ds  = eval_ds.map(format_row).remove_columns(["sentence", "label", "idx"])

    parser = vf.XMLParser(["answer"])

    async def correct_label(completion, answer, parser) -> float:
        """Primary reward: does the model output the correct label?"""
        model_ans = parser.parse_answer(completion)
        if model_ans is None:
            return 0.0
        return 1.0 if model_ans.strip().lower() == answer.strip().lower() else 0.0

    # Pass parser= so it is injected into correct_label by name
    rubric = vf.Rubric(
        funcs=[correct_label, parser.get_format_reward_func()],
        weights=[1.0, 0.2],
        parser=parser,
    )

    env = vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        rubric=rubric,
        system_prompt=system_prompt,
    )
    return env
```

---

## Complete Multi-Choice Environment

```python
import re
import verifiers as vf
from datasets import load_dataset

SYSTEM_PROMPT = """Answer the multiple-choice question.
Show your reasoning, then give your answer letter in <answer> tags.

Format:
[your reasoning]
<answer>[A/B/C/D]</answer>"""

def load_environment(
    num_examples: int = -1,
    system_prompt: str = SYSTEM_PROMPT,
) -> vf.Environment:

    # MMLU has "validation" and "test" splits (no "train" per-subject).
    # Use "validation" for training, "test" as the held-out eval split.
    train_ds = load_dataset("cais/mmlu", "abstract_algebra", split="validation")
    eval_ds  = load_dataset("cais/mmlu", "abstract_algebra", split="test")

    if num_examples > 0:
        train_ds = train_ds.select(range(num_examples))

    # Build formatted prompt with choices, in ChatMessage format
    LETTERS = ["A", "B", "C", "D"]
    def format_row(row):
        choices = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(row["choices"]))
        return {
            "prompt": [{"role": "user", "content": f"{row['question']}\n\n{choices}"}],
            "answer": LETTERS[row["answer"]],
        }
    train_ds = train_ds.map(format_row).remove_columns(["question", "choices", "subject"])
    eval_ds  = eval_ds.map(format_row).remove_columns(["question", "choices", "subject"])

    parser = vf.XMLParser(["answer"])

    async def correct_choice(completion, answer, parser) -> float:
        """Primary reward: did the model pick the right letter?"""
        model_ans = parser.parse_answer(completion)
        if model_ans is None:
            return 0.0
        # Extract single letter from model output
        m = re.search(r"[A-Da-d]", model_ans.strip())
        if not m:
            return 0.0
        return 1.0 if m.group().upper() == answer.strip().upper() else 0.0

    # Pass parser= so it is injected into correct_choice by name
    rubric = vf.Rubric(
        funcs=[correct_choice, parser.get_format_reward_func()],
        weights=[1.0, 0.2],
        parser=parser,
    )

    env = vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        rubric=rubric,
        system_prompt=system_prompt,
    )
    return env
```

---

## Rubric Weights

`weights` scales each reward function's contribution to the total reward used for RL updates:

```python
rubric = vf.Rubric(
    funcs=[correct_answer, format_reward],
    weights=[1.0, 0.2],
    # total = 1.0 * correct_answer(.) + 0.2 * format_reward(.)
    # range ≈ [0.0, 1.2]
)
```

Rules of thumb:
- Correctness weight should dominate (≥ 0.7 of total)
- Format reward at 0.1–0.3 is enough to encourage structure without gaming
- Weight `0.0` means "log but don't train on" (useful for monitor metrics)

### Adding Metrics (Log-Only Reward Functions)

Use `add_metric()` to log a signal without including it in training reward:

```python
async def response_length(completion) -> float:
    return float(len(completion[-1]["content"]))

rubric = vf.Rubric(funcs=[correct_answer], weights=[1.0])
rubric.add_metric(response_length)  # shorthand for weight=0; appears in rollout metrics
```

Equivalent to `rubric.add_reward_func(response_length, weight=0)` but more explicit.

---

## Group-Based Reward Functions

During RL training, rollouts are organized into groups from the same input (e.g. 16 rollouts per example for GRPO advantage computation). Group-based reward functions receive all completions for an example at once and return a list of scores.

Use **plural argument names** to signal a group-level function:

```python
async def diversity_bonus(completions) -> list[float]:
    """Reward unique responses within a group."""
    responses = [c[-1]["content"] for c in completions]
    unique = set(responses)
    return [0.2 if responses.count(r) == 1 else 0.0 for r in responses]

async def relative_length(completions) -> list[float]:
    """Reward responses shorter than the group average."""
    lengths = [len(c[-1]["content"]) for c in completions]
    avg = sum(lengths) / len(lengths)
    return [1.0 if l < avg else 0.0 for l in lengths]

rubric = vf.Rubric(
    funcs=[correct_answer, diversity_bonus],
    weights=[1.0, 0.1],
)
```

Plural arg names: `completions`, `prompts`, `answers`, `infos`, `states`.

---

## Shared State Between Reward Functions

When two reward functions share expensive computation:

```python
async def parse_and_score(completion, answer, state, parser) -> float:
    # Store parsed value for next function
    model_ans = parser.parse_answer(completion)
    state["model_ans"] = model_ans
    gt = re.search(r"####\s*([\d,\.\-]+)", answer)
    state["gt"] = gt.group(1).replace(",", "") if gt else None
    if state["model_ans"] is None or state["gt"] is None:
        return 0.0
    return 1.0 if state["model_ans"] == state["gt"] else 0.0

async def partial_credit(state) -> float:
    # Reuse parsed values from previous function
    if state.get("model_ans") is None or state.get("gt") is None:
        return 0.0
    try:
        ratio = float(state["model_ans"]) / float(state["gt"])
        return max(0.0, 1.0 - abs(1.0 - ratio))  # 0.9 for 10% off
    except (ValueError, ZeroDivisionError):
        return 0.0

parser = vf.XMLParser(["answer"])
rubric = vf.Rubric(
    funcs=[parse_and_score, partial_credit],
    weights=[1.0, 0.0],  # log partial credit but don't train on it
    parser=parser,
)
```

---

## JudgeRubric for Free-Text Tasks

For tasks where deterministic evaluation is impractical, use `vf.JudgeRubric`:

```python
import verifiers as vf
from datasets import load_dataset

def load_environment() -> vf.Environment:
    vf.ensure_keys(["PRIME_API_KEY"])  # fail early if not logged in via prime login

    dataset      = load_dataset("my-org/my-dataset", split="train")
    eval_dataset = load_dataset("my-org/my-dataset", split="test")

    judge_rubric = vf.JudgeRubric(judge_model="openai/gpt-5-mini")

    async def judge_correctness(prompt, completion, answer, judge) -> float:
        verdict = await judge(prompt, completion, answer)
        return 1.0 if "yes" in verdict.lower() else 0.0

    judge_rubric.add_reward_func(judge_correctness)

    return vf.SingleTurnEnv(dataset=dataset, eval_dataset=eval_dataset, rubric=judge_rubric)
```

---

## Validating Required API Keys

Environments that require external API keys should fail early with a clear message using `vf.ensure_keys()`:

```python
import verifiers as vf

def load_environment(api_key_var: str = "PRIME_API_KEY") -> vf.Environment:
    vf.ensure_keys([api_key_var])
    # raises MissingKeyError listing all missing keys and how to set them
    # safe to use os.environ[api_key_var] below this line
    ...
```

Document required variables in your README under a "Required Environment Variables" section.

---

## Environment Module Structure

Always scaffold a new environment with `prime env init` — it creates all three files with correct defaults:

```bash
prime env init my-env
# creates: environments/my_env/my_env.py, pyproject.toml, README.md
```

```
environments/my_env/
├── my_env.py        ← load_environment() lives here (edit this)
├── pyproject.toml   ← package metadata (edit dependencies and tags)
└── README.md        ← describes the task, required env vars
```

Then fill in `my_env.py` with your dataset, reward function, and rubric using the patterns in this guide. **README.md is required** — the Hub will not display your environment correctly without it, and `prime env push` will warn.

### README.md Template

```markdown
# my-env

One-line description of what the task is.

## Task

- **Dataset:** `owner/dataset-name` (config, split)
- **Input:** What the model receives
- **Output:** `<answer>[expected format]</answer>`
- **Reward:** 1.0 for correct, 0.0 otherwise (+ 0.2 format bonus)

## Example

**Prompt:**
\```
The user message shown to the model
\```

**Expected response:**
\```
[reasoning]
<answer>correct answer</answer>
\```

## Reward

| Component | Weight | Description |
|-----------|--------|-------------|
| `correct_answer` | 1.0 | Brief description of correctness check |
| `format_reward`  | 0.2 | Model used `<answer>` tags correctly |

## Usage

\```python
from my_env import load_environment

env = load_environment()
env = load_environment(num_examples=100)  # subset for testing
\```

## Environment Variables

None required for standard environments.
<!-- For JudgeRubric environments, add:
- `PRIME_API_KEY` — set automatically via `prime login`; used for judge model calls
-->
```

`pyproject.toml` — update after `prime env init`:

```toml
[project]
name = "my-env"
description = "One-line description of the task"
tags = ["single-turn", "classification"]  # optional Hub metadata
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["verifiers>=0.1.8", "datasets"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
include = ["my_env.py", "pyproject.toml"]

[tool.verifiers.eval]
num_examples = 20        # default for prime eval run (overridable with -n)
rollouts_per_example = 5
```

---

## Common Mistakes

**Using `completion` as a string:**
```python
# WRONG
if answer in completion:
    return 1.0

# RIGHT
response = completion[-1]["content"]
if answer in response:
    return 1.0
```

**Not handling None from parser:**
```python
# WRONG — crashes when model doesn't include <answer> tag
model_ans = parser.parse_answer(completion)
return 1.0 if model_ans.strip() == gt else 0.0

# RIGHT
model_ans = parser.parse_answer(completion)
if model_ans is None:
    return 0.0
return 1.0 if model_ans.strip() == gt else 0.0
```

**Substring match on short answers:**
```python
# WRONG — "1" matches in "10", "21", "100"
return 1.0 if gt in response else 0.0

# RIGHT — exact match after parsing
return 1.0 if model_num == gt_num else 0.0
```

**Sync reward function doing I/O:**
```python
# WRONG — will block the async event loop
def judge_reward(completion, answer) -> float:
    result = requests.post(...)  # blocking I/O

# RIGHT
async def judge_reward(completion, answer) -> float:
    result = await async_client.post(...)
```

**Parser not passed to Rubric:**
```python
# WRONG — parser not injected; reward function must close over it manually
parser = vf.XMLParser(["answer"])
async def my_reward(completion, answer) -> float:
    model_ans = parser.parse_answer(completion)  # closure, not injection

rubric = vf.Rubric(funcs=[my_reward], weights=[1.0])

# RIGHT — declare parser as arg, pass parser= to Rubric for injection
async def my_reward(completion, answer, parser) -> float:
    model_ans = parser.parse_answer(completion)

rubric = vf.Rubric(funcs=[my_reward], weights=[1.0], parser=parser)
```

**Including "think" in XMLParser tags:**
```python
# WRONG — causes warnings in verifiers v0.1.10+ with Qwen3/DeepSeek models
parser = vf.XMLParser(["think", "answer"])

# RIGHT — only tags the model explicitly outputs in its answer
parser = vf.XMLParser(["answer"])
```
