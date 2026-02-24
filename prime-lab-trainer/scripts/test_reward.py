#!/usr/bin/env python3
"""
Step 3: Reward Function Unit Tests
Run after writing the environment, before local eval.

Usage:
  python scripts/test_reward.py environments/my_env/my_env.py
  python scripts/test_reward.py environments/my_env/my_env.py --preset gsm8k
  python scripts/test_reward.py environments/my_env/my_env.py --custom tests.json
  python scripts/test_reward.py environments/my_env/my_env.py --verbose

Modes:
  Auto (default)    Auto-generate test cases from the environment's own dataset.
                    Works for any answer type: numeric, boolean, multi-choice, text.
  --preset gsm8k   Use 5 hardcoded GSM8K math test cases (backward compatible).
  --custom FILE    Load test cases from a JSON file (advanced).

Auto mode detects the answer format, generates gold/wrong/malformed completions,
and runs format-agnostic sanity checks instead of exact-match assertions.
"""

import re
import sys
import json
import asyncio
import importlib.util
import argparse
from pathlib import Path


# ─── Answer Format Classification ────────────────────────────────────────────

BOOLEAN_VALUES = {"yes", "no", "true", "false", "0", "1"}

def classify_answer_format(samples: list[str]) -> str:
    """Classify the answer column format from a list of sample values."""
    if not samples:
        return "long_text"

    # GSM8K: contains #### marker
    gsm_count = sum(1 for s in samples if "####" in s)
    if gsm_count > len(samples) * 0.5:
        return "gsm8k"

    stripped = [s.strip() for s in samples]

    # Boolean: all answers in boolean set
    if all(s.lower() in BOOLEAN_VALUES for s in stripped if s):
        return "boolean"

    # Multi-choice: all answers are single letter A-E or digit 1-5
    if all(re.match(r"^[A-Ea-e1-5]$", s) for s in stripped if s):
        return "multi_choice"

    # Plain numeric
    numeric_count = sum(1 for s in stripped if re.match(r"^[\d,\.\-]+$", s))
    if numeric_count > len(samples) * 0.5:
        return "numeric"

    # Short label: all under 30 chars
    if all(len(s) < 30 for s in stripped):
        return "short_label"

    return "long_text"


def extract_gold_answer(answer: str, fmt: str) -> str:
    """Extract the usable answer value from a dataset answer column."""
    if fmt == "gsm8k":
        m = re.search(r"####\s*([\d,\.\-]+)", answer)
        if m:
            return m.group(1).replace(",", "").strip()
        return answer.strip()
    if fmt == "boolean":
        return answer.strip().lower()
    if fmt == "multi_choice":
        return answer.strip().upper()[0] if answer.strip() else answer
    # numeric, short_label, long_text
    return answer.strip()


def generate_wrong_answer(gold: str, fmt: str) -> str:
    """Generate a clearly wrong answer for the given format."""
    if fmt in ("gsm8k", "numeric"):
        return "999999"
    if fmt == "boolean":
        return "false" if gold.lower() in ("true", "yes", "1") else "true"
    if fmt == "multi_choice":
        # Rotate to next letter
        letters = "ABCDE"
        idx = letters.find(gold.upper())
        return letters[(idx + 1) % len(letters)] if idx >= 0 else "Z"
    # short_label, long_text
    return "DELIBERATELY_WRONG_ANSWER"


# ─── Output Format Detection ─────────────────────────────────────────────────

def detect_output_tags(env, module_path: Path) -> list[str] | None:
    """Detect XML tags the environment expects in model output.
    Tries env.system_prompt first, then falls back to scanning module source.
    """
    # Try system_prompt attribute
    system_prompt = getattr(env, "system_prompt", None) or ""
    tags = _extract_tags_from_text(system_prompt)
    if tags:
        return tags

    # Fall back: scan module source for XMLParser(["tag1", "tag2"])
    try:
        source = module_path.read_text()
        m = re.search(r'XMLParser\(\s*\[([^\]]+)\]', source)
        if m:
            raw = m.group(1)
            tags = re.findall(r'["\'](\w+)["\']', raw)
            if tags:
                return tags
    except Exception:
        pass

    return None


def _extract_tags_from_text(text: str) -> list[str] | None:
    """Extract XML-style tag names from a system prompt or instruction text."""
    if not text:
        return None
    tags = re.findall(r"<(\w+)>", text)
    # Deduplicate preserving order
    seen = set()
    unique = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique if unique else None


def format_completion(answer_val: str, tags: list[str] | None) -> str:
    """Wrap an answer value in the expected output format."""
    if not tags:
        return answer_val

    # Use the last tag as the answer tag, earlier tags as reasoning
    if len(tags) >= 2:
        reasoning_tag = tags[0]
        answer_tag = tags[-1]
        return f"<{reasoning_tag}>step-by-step reasoning</{reasoning_tag}>\n<{answer_tag}>{answer_val}</{answer_tag}>"
    else:
        tag = tags[0]
        return f"<{tag}>{answer_val}</{tag}>"


# ─── Rubric Access ────────────────────────────────────────────────────────────

def extract_reward_funcs(rubric) -> list:
    """Extract reward functions from a Rubric or RubricGroup.
    verifiers v0.1.10+ wraps user rubrics in a RubricGroup.
    """
    # Direct Rubric with funcs populated
    if rubric.funcs:
        return rubric.funcs

    # RubricGroup: collect from nested rubrics (skip weight-0 system rubrics)
    if hasattr(rubric, "rubrics"):
        funcs = []
        for r in rubric.rubrics:
            for func, weight in zip(r.funcs, r.weights):
                if weight > 0:
                    funcs.append(func)
        if funcs:
            return funcs
        # If all weights are 0, return all funcs anyway for testing
        for r in rubric.rubrics:
            funcs.extend(r.funcs)
        return funcs

    return []


def extract_prompt_text(prompt_val) -> str:
    """Extract plain text from a prompt that may be a string or ChatMessages list."""
    if isinstance(prompt_val, str):
        return prompt_val
    if isinstance(prompt_val, list):
        # ChatMessage format — find the user message
        for msg in prompt_val:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        # Fallback: last message content
        if prompt_val and isinstance(prompt_val[-1], dict):
            return prompt_val[-1].get("content", str(prompt_val))
    return str(prompt_val)


# ─── Test Case Generation ────────────────────────────────────────────────────

def generate_test_cases(
    dataset,
    answer_fmt: str,
    tags: list[str] | None,
    num_examples: int = 3,
) -> list[dict]:
    """Generate gold/wrong/malformed test cases from real dataset examples."""
    cases = []
    n = min(num_examples, len(dataset))

    for i in range(n):
        row = dataset[i]
        prompt_raw = row.get("prompt", row.get("question", ""))
        prompt = extract_prompt_text(prompt_raw)
        answer_raw = str(row.get("answer", row.get("solution", row.get("target", ""))))
        gold = extract_gold_answer(answer_raw, answer_fmt)
        wrong = generate_wrong_answer(gold, answer_fmt)

        user_msg = {"role": "user", "content": prompt[:200]}

        # Gold: correct answer in correct format
        cases.append({
            "name": f"Example {i}: gold answer in correct format",
            "completion": [user_msg, {"role": "assistant", "content": format_completion(gold, tags)}],
            "answer": answer_raw,
            "expect": "high",
        })

        # Wrong: wrong answer in correct format
        cases.append({
            "name": f"Example {i}: wrong answer in correct format",
            "completion": [user_msg, {"role": "assistant", "content": format_completion(wrong, tags)}],
            "answer": answer_raw,
            "expect": "low",
        })

        # Malformed: correct answer value but no formatting
        cases.append({
            "name": f"Example {i}: malformed output (no formatting)",
            "completion": [user_msg, {"role": "assistant", "content": f"I think the answer is {gold}"}],
            "answer": answer_raw,
            "expect": "low",
        })

    return cases


# ─── GSM8K Preset ─────────────────────────────────────────────────────────────

GSM8K_PRESET = [
    {
        "name": "Correct answer in <answer> tag",
        "completion": [
            {"role": "user", "content": "Janet has 3 ducks. Each duck lays 16 eggs per day. How many eggs does she get per week?"},
            {"role": "assistant", "content": "<think>3 * 16 * 7 = 336</think>\n<answer>336</answer>"},
        ],
        "answer": "Janet's ducks lay 3 \u00d7 16 = 48 eggs per day. Per week: 48 \u00d7 7 = 336. #### 336",
        "expected_reward": 1.0,
    },
    {
        "name": "Wrong answer in <answer> tag",
        "completion": [
            {"role": "user", "content": "If you have 5 apples and eat 2, how many remain?"},
            {"role": "assistant", "content": "<think>5 - 2 = 4</think>\n<answer>4</answer>"},
        ],
        "answer": "5 - 2 = 3. #### 3",
        "expected_reward": 0.0,
    },
    {
        "name": "Malformed output (no XML tags)",
        "completion": [
            {"role": "user", "content": "What is 10 + 5?"},
            {"role": "assistant", "content": "The answer is 15. I calculated this by adding 10 and 5."},
        ],
        "answer": "10 + 5 = 15. #### 15",
        "expected_reward": 0.0,
    },
    {
        "name": "Answer with comma formatting (1,234 should equal 1234)",
        "completion": [
            {"role": "user", "content": "A factory produces 1000 units per day. How many in 1234 days? Wait, just compute 617 * 2."},
            {"role": "assistant", "content": "<think>617 * 2 = 1234</think>\n<answer>1,234</answer>"},
        ],
        "answer": "617 \u00d7 2 = 1234. #### 1234",
        "expected_reward": 1.0,
    },
    {
        "name": "Correct answer but extra text after tag",
        "completion": [
            {"role": "user", "content": "What is 7 * 8?"},
            {"role": "assistant", "content": "<think>7 * 8 = 56</think>\n<answer>56</answer>\nHope that helps!"},
        ],
        "answer": "7 \u00d7 8 = 56. #### 56",
        "expected_reward": 1.0,
    },
]


# ─── Call Reward Function ─────────────────────────────────────────────────────

async def call_reward(func, completion, answer) -> float | None:
    """Call a reward function, handling kwargs vs positional dispatch."""
    try:
        return await func(completion=completion, answer=answer)
    except TypeError:
        try:
            return await func(completion, answer)
        except Exception as e:
            print(f"    ERROR calling {func.__name__}: {e}")
            return None


# ─── Preset Runner ────────────────────────────────────────────────────────────

async def run_preset_tests(
    reward_funcs: list,
    preset: list[dict],
    verbose: bool = False,
) -> bool:
    """Run preset test cases with exact-match expectations (backward compat)."""
    correctness_func = reward_funcs[0]
    print(f"Found {len(reward_funcs)} reward function(s) in rubric")
    print(f"Testing correctness function: {correctness_func.__name__}\n")

    all_passed = True
    results = []

    for case in preset:
        reward = await call_reward(correctness_func, case["completion"], case["answer"])
        expected = case["expected_reward"]
        passed = (reward == expected) if reward is not None else False

        if not passed:
            all_passed = False

        status = "\u2713 PASS" if passed else "\u2717 FAIL"
        print(f"[{status}] {case['name']}")
        print(f"         reward={reward}  expected={expected}")

        if verbose or not passed:
            last_msg = case["completion"][-1]["content"]
            print(f"         completion: {last_msg[:100]!r}")
            print(f"         answer col: {case['answer'][:80]!r}")
        print()

        results.append({"case": case["name"], "passed": passed})

    print("\u2500" * 60)
    passed_count = sum(1 for r in results if r["passed"])
    print(f"Results: {passed_count}/{len(results)} tests passed\n")

    if all_passed:
        print("\u2713 All tests passed. Reward function is sane.")
        print("  Proceed to: prime eval run <env-name> -m openai/gpt-5-nano -n 50\n")
    else:
        print("\u2717 Some tests failed. Fix the reward function before running eval.")
        print("\nCommon fixes:")
        print("  - Check answer parsing matches your dataset format")
        print("  - Use parser.parse_answer(completion) not completion[-1]['content']")
        print("  - Ensure function signature: async def fn(completion, answer) -> float")
        sys.exit(1)

    return all_passed


# ─── Auto Runner ──────────────────────────────────────────────────────────────

async def run_auto_tests(
    reward_funcs: list,
    test_cases: list[dict],
    answer_fmt: str,
    verbose: bool = False,
) -> bool:
    """Run auto-generated tests with sanity-check assertions."""
    print(f"Found {len(reward_funcs)} reward function(s) in rubric")
    print(f"Detected answer format: {answer_fmt}")
    print(f"Generated {len(test_cases)} test cases\n")

    gold_rewards = []
    wrong_rewards = []
    malformed_rewards = []
    all_rewards = []

    for case in test_cases:
        # Run ALL reward functions and take weighted-ish sum (use first as primary)
        primary = reward_funcs[0]
        reward = await call_reward(primary, case["completion"], case["answer"])

        if reward is None:
            reward = 0.0

        bucket = case["expect"]  # "high" or "low"
        if "gold" in case["name"]:
            gold_rewards.append(reward)
        elif "wrong" in case["name"]:
            wrong_rewards.append(reward)
        else:
            malformed_rewards.append(reward)
        all_rewards.append(reward)

        if verbose:
            last_msg = case["completion"][-1]["content"]
            print(f"  [{bucket:4s}] {case['name']}")
            print(f"          reward={reward:.3f}  completion={last_msg[:80]!r}")
            print()

    # ── Sanity Checks ─────────────────────────────────────────
    print("\u2500" * 60)
    print("SANITY CHECKS\n")

    checks_passed = True

    mean_gold = sum(gold_rewards) / len(gold_rewards) if gold_rewards else 0.0
    mean_wrong = sum(wrong_rewards) / len(wrong_rewards) if wrong_rewards else 0.0
    mean_malformed = sum(malformed_rewards) / len(malformed_rewards) if malformed_rewards else 0.0

    checks = [
        (
            "Gold rewards > wrong rewards",
            mean_gold > mean_wrong,
            f"mean_gold={mean_gold:.3f}  mean_wrong={mean_wrong:.3f}",
            "Reward function doesn't differentiate correct from incorrect answers.\n"
            "  \u2192 Check answer parsing and comparison logic.",
        ),
        (
            "At least one gold > 0.5",
            any(r > 0.5 for r in gold_rewards) if gold_rewards else False,
            f"max_gold={max(gold_rewards):.3f}" if gold_rewards else "no gold cases",
            "Correct answers never score above 0.5.\n"
            "  \u2192 Check that the reward function can recognize a correct answer at all.",
        ),
        (
            "At least one wrong < 0.5",
            any(r < 0.5 for r in wrong_rewards) if wrong_rewards else False,
            f"min_wrong={min(wrong_rewards):.3f}" if wrong_rewards else "no wrong cases",
            "Wrong answers never score below 0.5 \u2014 reward may be trivially easy.\n"
            "  \u2192 Check for overly lenient matching (substring, type coercion).",
        ),
        (
            "Not all rewards identical",
            len(set(round(r, 6) for r in all_rewards)) > 1,
            f"unique_values={len(set(round(r, 6) for r in all_rewards))}",
            "Every test case got the same reward \u2014 function may be constant.\n"
            "  \u2192 Verify the function actually reads completion and answer args.",
        ),
    ]

    for name, passed, detail, failure_msg in checks:
        status = "\u2713 PASS" if passed else "\u2717 FAIL"
        print(f"  [{status}] {name}")
        print(f"         {detail}")
        if not passed:
            checks_passed = False
            for line in failure_msg.split("\n"):
                print(f"         {line}")
        print()

    # ── Summary ────────────────────────────────────────────────
    print("\u2500" * 60)
    print(f"  Gold    mean={mean_gold:.3f}  (n={len(gold_rewards)})")
    print(f"  Wrong   mean={mean_wrong:.3f}  (n={len(wrong_rewards)})")
    print(f"  Malform mean={mean_malformed:.3f}  (n={len(malformed_rewards)})")
    print()

    if checks_passed:
        print("\u2713 All sanity checks passed. Reward function looks correct.")
        print("  Proceed to: prime eval run <env-name> -m openai/gpt-5-nano -n 50\n")
    else:
        print("\u2717 Sanity checks failed. Fix the reward function before running eval.")
        print("  Re-run with --verbose to see individual test case details.\n")
        sys.exit(1)

    return checks_passed


# ─── Main ─────────────────────────────────────────────────────────────────────

async def run(
    env_module_path: str,
    preset: str | None = None,
    custom_path: str | None = None,
    verbose: bool = False,
):
    path = Path(env_module_path)
    if not path.exists():
        print(f"ERROR: File not found: {env_module_path}")
        sys.exit(1)

    # Dynamically import the environment module
    spec = importlib.util.spec_from_file_location("user_env", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"ERROR: Failed to import {env_module_path}: {e}")
        sys.exit(1)

    if not hasattr(mod, "load_environment"):
        print(f"ERROR: Module {env_module_path} has no 'load_environment' function")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("REWARD FUNCTION UNIT TESTS")
    print(f"Module: {env_module_path}")
    print(f"{'='*60}\n")

    # ── Preset mode ────────────────────────────────────────────
    if preset:
        presets = {"gsm8k": GSM8K_PRESET}
        if preset not in presets:
            print(f"ERROR: Unknown preset '{preset}'. Available: {list(presets.keys())}")
            sys.exit(1)

        try:
            env = mod.load_environment(num_examples=1)
        except TypeError:
            env = mod.load_environment()
        except Exception as e:
            print(f"ERROR: load_environment() raised: {e}")
            sys.exit(1)

        reward_funcs = extract_reward_funcs(env.rubric)
        if not reward_funcs:
            print("ERROR: No reward functions found in rubric.")
            sys.exit(1)
        await run_preset_tests(reward_funcs, presets[preset], verbose)
        return

    # ── Custom mode ────────────────────────────────────────────
    if custom_path:
        custom_file = Path(custom_path)
        if not custom_file.exists():
            print(f"ERROR: Custom test file not found: {custom_path}")
            sys.exit(1)
        custom_cases = json.loads(custom_file.read_text())

        try:
            env = mod.load_environment(num_examples=1)
        except TypeError:
            env = mod.load_environment()
        except Exception as e:
            print(f"ERROR: load_environment() raised: {e}")
            sys.exit(1)

        # Custom cases use exact-match like presets
        reward_funcs = extract_reward_funcs(env.rubric)
        if not reward_funcs:
            print("ERROR: No reward functions found in rubric.")
            sys.exit(1)
        await run_preset_tests(reward_funcs, custom_cases, verbose)
        return

    # ── Auto mode (default) ────────────────────────────────────
    try:
        env = mod.load_environment(num_examples=5)
    except TypeError:
        try:
            env = mod.load_environment()
        except Exception as e:
            print(f"ERROR: load_environment() raised: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: load_environment() raised: {e}")
        sys.exit(1)

    # Access the dataset from the environment
    dataset = getattr(env, "dataset", None)
    if dataset is None:
        print("ERROR: Could not access env.dataset for auto test generation.")
        print("  Options:")
        print("    --preset gsm8k     Use hardcoded GSM8K test cases")
        print("    --custom FILE      Provide your own test cases as JSON")
        print("  JSON format: [{\"name\": \"...\", \"completion\": [...], \"answer\": \"...\", \"expected_reward\": 1.0}]")
        sys.exit(1)

    # Classify answer format from dataset samples
    answer_col = "answer"
    available_cols = dataset.column_names if hasattr(dataset, "column_names") else []
    for col in ["answer", "solution", "target", "output", "label"]:
        if col in available_cols:
            answer_col = col
            break

    sample_answers = [str(dataset[i][answer_col]) for i in range(min(10, len(dataset)))]
    answer_fmt = classify_answer_format(sample_answers)

    # Detect output format tags
    tags = detect_output_tags(env, path)

    if tags:
        print(f"Detected output tags: {tags}")
    else:
        print("No XML output tags detected \u2014 using plain text format")

    # Generate test cases
    test_cases = generate_test_cases(dataset, answer_fmt, tags)

    if not test_cases:
        print("ERROR: Could not generate test cases from dataset.")
        print("  Dataset may be empty. Check: python scripts/inspect_dataset.py <dataset>")
        sys.exit(1)

    reward_funcs = extract_reward_funcs(env.rubric)
    if not reward_funcs:
        print("ERROR: No reward functions found in rubric.")
        print("  Check that your Rubric has funcs=[...] with at least one function.")
        sys.exit(1)

    await run_auto_tests(reward_funcs, test_cases, answer_fmt, verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("env_module", help="Path to environment .py file")
    parser.add_argument("--preset", "-p", help="Use a preset test suite (e.g. gsm8k)")
    parser.add_argument("--custom", "-c", help="Path to custom test cases JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show individual test case details")
    args = parser.parse_args()
    asyncio.run(run(args.env_module, preset=args.preset, custom_path=args.custom, verbose=args.verbose))
