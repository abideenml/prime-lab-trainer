#!/usr/bin/env python3
"""
Step 5: Reward Distribution Check
Run after 'prime env eval' completes, before pushing to Hub.

Usage:
  python scripts/check_reward_distribution.py --results path/to/results.json
  python scripts/check_reward_distribution.py --results path/to/results.jsonl --last-n 100
  python scripts/check_reward_distribution.py --verbose

⚠️  IMPORTANT — PROVIDE --results EXPLICITLY:
  The auto-discovery of results files (searching .prime/, eval_results/, etc.)
  is a best-effort guess. The exact output path of 'prime env eval' is not
  documented and may differ from what this script searches.

  The safest usage is to pass the file path directly:
    python scripts/check_reward_distribution.py --results outputs/evals/my-env--gpt-4.1-mini/abc123/results.jsonl

  Similarly, the field names this script looks for ('reward', 'reward_breakdown',
  etc.) are guessed from common conventions — not confirmed from Prime docs.
  If extraction fails, the script will print the actual field names from your
  results file so you can inspect and adjust.

What it checks:
  ┌─────────────────────────────────────────────────────────────┐
  │ Check           │ Pass Condition │ Failure Means             │
  ├─────────────────┼────────────────┼───────────────────────────┤
  │ Not all zero    │ mean > 0.05    │ Reward function broken    │
  │ Not all ones    │ mean < 0.95    │ Trivially easy / hacked   │
  │ Variance        │ std > 0.05     │ Signal too weak           │
  │ Format reward   │ mean > 0.30    │ Model can't follow format │
  │ Sample count    │ n >= 20        │ Too few to trust          │
  └─────────────────────────────────────────────────────────────┘

Exit code 0 = all checks pass (safe to push to hub)
Exit code 1 = one or more checks failed (fix before submitting training)
"""

import sys
import json
import math
import argparse
from pathlib import Path
from typing import Optional


# ─── Prime eval stores results locally ────────────────────────────────────────
# 'prime eval run' saves results to: environments/<env>/outputs/evals/<env>--<model>/<hash>/
# The static dirs below are fallbacks for other tooling conventions.
PRIME_RESULTS_DIRS = [
    Path("outputs/evals"),       # legacy / direct verifiers runner
    Path(".prime/eval_results"),
    Path("eval_results"),
    Path(".prime/results"),
]


def find_latest_results() -> Optional[Path]:
    """Find the most recently written eval results file.

    Prefers results.jsonl (per-rollout data) over metadata.json (aggregate summary).
    metadata.json is written after results.jsonl so it would otherwise always win on mtime.
    """
    jsonl_candidates = []
    json_candidates = []

    def _collect(directory: Path):
        for f in directory.glob("**/*.jsonl"):
            jsonl_candidates.append(f)
        for f in directory.glob("**/*.json"):
            if f.name != "metadata.json":   # exclude aggregate summary files
                json_candidates.append(f)

    # Primary: environments/*/outputs/evals/ — where prime eval run saves by default
    envs_dir = Path("environments")
    if envs_dir.exists():
        for eval_dir in envs_dir.glob("*/outputs/evals"):
            if eval_dir.is_dir():
                _collect(eval_dir)

    # Fallback: legacy static dirs
    for d in PRIME_RESULTS_DIRS:
        if d.exists():
            _collect(d)

    # Prefer results.jsonl; only fall back to .json if no .jsonl found
    candidates = jsonl_candidates or json_candidates
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_results(path: Path) -> list[dict]:
    """Load eval results from JSON or JSONL."""
    rows = []
    text = path.read_text()
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    else:
        data = json.loads(text)
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict) and "results" in data:
            rows = data["results"]
        else:
            rows = [data]
    return rows


def extract_rewards(rows: list[dict], last_n: int = 0) -> tuple[list[float], list[float]]:
    """
    Extract (correctness_rewards, format_rewards) from result rows.

    Handles two file formats produced by prime eval run:
      results.jsonl  — one row per rollout, fields: reward, metrics (correct_label, format_reward_func)
      metadata.json  — one aggregate row, fields: avg_reward, avg_metrics
    """
    correctness = []
    fmt = []

    if last_n > 0:
        rows = rows[-last_n:]

    for row in rows:
        # ── Format 1: results.jsonl (per-rollout) ──────────────────
        # Fields: reward (float), metrics (dict with correct_label, format_reward_func)
        r = row.get("reward")
        if r is None:
            r = row.get("rewards")
        if r is None:
            r = row.get("total_reward")
        if r is None and "reward_breakdown" in row:
            breakdown = row["reward_breakdown"]
            if isinstance(breakdown, list) and len(breakdown) > 0:
                r = breakdown[0]
            elif isinstance(breakdown, dict):
                r = breakdown.get("correctness") or breakdown.get("reward_0")

        # metrics dict in results.jsonl: {"correct_label": 0.8, "format_reward_func": 0.6, ...}
        metrics = row.get("metrics") or {}
        if r is None:
            r = metrics.get("correct_label")

        if r is not None:
            correctness.append(float(r))

        # Format reward from metrics dict (results.jsonl)
        fr = metrics.get("format_reward_func") or metrics.get("format_reward")
        if fr is None and "reward_breakdown" in row:
            breakdown = row["reward_breakdown"]
            if isinstance(breakdown, list) and len(breakdown) > 1:
                fr = breakdown[1]
            elif isinstance(breakdown, dict):
                fr = breakdown.get("format") or breakdown.get("reward_1")

        # ── Format 2: metadata.json (aggregate summary) ────────────
        # Fields: avg_reward (float), avg_metrics (dict)
        if r is None:
            r = row.get("avg_reward")
            if r is not None:
                correctness.append(float(r))

        if fr is None:
            avg_metrics = row.get("avg_metrics") or {}
            fr = avg_metrics.get("correct_label")
            if fr is None:
                fr = avg_metrics.get("format_reward_func")

        if fr is not None:
            fmt.append(float(fr))

    return correctness, fmt


def stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    n = len(vals)
    mean = sum(vals) / n
    variance = sum((v - mean) ** 2 for v in vals) / n
    std = math.sqrt(variance)
    return {"n": n, "mean": mean, "std": std, "min": min(vals), "max": max(vals)}


def histogram(vals: list[float], bins: int = 10, width: int = 40) -> str:
    if not vals:
        return "  (no data)"
    lo, hi = min(vals), max(vals)
    if lo == hi:
        return f"  All values = {lo:.3f} (no variance)"
    bin_size = (hi - lo) / bins
    counts = [0] * bins
    for v in vals:
        idx = min(int((v - lo) / bin_size), bins - 1)
        counts[idx] += 1
    max_count = max(counts)
    lines = []
    for i, count in enumerate(counts):
        lo_b = lo + i * bin_size
        hi_b = lo_b + bin_size
        bar = "█" * int(count / max_count * width) if max_count > 0 else ""
        lines.append(f"  [{lo_b:.2f}-{hi_b:.2f}] {bar} {count}")
    return "\n".join(lines)


def run_checks(
    correctness: list[float],
    fmt: list[float],
    verbose: bool = False,
) -> bool:
    checks_passed = True

    print(f"\n{'='*60}")
    print("REWARD DISTRIBUTION CHECK")
    print(f"{'='*60}\n")

    # ── Correctness reward ──────────────────────────────────────
    cs = stats(correctness)
    print(f"Correctness Reward  (n={cs['n']})")
    print(f"  mean={cs['mean']:.3f}  std={cs['std']:.3f}  min={cs['min']:.3f}  max={cs['max']:.3f}")

    if verbose:
        print(histogram(correctness))

    print()

    checks = [
        (
            "Sample count",
            cs["n"] >= 20,
            f"Only {cs['n']} samples — too few to trust distribution. Run eval with -n 50+",
            "green",
        ),
        (
            "Not all zero  (mean > 0.05)",
            cs["mean"] is not None and cs["mean"] > 0.05,
            "Mean reward ≤ 0.05 — reward function likely broken.\n"
            "  → Check column mapping, regex parsing, and parser.parse_answer()",
            "red",
        ),
        (
            "Not all ones  (mean < 0.95)",
            cs["mean"] is not None and cs["mean"] < 0.95,
            "Mean reward ≥ 0.95 — reward is trivially easy or reward-hacked.\n"
            "  → Check for substring match on short answers, wrong answer column",
            "red",
        ),
        (
            "Variance      (std > 0.05)",
            cs["std"] is not None and cs["std"] > 0.05,
            "Std ≤ 0.05 — reward signal has near-zero variance, model cannot learn.\n"
            "  → Dataset too easy/hard for this model, or reward too coarse",
            "yellow",
        ),
    ]

    for name, passed, failure_msg, _ in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            checks_passed = False
            for line in failure_msg.split("\n"):
                print(f"           {line}")
        print()

    # ── Format reward ───────────────────────────────────────────
    if fmt:
        fs = stats(fmt)
        print(f"Format Reward  (n={fs['n']})")
        print(f"  mean={fs['mean']:.3f}  std={fs['std']:.3f}")
        if verbose:
            print(histogram(fmt))
        print()

        format_check = (
            "Format compliance (mean > 0.30)",
            fs["mean"] is not None and fs["mean"] > 0.30,
            "Format reward mean ≤ 0.30 — model rarely follows output format.\n"
            "  → Improve system prompt, consider SFT warmup before RL",
            "yellow",
        )
        name, passed, failure_msg, _ = format_check
        status = "✓ PASS" if passed else "⚠ WARN"
        print(f"  [{status}] {name}")
        if not passed:
            for line in failure_msg.split("\n"):
                print(f"           {line}")
            # Format is a warning, not a hard failure
        print()
    else:
        print("  (no format reward data found — skipping format check)\n")

    # ── Summary ─────────────────────────────────────────────────
    print("─" * 60)
    if checks_passed:
        print("✓ All checks passed. Safe to push environment to Hub.")
        print("  Next: prime env push my-env")
    else:
        print("✗ Checks failed. Fix reward function before submitting training.")
        print("  Review failures above, update environment, re-run:")
        print("    python scripts/test_reward.py environments/my_env/my_env.py")
        print("    prime eval run my-env -m openai/gpt-5-nano -n 50")
        print("    python scripts/check_reward_distribution.py")

    print()
    return checks_passed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", "-r", help="Path to eval results JSON/JSONL file")
    parser.add_argument("--last-n", "-n", type=int, default=0, help="Only use last N results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show histogram")
    args = parser.parse_args()

    # Find results
    if args.results:
        results_path = Path(args.results)
        if not results_path.exists():
            print(f"ERROR: Results file not found: {args.results}")
            sys.exit(1)
    else:
        results_path = find_latest_results()
        if results_path is None:
            print("ERROR: No eval results found via auto-discovery.")
            print("  Searched: environments/*/outputs/evals/, " + ", ".join(str(d) for d in PRIME_RESULTS_DIRS))
            print("  Provide the path explicitly:")
            print("    python scripts/check_reward_distribution.py --results <path/to/results.jsonl>")
            print("  Run eval first:")
            print("    prime eval run <env-name> -m openai/gpt-5-nano -n 50")
            sys.exit(1)
        print(f"Auto-discovered results: {results_path}")
        print("  (WARNING: auto-discovery path is a guess — use --results <path> to be explicit)\n")

    rows = load_results(results_path)
    print(f"Loaded {len(rows)} result rows")

    correctness, fmt = extract_rewards(rows, last_n=args.last_n)

    if not correctness:
        print("\nERROR: Could not extract reward values from results file.")
        print("Expected fields: 'reward', 'rewards', 'total_reward', or 'reward_breakdown'")
        print(f"\nSample row keys: {list(rows[0].keys()) if rows else 'empty'}")
        if rows:
            print(f"Sample row: {json.dumps(rows[0], indent=2)[:500]}")
        sys.exit(1)

    passed = run_checks(correctness, fmt, verbose=args.verbose)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
