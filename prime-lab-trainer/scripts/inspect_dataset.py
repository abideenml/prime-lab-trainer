#!/usr/bin/env python3
"""
Step 1: Dataset Inspection
Run before writing any environment code.
Usage: python scripts/inspect_dataset.py <dataset_name> [config] [split]
Examples:
  python scripts/inspect_dataset.py gsm8k
  python scripts/inspect_dataset.py openai/gsm8k main train
  python scripts/inspect_dataset.py openlifescienceai/pubmedqa default train
"""

import re
import sys
from typing import Any


def truncate(val: Any, max_len: int = 200) -> str:
    s = str(val)
    return s[:max_len] + "..." if len(s) > max_len else s


# ─── Nested dict exploration ─────────────────────────────────────────────────

PROMPT_KEYS = ["prompt", "question", "problem", "input", "text", "sentence"]
ANSWER_KEYS = [
    "answer", "correct answer", "correct_answer", "solution",
    "target", "output", "label", "classification", "correct option",
    "correct_option",
]


def _find_in_dict_keys(dict_keys: list[str], candidates: list[str]) -> str | None:
    """Case-insensitive match of dict keys against candidate names."""
    lower_map = {k.lower().replace(" ", "_"): k for k in dict_keys}
    for cand in candidates:
        norm = cand.lower().replace(" ", "_")
        if norm in lower_map:
            return lower_map[norm]
    return None


def _detect_dict_columns(ds) -> list[tuple[str, list[str]]]:
    """Return [(col_name, [key1, key2, ...])] for columns that are dicts."""
    dict_cols = []
    for col, feat in ds.features.items():
        if type(feat).__name__ in ("dict", "Value"):
            # Check actual data
            sample = ds[0][col]
            if isinstance(sample, dict):
                dict_cols.append((col, list(sample.keys())))
    # Also check by feature type name
    for col, feat in ds.features.items():
        if col not in [d[0] for d in dict_cols]:
            type_name = type(feat).__name__
            if type_name in ("Sequence",) and hasattr(feat, "feature"):
                continue
            # Try loading a sample to check
            try:
                sample = ds[0][col]
                if isinstance(sample, dict):
                    dict_cols.append((col, list(sample.keys())))
            except Exception:
                pass
    return dict_cols


def inspect(dataset_name: str, config: str = None, split: str = "train"):
    try:
        from datasets import load_dataset, get_dataset_config_names
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"DATASET INSPECTION: {dataset_name}")
    print(f"{'='*60}\n")

    # Try to list configs
    try:
        configs = get_dataset_config_names(dataset_name)
        if configs:
            print(f"Available configs: {configs}")
            if config is None:
                config = configs[0]
                print(f"Auto-selected config: '{config}'\n")
    except Exception:
        pass

    # Load
    print(f"Loading split='{split}'...")
    try:
        if config:
            ds = load_dataset(dataset_name, config, split=split)
        else:
            ds = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        sys.exit(1)

    print(f"Loaded {len(ds)} examples\n")

    # Schema
    print("─" * 40)
    print("COLUMNS & TYPES")
    print("─" * 40)
    dict_cols = _detect_dict_columns(ds)
    dict_col_names = {d[0] for d in dict_cols}

    for col, feat in ds.features.items():
        type_label = type(feat).__name__
        print(f"  {col!r:30s} {type_label}")
        # If it's a dict column, show its internal keys
        if col in dict_col_names:
            keys = next(keys for c, keys in dict_cols if c == col)
            for key in keys:
                sample_val = ds[0][col].get(key, "")
                val_type = type(sample_val).__name__
                print(f"    .{key!r:26s} {val_type}")

    # ── Verifiers compatibility check ─────────────────────────────────────────
    print("\n" + "─" * 40)
    print("VERIFIERS COMPATIBILITY CHECK")
    print("─" * 40)
    cols = set(ds.column_names)

    # Search top-level columns first
    prompt_col = None
    for c in PROMPT_KEYS:
        if c in cols:
            prompt_col = c
            break

    answer_col = None
    for c in ANSWER_KEYS:
        if c in cols:
            answer_col = c
            break

    # If not found at top level, search inside dict columns
    nested_prompt = None  # (dict_col, key)
    nested_answer = None  # (dict_col, key)

    if not prompt_col or not answer_col:
        for dict_col_name, dict_keys in dict_cols:
            if not prompt_col and nested_prompt is None:
                found = _find_in_dict_keys(dict_keys, PROMPT_KEYS)
                if found:
                    nested_prompt = (dict_col_name, found)
            if not answer_col and nested_answer is None:
                found = _find_in_dict_keys(dict_keys, ANSWER_KEYS)
                if found:
                    nested_answer = (dict_col_name, found)

    if prompt_col:
        print(f"  ✓ Prompt column found: '{prompt_col}'")
        if prompt_col != "prompt":
            print(f"    → Rename with: dataset.rename_column('{prompt_col}', 'prompt')")
    elif nested_prompt:
        print(f"  ~ Prompt found NESTED: row['{nested_prompt[0]}']['{nested_prompt[1]}']")
        print(f"    → Extract with dataset.map() — see snippet below")
    else:
        print(f"  ✗ No prompt column found (checked: {', '.join(PROMPT_KEYS)})")
        print(f"    → You must rename or create a 'prompt' column")

    if answer_col:
        print(f"  ✓ Answer column found: '{answer_col}'")
        if answer_col != "answer":
            print(f"    → Rename with: dataset.rename_column('{answer_col}', 'answer')")
    elif nested_answer:
        print(f"  ~ Answer found NESTED: row['{nested_answer[0]}']['{nested_answer[1]}']")
        print(f"    → Extract with dataset.map() — see snippet below")
    else:
        print(f"  ✗ No answer column found (checked: {', '.join(ANSWER_KEYS[:6])})")

    # ── Sample rows ───────────────────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("SAMPLE ROWS (first 3)")
    print("─" * 40)
    for i in range(min(3, len(ds))):
        print(f"\n[Example {i}]")
        row = ds[i]
        for col in ds.column_names:
            val = row[col]
            if isinstance(val, dict):
                print(f"  {col}: {{")
                for k, v in val.items():
                    print(f"    {k!r}: {truncate(v, 120)}")
                print(f"  }}")
            elif isinstance(val, list) and val and isinstance(val[0], dict):
                print(f"  {col}: [{truncate(val, 150)}]")
            else:
                print(f"  {col}: {truncate(val)}")

    # ── Answer format analysis ────────────────────────────────────────────────
    # Determine which samples to analyze for answer format
    effective_answer_col = answer_col
    effective_answer_getter = None

    if answer_col:
        effective_answer_getter = lambda row: str(row[answer_col])
    elif nested_answer:
        dcol, dkey = nested_answer
        effective_answer_getter = lambda row: str(row[dcol][dkey])
        effective_answer_col = f"{dcol}.{dkey}"

    if effective_answer_getter:
        print("\n" + "─" * 40)
        print("ANSWER FORMAT ANALYSIS")
        print("─" * 40)
        if nested_answer and not answer_col:
            print(f"  (Analyzing nested field: row['{nested_answer[0]}']['{nested_answer[1]}'])\n")
        samples = [effective_answer_getter(ds[i]) for i in range(min(20, len(ds)))]
        stripped = [s.strip() for s in samples]

        # 1. GSM8K #### marker
        has_gsm_marker = sum(1 for s in samples if "####" in s)
        if has_gsm_marker > len(samples) * 0.5:
            print(f"  Type: GSM8K numeric  ({has_gsm_marker}/{len(samples)} have '####' marker)")
            print(f"    → Parse with: re.search(r'####\\s*([\\d,\\.\\-]+)', answer)")
            print("    Sample extractions:")
            for s in samples[:3]:
                m = re.search(r"####\s*([\d,\.\-]+)", s)
                print(f"      {truncate(s, 80)} → {m.group(1) if m else 'NO MATCH'}")
            print(f"    → Reward: exact numeric match after stripping commas")

        # 2. Boolean (yes/no, true/false)
        elif all(s.lower() in {"yes", "no", "true", "false", "maybe", "0", "1"} for s in stripped if s):
            unique = sorted(set(s.lower() for s in stripped))
            print(f"  Type: Boolean / categorical")
            print(f"    Values: {unique}")
            print(f"    → Reward: answer.strip().lower() == model_answer.strip().lower()")

        # 3. Multi-choice (single letter A-E or digit 1-5)
        elif all(re.match(r"^[A-Ea-e1-5]$", s) for s in stripped if s):
            unique = sorted(set(s.upper() for s in stripped))
            print(f"  Type: Multiple choice")
            print(f"    Choices: {unique}")
            print(f"    → Reward: answer.strip().upper() == model_choice.strip().upper()")

        # 4. Plain numeric
        elif sum(1 for s in stripped if re.match(r"^[\d,\.\-]+$", s)) > len(samples) * 0.5:
            numeric = sum(1 for s in stripped if re.match(r"^[\d,\.\-]+$", s))
            print(f"  Type: Plain numeric  ({numeric}/{len(samples)} samples)")
            print(f"    → Reward: answer.strip() == model_answer.strip()")

        # 5. Short label (< 30 chars, likely classification)
        elif all(len(s) < 30 for s in stripped):
            unique = sorted(set(stripped))
            print(f"  Type: Short label / classification")
            if len(unique) <= 10:
                print(f"    Labels ({len(unique)}): {unique}")
            else:
                print(f"    {len(unique)} unique labels (first 5): {unique[:5]}")
            print(f"    → Reward: answer.strip().lower() == model_answer.strip().lower()")

        # 6. Long text (fallback)
        else:
            avg_len = sum(len(s) for s in stripped) / max(len(stripped), 1)
            print(f"  Type: Free text  (avg length: {avg_len:.0f} chars)")
            print(f"    Sample answers:")
            for s in samples[:3]:
                print(f"      {truncate(s, 100)!r}")
            print(f"    → Reward: use JudgeRubric or semantic similarity")
            print(f"    → See references/environment_guide.md for free-text patterns")

    # ── Recommended snippet ───────────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("RECOMMENDED load_environment() SNIPPET")
    print("─" * 40)
    config_str = f'"{config}", ' if config else ""

    if prompt_col and answer_col:
        # Simple case: flat columns
        rename_lines = []
        if prompt_col != "prompt":
            rename_lines.append(f'    dataset = dataset.rename_column("{prompt_col}", "prompt")')
        if answer_col != "answer":
            rename_lines.append(f'    dataset = dataset.rename_column("{answer_col}", "answer")')
        rename_block = "\n".join(rename_lines) if rename_lines else "    # columns already named correctly"
        print(f"""
    dataset = load_dataset("{dataset_name}", {config_str}split=split)
{rename_block}

    # Convert prompts to ChatMessage format (required by SingleTurnEnv)
    dataset = dataset.map(lambda row: {{
        "prompt": [{{"role": "user", "content": row["prompt"]}}],
    }})
""")

    elif nested_prompt or nested_answer:
        # Nested case: need dataset.map() to flatten
        p_col, p_key = nested_prompt if nested_prompt else (None, None)
        a_col, a_key = nested_answer if nested_answer else (None, None)

        map_fields = []
        if p_col:
            map_fields.append(
                f'        "prompt": [{{"role": "user", "content": row["{p_col}"]["{p_key}"]}}],'
            )
        else:
            map_fields.append(
                '        "prompt": [{"role": "user", "content": row["<FIXME_PROMPT_FIELD>"]}],'
            )
        if a_col:
            map_fields.append(f'        "answer": row["{a_col}"]["{a_key}"],')
        else:
            map_fields.append('        "answer": row["<FIXME_ANSWER_FIELD>"],')

        map_block = "\n".join(map_fields)

        # Find other potentially useful nested fields
        extra_fields = []
        for dcol, dkeys in dict_cols:
            for dkey in dkeys:
                lower = dkey.lower().replace(" ", "_")
                if lower in ("context", "contexts", "passage", "passages", "long_answer", "explanation", "options"):
                    extra_fields.append((dcol, dkey))

        extra_comment = ""
        if extra_fields:
            extras = ", ".join(f'row["{dc}"]["{dk}"]' for dc, dk in extra_fields)
            extra_comment = f"\n    # Other useful fields: {extras}"

        print(f"""
    dataset = load_dataset("{dataset_name}", {config_str}split=split)
    dataset = dataset.map(lambda row: {{
{map_block}
    }}){extra_comment}
""")

    else:
        # No prompt or answer found anywhere
        print(f"""
    # ⚠ Could not auto-detect prompt or answer fields.
    # Review the sample rows above and manually map columns:
    dataset = load_dataset("{dataset_name}", {config_str}split=split)
    dataset = dataset.map(lambda row: {{
        "prompt": [{{"role": "user", "content": row["<FIXME>"]}}],
        "answer": row["<FIXME>"],
    }})
""")

    print("=" * 60)
    print("Inspection complete. Read output before writing reward function.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    dataset_name = sys.argv[1]
    config = sys.argv[2] if len(sys.argv) > 2 else None
    split = sys.argv[3] if len(sys.argv) > 3 else "train"
    inspect(dataset_name, config, split)
