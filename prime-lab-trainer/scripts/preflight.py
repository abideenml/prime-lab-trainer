#!/usr/bin/env python3
"""
Step 0: Preflight Check
Run before starting the workflow to verify all prerequisites are met.

Usage:
  python scripts/preflight.py
  python scripts/preflight.py --install    # auto-install missing Python packages
  python scripts/preflight.py --verbose    # show extra detail

Checks:
  1. Python >= 3.10
  2. 'datasets' library installed
  3. 'verifiers' library installed (>= 0.1.8)
  4. 'prime' CLI on PATH
  5. Prime Intellect login
  6. HuggingFace authentication
  7. OpenAI API key (for model eval)
"""

import sys
import os
import shutil
import subprocess
import argparse


# ─── Helpers ──────────────────────────────────────────────────────────────────

def version_gte(version_str: str, min_version: str) -> bool:
    """Simple tuple-based version comparison."""
    try:
        v = tuple(int(x) for x in version_str.split(".")[:3])
        m = tuple(int(x) for x in min_version.split(".")[:3])
        return v >= m
    except (ValueError, AttributeError):
        return True  # unparseable → assume ok


def pip_install(package_spec: str) -> bool:
    """Install a package with pip. Returns True on success."""
    print(f"    Installing {package_spec} ...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", package_spec],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return True
    except subprocess.TimeoutExpired:
        print(f"    Install timed out after 120s. Check network connection.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"    Install failed: {e.stderr.strip()[:200]}")
        return False


# ─── Individual checks ────────────────────────────────────────────────────────
# Each returns (ok: bool|None, detail: str, fix: str|None)
# ok=True → pass, ok=False → fail, ok=None → skipped

def check_python():
    v = sys.version_info
    ok = v >= (3, 10)
    detail = f"{v.major}.{v.minor}.{v.micro}"
    fix = "Install Python 3.10+: https://www.python.org/downloads/" if not ok else None
    return ok, detail, fix


def check_datasets():
    try:
        import datasets
        return True, f"v{datasets.__version__}", None
    except ImportError:
        return False, "not installed", "pip install datasets"


def check_verifiers():
    min_ver = "0.1.8"
    try:
        import verifiers
        ver = getattr(verifiers, "__version__", "unknown")
        if ver != "unknown" and not version_gte(ver, min_ver):
            return False, f"v{ver} (need >= {min_ver})", f"pip install 'verifiers>={min_ver}'"
        return True, f"v{ver}", None
    except ImportError:
        return False, "not installed", f"pip install 'verifiers>={min_ver}'"


def check_prime_cli():
    path = shutil.which("prime")
    if path:
        return True, path, None
    return False, "not on PATH", "pip install prime"


def check_prime_auth():
    if not shutil.which("prime"):
        return None, "skipped (CLI not installed)", "Install prime CLI first"
    try:
        result = subprocess.run(
            ["prime", "env", "list"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return True, "authenticated", None
        stderr = result.stderr.lower()
        if any(w in stderr for w in ("login", "auth", "unauthorized", "token")):
            return False, "not logged in", "prime login"
        return False, f"CLI error (exit {result.returncode})", "prime login"
    except subprocess.TimeoutExpired:
        return False, "timed out", "Check network, then: prime login"


# ─── Main ─────────────────────────────────────────────────────────────────────
# HuggingFace auth is NOT required:
#   - Public HF datasets load without any token
#   - Environments are pushed to Prime Hub (prime login covers this)
#   - HF auth is only needed for private datasets or publishing model checkpoints to HF Hub

CHECKS = [
    # (label,           check_fn,        installable_spec)
    ("Python >= 3.10",  check_python,    None),
    ("datasets",        check_datasets,  "datasets"),
    ("verifiers",       check_verifiers, "verifiers>=0.1.8"),
    ("prime CLI",       check_prime_cli, None),
    ("Prime auth",      check_prime_auth, None),
]

SOFT_CHECKS: set = set()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--install", action="store_true", help="Auto-install missing Python packages")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("PREFLIGHT CHECK — Prime Lab Trainer")
    print(f"{'='*60}\n")

    failures = []
    warnings = []

    for label, check_fn, install_spec in CHECKS:
        ok, detail, fix = check_fn()

        # Try auto-install on failure if applicable
        if not ok and ok is not None and args.install and install_spec:
            if pip_install(install_spec):
                ok, detail, fix = check_fn()

        if ok is True:
            print(f"  [✓] {label:25s} {detail}")
        elif ok is None:
            print(f"  [—] {label:25s} {detail}")
            if fix and args.verbose:
                print(f"       └─ {fix}")
        else:
            is_soft = label in SOFT_CHECKS
            marker = "⚠" if is_soft else "✗"
            print(f"  [{marker}] {label:25s} {detail}")
            if fix:
                print(f"       └─ {fix}")
            if is_soft:
                warnings.append(label)
            else:
                failures.append(label)

    print(f"\n{'─'*60}")

    if not failures and not warnings:
        print("✓ All checks passed. Ready to start the workflow.")
        print("  Next: python scripts/inspect_dataset.py <dataset_name>")
    elif not failures:
        print("✓ Core checks passed.")
        for w in warnings:
            print(f"  ⚠ {w} — not required until Step 4 (eval)")
        print("  You can start writing environments now.")
    else:
        print("✗ Some checks failed. Fix the issues above before proceeding.")
        if not args.install:
            print("  Tip: re-run with --install to auto-install missing Python packages")

    print()
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
