"""Run candidate programs against a test suite in an isolated subprocess.

Buggy and model-generated code can loop forever, exhaust memory, or exit the
interpreter, so every execution happens in a separate process with a wall-clock
timeout. A task counts as fixed only if the process exits 0, which requires the
whole EvalPlus suite to pass.
"""

import os
import subprocess
import sys
import tempfile

# EvalPlus suites are far larger than the original HumanEval ones; a correct but
# unoptimised solution can legitimately need well over 30s on the plus inputs.
DEFAULT_TIMEOUT = 90.0

# EvalPlus suites call check(<entry_point>) themselves in some releases and only
# define it in others, so we invoke it explicitly under a guard.
_HARNESS = """
if 'check' in dir():
    check({entry_point})
"""


def run(program_src, test_src, entry_point, timeout=DEFAULT_TIMEOUT):
    """Return (passed: bool, detail: str)."""
    # Several HumanEval problems embed non-ASCII literals, so the encoding must be
    # pinned explicitly -- relying on the locale default makes those tasks die with
    # a spurious "Non-UTF-8 code" SyntaxError that looks like a model failure.
    script = (
        "# -*- coding: utf-8 -*-\n"
        f"{program_src}\n\n{test_src}\n{_HARNESS.format(entry_point=entry_point)}"
    )

    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(script)
        path = fh.name

    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
            # Prevent the child from inheriting anything that could let it
            # interfere with the parent run.
            env={"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")},
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"harness-error: {exc}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    if proc.returncode == 0:
        return True, "pass"
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return False, tail[-1][:200] if tail else f"exit {proc.returncode}"
