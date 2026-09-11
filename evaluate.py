"""Execute generated fixes against the EvalPlus suites and print Table 1.

Repair accuracy = fraction of tasks whose generated program passes the *entire*
EvalPlus suite, matching the paper's definition (App. B.2: a fix counts only if
it passes every test). Output mirrors the layout of the paper's Table 1, with a
delta column against the vanilla row so the claim under test is readable directly.
"""

import argparse
import json
import pathlib
from concurrent.futures import ThreadPoolExecutor

import config
import prompts
import providers
import sandbox

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data" / "hefixplus.jsonl"
GENERATIONS = HERE / "results" / "generations.jsonl"
SCORES = HERE / "results" / "scores.jsonl"

ROW_LABELS = {
    "vanilla_trace": "Vanilla w/ trace",
    "cot_trace": "+ CoT",
    "cot_notrace": "+ CoT; - trace",
}

# The paper's Table 1, HumanEvalFix-Plus block, for side-by-side comparison.
PAPER_HEFIX = {
    "vanilla_trace": {"PaLM 2-L": 59.1, "GPT-3.5": 70.1, "GPT-4": 88.4,
                      "Mixtral 8x7B": 32.9, "DeepSeek Coder 33B": 57.3, "StarCoder 15.5B": 29.3},
    "cot_trace": {"PaLM 2-L": 48.8, "GPT-3.5": 75.6, "GPT-4": 84.8,
                  "Mixtral 8x7B": 34.1, "DeepSeek Coder 33B": 30.5, "StarCoder 15.5B": 16.5},
    "cot_notrace": {"PaLM 2-L": 43.3, "GPT-3.5": 72.0, "GPT-4": 82.9,
                    "Mixtral 8x7B": 25.6, "DeepSeek Coder 33B": 22.6, "StarCoder 15.5B": 18.3},
}


def load_tasks():
    return {json.loads(l)["task_id"]: json.loads(l) for l in DATA.open()}


def import_preamble(task):
    """The import lines the buggy program was shown with.

    Models often reply with just the repaired function, dropping the
    `from typing import List` that sat above it in the prompt. Executing that
    bare function raises NameError on the annotation and scores a correct repair
    as a failure. HumanEval harnesses conventionally evaluate a completion
    against its prompt, so restoring the preamble is the faithful behaviour, not
    a favour to the model. Re-importing is harmless when the model kept it.
    """
    prelude = task.get("import") or ""
    head = []
    for line in (task.get("declaration") or "").splitlines():
        if line.lstrip().startswith("def "):
            break
        head.append(line)
    return (prelude + "\n".join(head)).strip()


def load_generations():
    if not GENERATIONS.exists():
        raise SystemExit(
            f"No generations found at {GENERATIONS}\n\n"
            "evaluate.py scores model output, so something has to produce it first:\n"
            "  pip install -r requirements.txt\n"
            "  export ANTHROPIC_API_KEY=...   export OPENROUTER_API_KEY=...\n"
            "  python run.py --grid a --limit 5     # cheap smoke test\n"
            "  python evaluate.py"
        )
    out = []
    for line in GENERATIONS.open():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not rec.get("error"):
            out.append(rec)
    # A resumed run can append a duplicate cell; last write wins.
    dedup = {}
    for rec in out:
        dedup[(rec["model"], rec["condition"], rec["thinking"], rec["task_id"])] = rec
    return list(dedup.values())


def score_all(generations, tasks, workers):
    def one(rec):
        task = tasks[rec["task_id"]]
        # Re-extract from the raw response rather than trusting the stored `code`,
        # so a fix to the extractor repairs past runs without re-spending on API
        # calls. Falls back to the stored value if the response was not kept.
        code = providers.extract_code(rec.get("response", ""), task["entry_point"]) \
            or rec.get("code", "")
        if not code.strip():
            passed, detail = False, "empty-generation"
        else:
            preamble = import_preamble(task)
            program = f"{preamble}\n{code}" if preamble else code
            passed, detail = sandbox.run(program, task["eval_test"], task["entry_point"])
        return {
            "model": rec["model"],
            "condition": rec["condition"],
            "thinking": rec["thinking"],
            "task_id": rec["task_id"],
            "passed": passed,
            "detail": detail,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, generations))


def report(scores):
    registry = config.registry()
    models = [m for m in registry if any(s["model"] == m for s in scores)]
    thinking_modes = sorted({s["thinking"] for s in scores})

    for thinking in thinking_modes:
        subset = [s for s in scores if s["thinking"] == thinking]
        if not subset:
            continue
        label = "reasoning ON (Grid B)" if thinking else "reasoning OFF (Grid A - replicates Table 1)"
        print(f"\n{'=' * 78}\nHumanEvalFix-Plus repair accuracy (%) - {label}\n{'=' * 78}")

        header = f"{'Prompting method':<20}" + "".join(f"{m:>16}" for m in models) + f"{'Avg.':>9}"
        print(header)
        print("-" * len(header))

        # The paper reports each row's delta against the row directly above it,
        # not against the vanilla baseline (Table 1: 48.4 - 44.1 = 4.3).
        previous_avg = None
        for condition in prompts.CONDITIONS:
            cells = []
            for model in models:
                rows = [s for s in subset if s["model"] == model and s["condition"] == condition]
                cells.append(100.0 * sum(r["passed"] for r in rows) / len(rows) if rows else None)

            present = [c for c in cells if c is not None]
            if not present:
                continue
            avg = sum(present) / len(present)
            delta = f"  {avg - previous_avg:+.1f}" if previous_avg is not None else ""
            previous_avg = avg

            line = f"{ROW_LABELS[condition]:<20}"
            line += "".join(f"{c:>16.1f}" if c is not None else f"{'-':>16}" for c in cells)
            line += f"{avg:>9.1f}{delta}"
            print(line)

        counts = {
            (s["model"], s["condition"]) for s in subset
        }
        n = len({s["task_id"] for s in subset})
        print(f"\n  n = {n} tasks per cell; {len(counts)} model x condition cells scored")

    # Side-by-side with the paper's own numbers.
    print(f"\n{'=' * 78}\nPaper's Table 1 (HumanEvalFix-Plus, April 2024 models)\n{'=' * 78}")
    paper_models = list(PAPER_HEFIX["vanilla_trace"])
    header = f"{'Prompting method':<20}" + "".join(f"{m:>20}" for m in paper_models) + f"{'Avg.':>9}"
    print(header)
    print("-" * len(header))
    previous = None
    for condition in prompts.CONDITIONS:
        vals = [PAPER_HEFIX[condition][m] for m in paper_models]
        avg = sum(vals) / len(vals)
        delta = f"  {avg - previous:+.1f}" if previous is not None else ""
        previous = avg
        print(f"{ROW_LABELS[condition]:<20}" + "".join(f"{v:>20.1f}" for v in vals)
              + f"{avg:>9.1f}{delta}")

    print("\nThe paper's claim: '+ CoT' is WORSE than 'Vanilla w/ trace' on HeFix+ "
          "(-7.8 avg).\nIf your '+ CoT' delta is >= 0, the premise no longer holds "
          "for current models.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reuse", action="store_true",
                    help="reuse results/scores.jsonl instead of re-executing code")
    args = ap.parse_args()

    if args.reuse and SCORES.exists():
        scores = [json.loads(l) for l in SCORES.open()]
        print(f"reusing {len(scores)} cached scores")
    else:
        tasks = load_tasks()
        generations = load_generations()
        if not generations:
            raise SystemExit("no successful generations found; run `python run.py` first")
        print(f"executing {len(generations)} generated programs ...")
        scores = score_all(generations, tasks, args.workers)
        SCORES.parent.mkdir(parents=True, exist_ok=True)
        with SCORES.open("w") as fh:
            for s in scores:
                fh.write(json.dumps(s) + "\n")
        print(f"wrote {SCORES}")

    report(scores)


if __name__ == "__main__":
    main()
