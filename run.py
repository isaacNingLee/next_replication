"""Generate repairs for every (model, condition, task) cell.

Writes one JSON object per generation to results/generations.jsonl and skips
cells already present, so the run is resumable after a crash, a rate limit, or a
deliberate Ctrl-C. Nothing here executes model output -- that happens in
evaluate.py, in a subprocess.

  python run.py --grid a --limit 5          # cheap smoke test
  python run.py --grid a                    # the Table 1 replication
  python run.py --grid b                    # reasoning-on ablation
"""

import argparse
import itertools
import json
import os
import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import config
import prompts
import providers

DATA = pathlib.Path(__file__).parent / "data" / "hefixplus.jsonl"
OUT = pathlib.Path(__file__).parent / "results" / "generations.jsonl"

_write_lock = threading.Lock()


def load_tasks(limit=None):
    records = [json.loads(line) for line in DATA.open()]
    return records[:limit] if limit else records


def load_done():
    if not OUT.exists():
        return set()
    done = set()
    for line in OUT.open():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not rec.get("error"):
            done.add((rec["model"], rec["condition"], rec["thinking"], rec["task_id"]))
    return done


def emit(record):
    with _write_lock:
        with OUT.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
            fh.flush()


def build_cells(models, conditions, thinking, tasks, done):
    cells = []
    for model, condition, task in itertools.product(models, conditions, tasks):
        key = (model["name"], condition, thinking, task["task_id"])
        if key not in done:
            cells.append((model, condition, task))
    return cells


def run_cell(model, condition, task, thinking, fewshot_cache, retries=3):
    key = (condition,)
    fewshot = fewshot_cache[key]
    user = prompts.build_task_prompt(task, condition)
    backend = providers.get_backend(model["backend"])

    last_error = None
    for attempt in range(retries):
        try:
            text, usage = backend.complete(
                model["model_id"], fewshot, user, thinking=thinking
            )
            # An empty completion is a failed call, not an answer. Providers
            # behind OpenRouter return one intermittently (~29% of qwen calls in
            # one run, all reproducing fine on retry). Scoring it as a wrong fix
            # silently depresses that model's accuracy, so retry, and if it never
            # yields text, record an error rather than a bogus zero-length fix.
            if not text.strip():
                raise RuntimeError("provider returned an empty completion")
            return {
                "model": model["name"],
                "condition": condition,
                "thinking": thinking,
                "task_id": task["task_id"],
                "response": text,
                "code": providers.extract_code(text, task["entry_point"]),
                "usage": usage,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - surfaced in the record
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2 ** attempt)

    return {
        "model": model["name"],
        "condition": condition,
        "thinking": thinking,
        "task_id": task["task_id"],
        "response": "",
        "code": "",
        "usage": {},
        "error": last_error,
    }


def preflight(models):
    """Fail fast on missing SDKs or keys.

    Backends construct their client lazily, so without this every one of the
    hundreds of cells would burn its full retry budget rediscovering the same
    missing import and writing an error record.
    """
    import importlib.util

    needs = {m["backend"] for m in models}
    problems, fixes = [], []

    if "anthropic" in needs:
        if importlib.util.find_spec("anthropic") is None:
            problems.append("the `anthropic` package is not installed")
            fixes.append("pip install -r requirements.txt")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            problems.append("ANTHROPIC_API_KEY is not set")
            fixes.append("export ANTHROPIC_API_KEY=...")
    if "openrouter" in needs:
        if importlib.util.find_spec("openai") is None:
            problems.append("the `openai` package is not installed")
            fixes.append("pip install -r requirements.txt")
        if not os.environ.get("OPENROUTER_API_KEY"):
            problems.append("OPENROUTER_API_KEY is not set")
            fixes.append("export OPENROUTER_API_KEY=...")

    if problems:
        detail = "\n".join(f"  - {p}" for p in problems)
        # Only suggest what is actually missing; listing a key the current roster
        # does not use sends people off to fund the wrong provider.
        steps = "\n".join(f"  {s}" for s in dict.fromkeys(fixes))
        example = next(iter(config.registry()), "")
        raise SystemExit(
            f"Cannot start: the roster needs {', '.join(sorted(needs))}.\n{detail}\n\n"
            f"{steps}\n\n"
            "To price the run without any of this, add --dry-run.\n"
            f"To run a single model, e.g.: python run.py --grid a --models {example}"
        )


def dry_run(models, conditions, tasks, thinking, fewshot_cache):
    """Render every prompt and price the run, without contacting any provider.

    Token counts are estimated from character length (~3.6 chars/token for code
    and prose of this shape), so treat the totals as an order-of-magnitude guide
    rather than an invoice. Output length is assumed, not known.
    """
    CHARS_PER_TOKEN = 3.6
    assumed_output = 1400 if thinking else 400

    print(f"{'model':<20}{'in tok':>12}{'out tok (est)':>15}{'$ est':>10}")
    print("-" * 57)
    grand = 0.0
    for model in models:
        price = config.PRICING.get(model["name"])
        in_tokens = out_tokens = 0
        for condition in conditions:
            fewshot = fewshot_cache[(condition,)]
            for task in tasks:
                user = prompts.build_task_prompt(task, condition)
                in_tokens += int((len(fewshot) + len(user)) / CHARS_PER_TOKEN)
                out_tokens += assumed_output
        if price:
            cost = in_tokens / 1e6 * price[0] + out_tokens / 1e6 * price[1]
            grand += cost
            cost_s = f"${cost:,.2f}"
        else:
            cost_s = "?"
        print(f"{model['name']:<20}{in_tokens:>12,}{out_tokens:>15,}{cost_s:>10}")
    print("-" * 57)
    print(f"{'TOTAL (no caching)':<20}{'':>12}{'':>15}{'$' + format(grand, ',.2f'):>10}")
    fewshot_share = 100.0 * sum(
        len(fewshot_cache[(c,)]) * len(tasks) for c in conditions
    ) / max(1, sum(len(fewshot_cache[(c,)]) * len(tasks)
                   + sum(len(prompts.build_task_prompt(t, c)) for t in tasks)
                   for c in conditions))
    print(f"\nThe repeated few-shot block is ~{fewshot_share:.0f}% of all input tokens.")
    print("OpenAI and DeepSeek both apply automatic prompt caching to a repeated")
    print("prefix, so the real input cost should land below this; the discount is")
    print("not under our control, so it is not modelled. Treat this as an upper bound.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["a", "b"], default="a",
                    help="a = Table 1 replication (reasoning off); b = reasoning-on ablation")
    ap.add_argument("--limit", type=int, default=None, help="only the first N tasks")
    ap.add_argument("--models", default=None, help="comma-separated subset of model names")
    ap.add_argument("--conditions", default=None, help="comma-separated subset of conditions")
    ap.add_argument("--concurrency", type=int, default=config.CONCURRENCY)
    ap.add_argument("--dry-run", action="store_true",
                    help="render prompts and estimate cost without calling any API")
    args = ap.parse_args()

    config.load_local_keys()
    registry = config.registry()

    if args.grid == "a":
        names = list(registry)
        conditions = list(prompts.CONDITIONS)
        thinking = False
    else:
        names = list(config.THINKING_ABLATION_MODELS)
        conditions = list(config.THINKING_ABLATION_CONDITIONS)
        thinking = True

    if args.models:
        names = [n.strip() for n in args.models.split(",")]
    if args.conditions:
        conditions = [c.strip() for c in args.conditions.split(",")]

    unknown = [n for n in names if n not in registry]
    if unknown:
        raise SystemExit(f"unknown model(s): {', '.join(unknown)}")

    models = [registry[n] for n in names]
    if thinking:
        skipped = [m["name"] for m in models if not m["thinking"]]
        models = [m for m in models if m["thinking"]]
        if skipped:
            print(f"grid b: skipping non-reasoning model(s): {', '.join(skipped)}")

    tasks = load_tasks(args.limit)
    done = load_done()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Rendering the few-shot block is deterministic and identical across tasks.
    fewshot_cache = {(c,): prompts.build_fewshot_block(c) for c in conditions}

    cells = build_cells(models, conditions, thinking, tasks, done)
    total = len(models) * len(conditions) * len(tasks)
    print(f"grid {args.grid}: {len(models)} models x {len(conditions)} conditions "
          f"x {len(tasks)} tasks = {total} cells")
    print(f"already done: {total - len(cells)}, to run: {len(cells)}\n")
    if not cells:
        return

    if args.dry_run:
        dry_run(models, conditions, tasks, thinking, fewshot_cache)
        return

    preflight(models)

    completed = [0]
    failures = [0]
    lock = threading.Lock()

    def worker(cell):
        model, condition, task = cell
        rec = run_cell(model, condition, task, thinking, fewshot_cache)
        emit(rec)
        with lock:
            completed[0] += 1
            if rec["error"]:
                failures[0] += 1
                print(f"  [{completed[0]}/{len(cells)}] ERROR {model['name']}/"
                      f"{condition}/{task['task_id']}: {rec['error'][:100]}")
            elif completed[0] % 25 == 0 or completed[0] == len(cells):
                print(f"  [{completed[0]}/{len(cells)}] ok")

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(worker, cells))

    print(f"\ndone. {completed[0] - failures[0]} succeeded, {failures[0]} failed.")
    print(f"-> {OUT}\nNext: python evaluate.py")


if __name__ == "__main__":
    main()
