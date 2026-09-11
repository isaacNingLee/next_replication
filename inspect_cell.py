"""Dump exactly what was sent to a model and what came back, for one task.

Use this to check the pipeline is doing what you think: the prompt is rendered
from the dataset, the response is read from results/, and the extracted code is
re-derived live, so what you see is what was scored.

  python inspect_cell.py --task Python/9
  python inspect_cell.py --task Python/9 --condition cot_notrace --model gpt-5.6-luna
  python inspect_cell.py --task Python/0 --prompt-only     # before any run exists
"""

import argparse
import json
import pathlib

import evaluate
import prompts
import providers
import sandbox

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data" / "hefixplus.jsonl"
GENERATIONS = HERE / "results" / "generations.jsonl"

RULE = "=" * 78


def banner(title):
    print(f"\n{RULE}\n{title}\n{RULE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Python/9")
    ap.add_argument("--condition", default="cot_trace", choices=list(prompts.CONDITIONS))
    ap.add_argument("--model", default=None, help="defaults to the first model found")
    ap.add_argument("--prompt-only", action="store_true",
                    help="show only the prompt; does not need a completed run")
    ap.add_argument("--show-fewshot", action="store_true",
                    help="also print the 3-shot exemplar block (long)")
    args = ap.parse_args()

    tasks = {json.loads(l)["task_id"]: json.loads(l) for l in DATA.open()}
    if args.task not in tasks:
        raise SystemExit(f"unknown task {args.task!r}; expected e.g. Python/9")
    task = tasks[args.task]

    if args.show_fewshot:
        banner(f"FEW-SHOT BLOCK  (system message, identical for all 164 tasks)")
        print(prompts.build_fewshot_block(args.condition))

    banner(f"PROMPT SENT  |  {args.task}  |  condition={args.condition}")
    print(prompts.build_task_prompt(task, args.condition))

    if args.prompt_only:
        return

    if not GENERATIONS.exists():
        raise SystemExit(f"\nNo results yet at {GENERATIONS} -- run `python run.py` first, "
                         "or pass --prompt-only.")

    rows = [json.loads(l) for l in GENERATIONS.open()]
    matches = [r for r in rows
               if r["task_id"] == args.task and r["condition"] == args.condition
               and (args.model is None or r["model"] == args.model)]
    if not matches:
        have = sorted({r["model"] for r in rows})
        raise SystemExit(f"\nNo generation for that cell. Models present: {', '.join(have)}")

    rec = matches[0]
    banner(f"RAW RESPONSE  |  model={rec['model']}")
    if rec.get("error"):
        print(f"[errored] {rec['error']}")
        return
    print(rec["response"])

    code = providers.extract_code(rec["response"], task["entry_point"])
    banner("EXTRACTED CODE  (this is what gets executed)")
    print(code if code.strip() else "[empty]")

    preamble = evaluate.import_preamble(task)
    program = f"{preamble}\n{code}" if preamble else code
    passed, detail = sandbox.run(program, task["eval_test"], task["entry_point"])

    banner("VERDICT")
    print(f"  usage        : {rec.get('usage', {})}")
    print(f"  open tests   : {len(task['open_tests'])} shown to the model")
    print(f"  EvalPlus     : {'PASS' if passed else 'FAIL'}  ({detail})")


if __name__ == "__main__":
    main()
