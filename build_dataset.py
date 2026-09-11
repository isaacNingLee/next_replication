"""Build HumanEvalFix-Plus (HeFix+), the benchmark from NExT Sec. 5.

HeFix+ = HumanEvalFix (bigcode/humanevalpack, python config: 164 buggy programs)
evaluated against the stricter EvalPlus test suites (evalplus/humanevalplus).

Per the paper's App. B.2, traces are generated from the *open* tests only, and
those same open tests are the ones shown to the model. Correctness is then judged
on the full EvalPlus suite. Open tests are extracted from `example_test` and kept
only if they pass against the canonical solution -- an assertion lifted out of its
`check()` wrapper that does not pass on correct code is malformed, not a real test.
"""

import ast
import json
import pathlib
import urllib.parse
import urllib.request

import tracer

ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
OUT_PATH = pathlib.Path(__file__).parent / "data" / "hefixplus.jsonl"
MAX_OPEN_TESTS = 3  # the paper's exemplars each carry three assertions

# Two tasks ship an empty `example_test` and a hidden suite built from randomly
# generated strings, so no deterministic assertion can be lifted out of it. Both
# are round-trip decoders whose encoder is defined in the prompt, so we synthesise
# equivalent open tests. These are validated against the canonical solution like
# every other assertion -- nothing is trusted just because it is hard-coded.
SYNTHESIZED_OPEN_TESTS = {
    "Python/38": [
        "assert decode_cyclic(encode_cyclic('abcdefghij')) == 'abcdefghij'",
        "assert decode_cyclic(encode_cyclic('hello world')) == 'hello world'",
        "assert decode_cyclic(encode_cyclic('ab')) == 'ab'",
    ],
    "Python/50": [
        "assert decode_shift(encode_shift('abcdefghij')) == 'abcdefghij'",
        "assert decode_shift(encode_shift('helloworld')) == 'helloworld'",
        "assert decode_shift(encode_shift('xyz')) == 'xyz'",
    ],
}


def fetch_split(dataset, config, split="test", page=100):
    """Page through the HF datasets-server rows API (no `datasets` dependency)."""
    rows, offset = [], 0
    while True:
        query = urllib.parse.urlencode(
            {"dataset": dataset, "config": config, "split": split,
             "offset": offset, "length": page}
        )
        with urllib.request.urlopen(f"{ROWS_ENDPOINT}?{query}", timeout=120) as resp:
            payload = json.load(resp)
        batch = [r["row"] for r in payload["rows"]]
        rows.extend(batch)
        total = payload["num_rows_total"]
        offset += len(batch)
        if offset >= total or not batch:
            return rows


class _RenameCandidate(ast.NodeTransformer):
    """`check(candidate)` bodies refer to the function under test as `candidate`."""

    def __init__(self, entry_point):
        self.entry_point = entry_point

    def visit_Name(self, node):
        if node.id == "candidate":
            return ast.copy_location(ast.Name(id=self.entry_point, ctx=node.ctx), node)
        return node


def extract_assertions(test_src, entry_point):
    """Lift standalone `assert ...` statements out of a check() function."""
    try:
        tree = ast.parse(test_src)
    except SyntaxError:
        return []
    renamer = _RenameCandidate(entry_point)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            cloned = renamer.visit(ast.fix_missing_locations(ast.parse(ast.unparse(node))))
            out.append(ast.unparse(cloned).strip())
    return out


def validate_against(program_src, assertions):
    """Keep only assertions that actually pass on the given (correct) program."""
    kept = []
    for assertion in assertions:
        try:
            _, outcome = tracer.trace(program_src, assertion)
        except Exception:
            continue
        if outcome == "pass":
            kept.append(assertion)
    return kept


def build():
    print("downloading bigcode/humanevalpack (python) ...")
    fix_rows = fetch_split("bigcode/humanevalpack", "python")
    print(f"  {len(fix_rows)} rows")

    print("downloading evalplus/humanevalplus ...")
    plus_rows = fetch_split("evalplus/humanevalplus", "default")
    print(f"  {len(plus_rows)} rows")
    # The two datasets namespace the same problems differently
    # ("Python/0" vs "HumanEval/0"), so join on the numeric index.
    def index_of(task_id):
        return task_id.rsplit("/", 1)[-1]

    plus_by_id = {index_of(r["task_id"]): r for r in plus_rows}

    records, skipped = [], []
    for row in fix_rows:
        task_id = row["task_id"]
        entry_point = row["entry_point"]
        prelude = row.get("import") or ""
        declaration = row["declaration"]

        buggy_program = f"{prelude}{declaration}{row['buggy_solution']}".strip("\n")
        canonical_program = f"{prelude}{declaration}{row['canonical_solution']}".strip("\n")

        # Open tests: prefer example_test, fall back to the hidden suite if absent.
        candidates = extract_assertions(row.get("example_test") or "", entry_point)
        source = "example_test"
        if not candidates:
            candidates = extract_assertions(row.get("test") or "", entry_point)
            source = "test(fallback)"
        open_tests = validate_against(canonical_program, candidates)[:MAX_OPEN_TESTS]

        # Fall back only once validation has actually rejected everything: the
        # hidden suite yields syntactically valid asserts for these tasks that
        # cannot run standalone, so the raw candidate list is a misleading gate.
        if not open_tests and task_id in SYNTHESIZED_OPEN_TESTS:
            source = "synthesized"
            open_tests = validate_against(
                canonical_program, SYNTHESIZED_OPEN_TESTS[task_id]
            )[:MAX_OPEN_TESTS]

        if not open_tests:
            skipped.append(task_id)
            continue

        plus = plus_by_id.get(index_of(task_id))
        if plus is None:
            skipped.append(task_id)
            continue

        # Trace once, here, in the main thread where the SIGALRM wall-clock guard
        # works. Doing it at run time instead would re-execute every buggy program
        # for every model in the roster, and would hang on the tasks that need the
        # timer (HumanEval/160 eval()s an unbounded expression).
        trace_text, trace_outcomes = tracer.trace_all(buggy_program, open_tests)

        records.append(
            {
                "task_id": task_id,
                "trace_text": trace_text,
                "trace_outcomes": trace_outcomes,
                "entry_point": entry_point,
                "prompt": row["prompt"],
                "declaration": declaration,
                "import": prelude,
                "buggy_program": buggy_program,
                "canonical_program": canonical_program,
                "open_tests": open_tests,
                "open_test_source": source,
                # EvalPlus stricter suite -- this is what makes it HeFix "+".
                "eval_test": plus["test"],
                "eval_prompt": plus["prompt"],
            }
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    print(f"\nwrote {len(records)} tasks -> {OUT_PATH}")
    if skipped:
        print(f"skipped {len(skipped)}: {', '.join(skipped)}")

    fallbacks = sum(1 for r in records if r["open_test_source"] != "example_test")
    print(f"open tests from example_test: {len(records) - fallbacks}, from hidden suite: {fallbacks}")


if __name__ == "__main__":
    build()
