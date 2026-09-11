# NExT replication — how to run it

Re-runs the preliminary study from **NExT: Teaching Large Language Models to Reason
about Code Execution** ([arXiv:2404.14662](https://arxiv.org/abs/2404.14662), §3 /
Table 1) on current models. Same benchmark (HumanEvalFix-Plus, 164 tasks), same three
prompting conditions, same 3-shot prompt from the paper's Appendix E.

---

## Setup

```bash
conda activate myenv                    # python 3.11
pip install -r requirements.txt         # anthropic, openai, numpy
```

Put your OpenRouter key in a file called `.openrouter` (single line, bare key). It is
loaded automatically and is gitignored:

```bash
echo "sk-or-v1-..." > .openrouter
```

`ANTHROPIC_API_KEY` / `.anthropic` is only needed if you re-enable a Claude model in
`config.py`. The current roster is OpenRouter-only.

---

## Commands

```bash
python build_dataset.py                 # already done; regenerates data/hefixplus.jsonl
python run.py --grid a --dry-run        # price it, no API calls
python run.py --grid a --limit 5        # smoke test, a few cents
python run.py --grid a                  # full run: 4 models x 3 conditions x 164 tasks
python evaluate.py                      # execute the fixes, print the table
```

Useful flags:

| Flag | Effect |
|---|---|
| `--limit N` | first N tasks only |
| `--models gpt-5.6-luna` | subset of the roster (comma-separated) |
| `--conditions cot_trace` | subset of conditions |
| `--concurrency 12` | parallel requests (default 8) |
| `--grid b` | reasoning-ON ablation instead of the replication |

`run.py` appends to `results/generations.jsonl` and **skips cells already present**, so
Ctrl-C is safe and re-running resumes. To redo a cell, delete its line.

Full run takes ~15 min at `--concurrency 12`; `evaluate.py` takes ~4 min (it executes
1,968 programs in subprocesses).

---

## Verifying it does the right thing

`inspect_cell.py` dumps the prompt that was sent, the raw reply, the code that got
extracted, and the pass/fail — for any single cell. Everything is re-derived live from
the dataset and `results/`, so what you see is what was scored.

```bash
python inspect_cell.py --task Python/9
python inspect_cell.py --task Python/9 --condition cot_notrace --model gpt-5.6-luna
python inspect_cell.py --task Python/0 --prompt-only     # works before any run
python inspect_cell.py --task Python/0 --show-fewshot    # also print the 3 exemplars
```

### What the prompt looks like

The trace is not a separate transcript — each line of the buggy program carries its
execution state as a `#` comment.

```
Here is the specification in natural language and the program that I wrote:
```
# Fix the bug in `rolling_max`.
from typing import List, Tuple


def rolling_max(numbers: List[int]) -> List[int]:
    running_max = None
    result = []

    for n in numbers:
        if running_max is None:
            running_max = n
        else:
            running_max = max(numbers)

        result.append(running_max)

    return result

```
Here are the assertions I need to pass:
```
assert rolling_max([1, 2, 3, 2, 3, 4, 2]) == [1, 2, 3, 3, 3, 4, 4]
```

Here is a *cheatsheet* for the gold execution trace for each assertions, use the
information here but pretend you did all the reasoning about execution yourself.

#################### cheatsheet starts ####################
def rolling_max(numbers: List[int]) -> List[int]: # (0) numbers=[1, 2, 3, 2, 3, 4, 2];
    running_max = None # (1) running_max=None;
    result = [] # (2) result=[];

    for n in numbers: # (3) n=1; (6) n=2; ...; (21) n=2;
        if running_max is None:
            running_max = n # (4) running_max=1;
        else:
            running_max = max(numbers) # (7) running_max=4; (10) NO_CHANGE; ...; (22) NO_CHANGE;

        result.append(running_max) # (5) result=[1]; (8) result=[1, 4]; ...; (23) result=[1, 4, 4, 4, 4, 4, 4];

    return result # (24) __return__=[1, 4, 4, 4, 4, 4, 4];


assert rolling_max([1, 2, 3, 2, 3, 4, 2]) == [1, 2, 3, 3, 3, 4, 4] # (25) __exception__ = AssertionError()
#################### cheatsheet ends ####################

Given the above information, can you tell me why my code is wrong, and also provide a
fixed version?
```

**Four things to check in that trace** (paper §2 / App. A.1):

- states numbered by **execution order**, `(0)`…`(25)` — not line order
- **only changed variables** appear; unchanged ones are never repeated
- loops **compress**: past 3 states, first two + last, joined by `...`
- `NO_CHANGE` means "ran, changed nothing" — distinct from a line never reached

Bare `if` lines correctly get no state at all: App. A.1 notes `sys.settrace()` cannot
observe branch conditions. `tracer.py` reproduces the paper's Figure 2 numbering
state-for-state, which is the cheapest way to confirm the trace format is faithful.

### What the model returns, and what gets graded

```
==============================================================================
RAW RESPONSE  |  model=gpt-5.6-luna
==============================================================================
The bug is in this line:

```python
running_max = max(numbers)
```

It computes the maximum of the entire list on every iteration. After processing the
first element, this makes `running_max` equal to `4`, even though `4` has not been
reached yet.
...

==============================================================================
EXTRACTED CODE  (this is what gets executed)
==============================================================================
def rolling_max(numbers: List[int]) -> List[int]:
    result = []
    running_max = None

    for n in numbers:
        running_max = n if running_max is None else max(running_max, n)
        result.append(running_max)

    return result

==============================================================================
VERDICT
==============================================================================
  usage        : {'input_tokens': 3168, 'output_tokens': 296, 'reasoning_mode': 'off'}
  open tests   : 1 shown to the model
  EvalPlus     : PASS  (pass)
```

`reasoning_mode` should read `off` for every model except `glm-5.3`, which reports
`minimal` — it rejects `reasoning: {enabled: false}` outright and cannot run
reasoning-off at all.

### What `evaluate.py` prints

```
HumanEvalFix-Plus repair accuracy (%) - reasoning OFF (Grid A - replicates Table 1)
==============================================================================
Prompting method        gpt-5.6-luna deepseek-v4-pro qwen3-coder-30b         glm-5.3     Avg.
---------------------------------------------------------------------------------------------
Vanilla w/ trace                73.8            80.5            79.5            84.1     79.5
+ CoT                           78.0            77.4            74.4            87.8     79.4  -0.1
+ CoT; - trace                  75.6            80.5            72.0            86.0     78.5  -0.9

  n = 164 tasks per cell; 12 model x condition cells scored
```

It then reprints the paper's Table 1 underneath for comparison. Deltas are against the
**row above**, matching the paper's convention (their 48.4 − 44.1 = 4.3).

### Harness sanity check

The benchmark itself should behave before you trust any model number:

```bash
python -c "
import json, sandbox
from concurrent.futures import ThreadPoolExecutor
recs=[json.loads(l) for l in open('data/hefixplus.jsonl')]
def chk(a):
    r,k=a; ok,_=sandbox.run(r[k], r['eval_test'], r['entry_point']); return k,ok
jobs=[(r,k) for k in ('canonical_program','buggy_program') for r in recs]
with ThreadPoolExecutor(max_workers=10) as ex: res=list(ex.map(chk,jobs))
for k in ('canonical_program','buggy_program'):
    s=[o for kk,o in res if kk==k]; print(f'{k}: {sum(s)}/{len(s)}')
"
```

Expected: `canonical_program: 144/164`, `buggy_program: 0/164`.

Buggy must be **0** — every task is genuinely broken. Canonical is **144, not 164**,
because HumanEval's original solutions fail EvalPlus's stricter tests on 20 tasks
(EvalPlus's own published finding). The EvalPlus reference solution scores 163/164, so
the harness is sound and the ceiling is ~99%, but **models must beat the original
reference to exceed 87.8%**. Report that number alongside any result.

---

## The three conditions

| Condition | Trace in prompt? | Answer format |
|---|---|---|
| `vanilla_trace` | yes | fixed code only |
| `cot_trace` | yes | NL rationale, then the fix ← **the claim under test** |
| `cot_notrace` | no | NL rationale, then the fix |

Grid A runs all three with reasoning disabled — the apples-to-apples match for the
paper's models, none of which had reasoning tokens. Grid B (`--grid b`) re-runs
`cot_trace` and `cot_notrace` with reasoning on, to separate "the trace helped" from
"the extra thinking tokens helped".

---

## Files

| File | Role |
|---|---|
| `tracer.py` | Inline trace representation (§2, App. A.1) |
| `build_dataset.py` | Builds HeFix+, precomputes traces → `data/hefixplus.jsonl` |
| `prompts.py` | The 3-shot prompt (App. E) and the 3 conditions |
| `config.py` | Model roster, pricing, key loading |
| `providers.py` | OpenRouter + Anthropic backends, code extraction |
| `run.py` | Resumable generation driver, `--dry-run` costing |
| `sandbox.py` | Subprocess execution with timeout |
| `evaluate.py` | Scoring + Table 1 report |
| `inspect_cell.py` | Dump prompt / response / extraction / verdict for one cell |

---

## Gotchas worth knowing

**Extraction policy is a real degree of freedom.** When a reply contains several code
blocks, we grade the *last* one that defines the entry point. Models under CoT often
append an alternative ("a shorter version, assuming…"), and multi-block replies appear
in 9.3% of vanilla answers but 21–23% of CoT answers — so the choice presses harder on
the conditions under test. Grading the *first* block instead moves the CoT delta from
−0.1 to −1.7. Neither is near the paper's −7.8, but report the range, not a point.

**Empty completions are retried, not scored.** Providers behind OpenRouter
intermittently return empty content (~29% of qwen calls in one run). These raise and
retry; if a cell never yields text it is written with an `error` and excluded, rather
than being scored as a wrong fix.

**Model aliases are undated.** `config.py` sends `openai/gpt-5.6-luna` etc., which
resolved on 2026-09-01 to `gpt-5.6-luna-20260709`, `deepseek-v4-pro-20260423`,
`glm-5.3-20260816`. Cite the dated snapshots — an alias rollover silently changes what a
re-run measures. OpenRouter also picks an inference provider per request and does not
pin it.

**Effect sizes are small.** At n=164, one task = 0.6%. Treat anything under ~5 points as
noise, and prefer the paired McNemar test over comparing percentages.
