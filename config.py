"""Model roster.

Each entry is paired against a model the paper actually tested, so the results
table reads as "2024 model -> 2026 successor" on the same benchmark.

`thinking` marks models that expose a reasoning-effort control. Grid A (the
replication of Table 1) runs every model with reasoning OFF, which is the
apples-to-apples setting -- none of the paper's models had reasoning tokens.
Grid B re-runs a subset with reasoning ON to separate "the trace helped" from
"the extra thinking tokens helped".
"""

import os
import pathlib

# Local key files, so the keys are not pasted into shell history or committed.
# Each holds a single bare key. An already-set environment variable always wins.
_KEY_FILES = {
    "OPENROUTER_API_KEY": ".openrouter",
    "ANTHROPIC_API_KEY": ".anthropic",
}


def load_local_keys():
    here = pathlib.Path(__file__).parent
    for env_var, filename in _KEY_FILES.items():
        if os.environ.get(env_var):
            continue
        path = here / filename
        if path.exists():
            value = path.read_text().strip().splitlines()
            if value and value[0].strip():
                os.environ[env_var] = value[0].strip()


# Snapshots the 2026-09-01 Grid A run actually resolved to. The slugs below are
# undated aliases, so cite these when reporting results -- an alias rollover would
# otherwise change what a re-run measures without any code change:
#   openai/gpt-5.6-luna       -> openai/gpt-5.6-luna-20260709
#   deepseek/deepseek-v4-pro  -> deepseek/deepseek-v4-pro-20260423
#   z-ai/glm-5.3              -> z-ai/glm-5.3-20260816
#   qwen/qwen3-coder-30b-a3b-instruct  (no dated snapshot published)
# Note also that OpenRouter picks an inference provider per request and does not
# pin it (qwen was served by Novita, the source of the empty-completion retries).
MODELS = [
    # name              backend       model_id                     replaces               thinking
    # `luna` is the cost-efficient tier of the GPT-5.6 series, so it maps to the
    # paper's cheap OpenAI model rather than to GPT-4. That pairing is the more
    # interesting one anyway: GPT-3.5 is the ONE model where CoT helped (70.1 ->
    # 75.6). Swap to `openai/gpt-5.6-sol` (flagship, $2/$10) for a GPT-4 match.
    ("gpt-5.6-luna",    "openrouter", "openai/gpt-5.6-luna",       "GPT-3.5-turbo-1106",  True),
    ("deepseek-v4-pro", "openrouter", "deepseek/deepseek-v4-pro",  "DeepSeek Coder 33B",  True),
    # Open-weight, code-specific, non-reasoning instruct model -- the closest
    # match to the paper's StarCoder 15.5B, which suffered the sharpest collapse
    # in Table 1 (29.3 -> 16.5). Verified to accept reasoning `enabled: False`.
    ("qwen3-coder-30b", "openrouter", "qwen/qwen3-coder-30b-a3b-instruct",
                                                                   "StarCoder 15.5B",     False),
    # Open-weight generalist, standing in for the paper's Mixtral 8x7B.
    # NOTE: GLM-5.3 rejects `reasoning: {enabled: False}` ("Reasoning is mandatory
    # for this endpoint"), so it CANNOT join the true reasoning-off arm; the
    # backend falls back to effort=minimal and records reasoning_mode per call.
    ("glm-5.3",         "openrouter", "z-ai/glm-5.3",              "Mixtral 8x7B",        True),
    # Claude is supported but out of the current roster, so only OPENROUTER_API_KEY
    # is needed. Uncomment (and set ANTHROPIC_API_KEY) to add it back:
    # ("sonnet-5",      "anthropic",  "claude-sonnet-5",           "PaLM 2-L",            True),
]

# Grid B: reasoning-on ablation. All three expose a reasoning control.
THINKING_ABLATION_MODELS = ("gpt-5.6-luna", "deepseek-v4-pro", "glm-5.3")
THINKING_ABLATION_CONDITIONS = ("cot_trace", "cot_notrace")

# USD per million tokens (input, output), for the --dry-run cost estimate only.
# Anthropic list price; OpenRouter prices as advertised at time of writing.
PRICING = {
    "gpt-5.6-luna": (0.20, 1.20),
    "deepseek-v4-pro": (1.60, 3.20),
    "glm-5.3": (1.40, 4.40),
    "qwen3-coder-30b": (0.07, 0.28),
    # not in the current roster, kept so --models can re-enable them
    "gpt-5.6-sol": (2.00, 10.00),
    "sonnet-5": (3.00, 15.00),
}

# Headroom so a reasoning model cannot spend the whole budget before answering.
MAX_OUTPUT_TOKENS = 8192
CONCURRENCY = 8


def registry():
    return {
        name: {
            "name": name,
            "backend": backend,
            "model_id": model_id,
            "replaces": replaces,
            "thinking": thinking,
        }
        for name, backend, model_id, replaces, thinking in MODELS
    }
