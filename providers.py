"""Two backends: the Anthropic SDK for Claude, an OpenAI-compatible client
(pointed at OpenRouter) for GPT / DeepSeek / Qwen.

Both expose the same call signature so run.py stays provider-agnostic. The
few-shot block is passed separately from the task so it can be marked for prompt
caching -- it is identical across all 164 tasks in a condition.
"""

import os
import re

import config

# Accept ANY info string after the fence, not just python/py. Models routinely
# emit ```text or ```output blocks mid-explanation; a pattern that only matches
# ```python fails to consume those, desynchronises, and starts pairing a closing
# fence with the next opening one -- which captures prose as if it were code.
_CODE_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def extract_code(text, entry_point=None):
    """Pull the fixed program out of the reply.

    The paper notes GPT models tend to write ```python rather than a bare fence,
    which the regex above absorbs.

    Choosing the block matters more than it looks. Taking simply the last block
    is wrong: under CoT models often close with an illustrative snippet after the
    fix ("This satisfies: ```assert truncate_number(3.5) == 0.5```"), and grabbing
    that scores a correct repair as a NameError. Since only the CoT conditions
    produce such trailing prose, the error is not evenly spread across conditions
    -- it manufactures exactly the CoT penalty this experiment is measuring. So
    prefer the last block that actually defines the function under repair.
    """
    if not text:
        return ""
    blocks = [b.strip() for b in _CODE_BLOCK.findall(text)]
    if not blocks:
        return text.strip()

    if entry_point:
        defining = [b for b in blocks
                    if re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", b, re.M)]
        if defining:
            return defining[-1]

    # No entry point given, or the model renamed it: fall back to any block that
    # defines something, then to the last block.
    any_def = [b for b in blocks if re.search(r"^\s*def\s+\w+\s*\(", b, re.M)]
    if any_def:
        return any_def[-1]
    return blocks[-1]


class AnthropicBackend:
    def __init__(self):
        import anthropic

        self._client = anthropic.Anthropic()

    def complete(self, model_id, fewshot, task, thinking=False):
        kwargs = {
            "model": model_id,
            "max_tokens": config.MAX_OUTPUT_TOKENS,
            # Cache the few-shot block: constant across every task in a condition.
            "system": [
                {"type": "text", "text": fewshot, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": task}],
        }
        if thinking:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": "medium"}
        else:
            # Grid A: no reasoning tokens, matching the paper's models.
            kwargs["thinking"] = {"type": "disabled"}
            kwargs["output_config"] = {"effort": "low"}

        resp = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if b.type == "text")
        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "cache_read": getattr(resp.usage, "cache_read_input_tokens", 0),
            "cache_write": getattr(resp.usage, "cache_creation_input_tokens", 0),
        }
        return text, usage


class OpenRouterBackend:
    def __init__(self):
        from openai import OpenAI

        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self._client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        # Remembers models that refuse `enabled: False`, so the fallback is
        # discovered once rather than on every one of 492 calls.
        self._reasoning_floor = {}

    def complete(self, model_id, fewshot, task, thinking=False):
        # OpenRouter normalises reasoning control across providers. Models with no
        # reasoning mode ignore it, so it is safe to send unconditionally.
        #
        # `exclude: True` is NOT "reasoning off" -- it hides reasoning from the
        # response while still generating (and billing) it. On DeepSeek V4 that
        # burned the whole max_tokens budget on hidden reasoning and returned an
        # empty answer on 2.7% of calls, which scored as repair failures. Grid A
        # is supposed to match the paper's non-reasoning models, so it must use
        # `enabled: False`, which genuinely disables it.
        if thinking:
            modes = [("medium", {"reasoning": {"effort": "medium"}})]
        else:
            # Some models (GLM-5.3) reject `enabled: False` outright with
            # "Reasoning is mandatory for this endpoint". They cannot join the
            # true reasoning-off arm, so fall back to the lowest effort they
            # accept and record that, rather than dropping them or silently
            # pretending the condition was met.
            modes = [("off", {"reasoning": {"enabled": False}}),
                     ("minimal", {"reasoning": {"effort": "minimal"}})]
            floor = self._reasoning_floor.get(model_id)
            if floor == "minimal":
                modes = modes[1:]

        last_exc = None
        for mode, extra in modes:
            try:
                resp = self._client.chat.completions.create(
                    model=model_id,
                    max_tokens=config.MAX_OUTPUT_TOKENS,
                    # Table 1 is greedy decoding. Without this the provider
                    # default (typically 0.7-1.0) applies, which makes the run
                    # unreproducible and adds sampling noise to deltas that are
                    # already smaller than the noise floor.
                    temperature=0,
                    messages=[
                        {"role": "system", "content": fewshot},
                        {"role": "user", "content": task},
                    ],
                    extra_body=extra,
                )
                break
            except Exception as exc:
                if "Reasoning is mandatory" not in str(exc):
                    raise
                self._reasoning_floor[model_id] = "minimal"
                last_exc = exc
        else:
            raise last_exc

        text = resp.choices[0].message.content or ""
        u = resp.usage
        usage = {
            "input_tokens": getattr(u, "prompt_tokens", 0),
            "output_tokens": getattr(u, "completion_tokens", 0),
            "cache_read": 0,
            "cache_write": 0,
            "reasoning_mode": mode,
        }
        return text, usage


_CACHE = {}


def get_backend(name):
    if name not in _CACHE:
        _CACHE[name] = {"anthropic": AnthropicBackend, "openrouter": OpenRouterBackend}[name]()
    return _CACHE[name]
