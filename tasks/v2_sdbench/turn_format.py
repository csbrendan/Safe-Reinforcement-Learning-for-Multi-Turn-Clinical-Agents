"""Policy-side turn-format interventions (opt-in via the env config key "policy_format").

SD-Bench rejects a whole turn that mixes <test> and <question> tags (test_runner_enforce_single_turn_type=True,
its default, which we keep). The base MedGemma mixes them in most sampled turns (V2/sdbench_env_validation.md §3).
These variants change only the agent side; the environment, gatekeeper, judge and scoring are untouched:

  official            SD-Bench's baseline agent prompt, unconstrained decoding (default; the published protocol).
  prompt              the same prompt plus one sentence restating the one-tag-type rule as a two-turn plan.
  constrained         unchanged prompt; decoding is constrained (vLLM structured outputs) to a reply whose tags are
                      all <question>, all <test>, or one <diagnosis>, optionally after tag-free prose.
  prompt+constrained  both.

The constraint is part of the agent's decoding interface, so if an arm trains with it, every arm and every
evaluation of that arm must use it too (report it as such). It cannot produce a reply SD-Bench would reject for
mixing; it can still produce a reply with no closing tag if the token cap cuts it.
"""
from __future__ import annotations

VARIANTS = ("official", "prompt", "constrained", "prompt+constrained")

PROMPT_ADDENDUM = ("Each reply must use only one kind of tag. If you need both, ask your questions in this reply "
                   "and order the tests in your next reply.")

# Tag contents must contain a non-space character; prose (without '<') may precede the tags and sit between them.
# Unbounded repeats on purpose: counted repeats ({0,1500}) cost the grammar engine 30x more CPU to compile (3.0 s
# vs 0.1 s, xgrammar, 8 threads), and that CPU burst heated the Spark's SoC by ~10 C in 2 s. max_tokens bounds length.
_Q = r"<question>\s*[^<\s][^<]*</question>"
_T = r"<test>\s*[^<\s][^<]*</test>"
_D = r"<diagnosis>\s*[^<\s][^<]*</diagnosis>"
_GAP = r"[^<]*"
TURN_REGEX = rf"[^<]*(({_Q}{_GAP})+|({_T}{_GAP})+|{_D}[^<]*)"


def check(variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"policy_format must be one of {VARIANTS}, got {variant!r}")
    return variant


def system_prompt(base: str, variant: str) -> str:
    return base.rstrip("\n") + "\n\n" + PROMPT_ADDENDUM + "\n" if "prompt" in check(variant) else base


def constrained(variant: str) -> bool:
    return "constrained" in check(variant)


def openai_extra(variant: str) -> dict:
    """extra_body for an OpenAI-compatible vLLM server."""
    return {"structured_outputs": {"regex": TURN_REGEX}} if constrained(variant) else {}


def trl_kwargs(variant: str) -> dict:
    """kwargs for trl VLLMClient.chat / generate."""
    return {"structured_outputs_regex": TURN_REGEX} if constrained(variant) else {}
