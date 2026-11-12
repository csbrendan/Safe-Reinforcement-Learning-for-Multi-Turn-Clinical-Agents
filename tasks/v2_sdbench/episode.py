"""One SD-Bench episode as a step-wise state machine, reproducing sdbench.agents.baseline.run.

Why not call baseline.run directly: GRPO must batch one policy turn across many episodes through the trainer's
vLLM client, so the loop has to be driven from outside. Evaluation uses this same class, so the training and
evaluation interaction protocols are identical.

Prompts, the results format and the tag parsers are SD-Bench's own (tests/test_v2_sdbench.py checks the copied
prompt text against baseline.run's source). Deliberate differences, all to be reported:
  1. When turns run out, SD-Bench appends the "No more tests" request as a second consecutive user message. We
     merge it into the previous user message with a blank line, exactly as the benchmark-only merge-roles chat
     template does, because MedGemma's template rejects consecutive user turns.
  2. SD-Bench never counts a reply with no tags against the turn budget, so a model that never emits tags loops
     forever. We cap total agent replies at `max_agent_msgs` (default 2 * max_turns), then force the diagnosis.
  3. A reply is cut at the first `<end_of_turn>`. Our text-only MedGemma's generation_config lists only <eos> as
     a stop token, so without a stop it writes past its turn: fake "model" turns, duplicate <diagnosis> tags,
     invented results. Servers should stop on id 106 anyway; this is the safety net, and `n_runon` counts it.
"""
from __future__ import annotations
import re
from typing import Any

from tasks.v2_sdbench import sdb

SYSTEM_PROMPT = """\
You are a diagnostic assistant. Order tests and ask patient questions to determine the diagnosis.

To order tests use <test></test> tags:
<test>CBC</test>
<test>Chest X-ray</test>
...more tests...

You can also ask questions directly (make sure to put each question in a separate <question> tag):
<question>What are your symptoms?</question>
<question>What is your medical history?</question>
...more questions...

You cannot mix <test> and <question> tags in the same turn, just use all <test> tags or all <question> tags. Make sure to ask for enough questions and tests to reach a diagnosis.

When ready to diagnose, use <diagnosis></diagnosis> tags to provide ONE final diagnosis:
<diagnosis>Your diagnosis here</diagnosis>
"""
FIRST_USER = "Patient presentation:\n\n{case_summary}\n\nWhat questions would you like to ask?\nOr what tests would you like to order?"
DEFAULT_SUMMARY = "A new patient case is available. You need to ask questions and order tests to reach a diagnosis."
NAG = "Please order tests using <test> tags, questions using <question> tags, or provide diagnosis using <diagnosis> tags."
FORCE = "No more tests available. Please provide your diagnosis using <diagnosis> tags."
_DIAG_TAG = re.compile(r"<diagnosis>", re.I)
TURN_END = "<end_of_turn>"
END_OF_TURN_ID = 106  # Gemma 3 / MedGemma
_SPECIALS = re.compile(r"<(eos|bos|pad|start_of_turn|end_of_turn)>")


def clean_reply(text: str) -> tuple[str, bool]:
    """Cut at the first <end_of_turn> and drop any remaining special-token text. Returns (text, was_cut)."""
    text = text or ""
    if not _SPECIALS.search(text):
        return text, False  # unchanged: identical to pre-fix behaviour when the server already stopped
    i = text.find(TURN_END)
    cut = i >= 0
    if cut:
        text = text[:i]
    return _SPECIALS.sub("", text).rstrip(), cut


def first_user_message(case: dict) -> str:
    summary = case.get("case_summary") or DEFAULT_SUMMARY
    return FIRST_USER.format(case_summary=summary)


class Episode:
    def __init__(self, case: dict, runner, max_agent_msgs: int | None = None):
        self.case = case
        self.runner = runner
        self.collector: dict[str, Any] = runner._collector
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT},
                                     {"role": "user", "content": first_user_message(case)}]
        self.max_agent_msgs = max_agent_msgs or 2 * runner.max_turns
        self.n_agent_msgs = 0
        self.n_nags = 0
        self.forced = False
        self.done = False
        self.diagnosis: str | None = None
        self.n_diagnosis_tags = 0
        self.raw_final = ""
        self.n_runon = 0        # replies that ran past <end_of_turn> (cut here)
        self.n_truncated = 0    # replies that hit the token cap (set by the caller)

    def _user(self, text: str) -> None:
        if self.messages[-1]["role"] == "user":
            self.messages[-1] = {"role": "user", "content": self.messages[-1]["content"] + "\n\n" + text}
        else:
            self.messages.append({"role": "user", "content": text})

    def step(self, content: str) -> bool:
        """Feed one agent reply. Returns True when the episode has ended."""
        assert not self.done, "episode already finished"
        content, cut = clean_reply(content)
        self.n_runon += cut
        self.messages.append({"role": "assistant", "content": content})
        self.n_agent_msgs += 1
        diagnosis = sdb.baseline.extract_diagnosis_from_response(content)
        if diagnosis or self.forced:
            self.diagnosis = diagnosis
            self.n_diagnosis_tags = len(_DIAG_TAG.findall(content))
            self.raw_final = content
            self.done = True
            return True
        tests = sdb.baseline.extract_test_list_from_response(content)
        if tests:
            batch = self.runner(tests)
            text = "Results:\n"
            for r in batch.test_results:
                text += f"- {r['test_name']}: {r['results']}\n"
            if batch.limit_messages:
                text += f"\nSYSTEM NOTICE: {'; '.join(batch.limit_messages)}\n"
            self._user(text)
        else:
            self.n_nags += 1
            self._user(NAG)
        if self.runner.turns_left <= 0 or self.n_agent_msgs >= self.max_agent_msgs:
            self._user(FORCE)
            self.forced = True
        return False

    def record(self) -> dict:
        """Everything needed for reward, metrics and audit (no reward computed here)."""
        logs = self.collector.get("test_logs", [])
        return {
            "case_id": str(self.case["case_id"]),
            "diagnosis": self.diagnosis,
            "raw_final": self.raw_final,
            "n_diagnosis_tags": self.n_diagnosis_tags,
            "forced": self.forced,
            "n_agent_msgs": self.n_agent_msgs,
            "n_nags": self.n_nags,
            "n_runon": self.n_runon,
            "n_truncated": self.n_truncated,
            "num_rounds": self.runner.num_rounds,
            "num_test_rounds": self.runner.num_test_rounds,
            "requested_tests": self.runner.requested_tests,
            "requested_tests_only": self.runner.requested_tests_only,
            "test_logs": logs,
            "messages": self.messages,
        }
