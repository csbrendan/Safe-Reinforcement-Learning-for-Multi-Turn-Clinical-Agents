"""Env-parameterized Claude process-reward judge for use inside the GRPO rollout loop.

Kept separate from judge_pilot.py (which is bound to the deephedge validation env) so grpo_disagg can pass its
OWN env module. The judge receives ONLY the student-visible transcript (presentation + revealed findings +
actions) — never true risk, hidden findings, oracle actions, utilities, or unrevealed findings.
"""
from __future__ import annotations
import hashlib, time
from framework.distill.teacher_client import TeacherClient


def build_judge_sys(env):
    return (
        "You are an expert acute-care clinician AUDITING THE PROCESS of a diagnostic work-up agent. You are shown "
        "ONLY what the agent saw: the presentation, the findings it revealed, and the actions it took. You do NOT "
        "know the patient's true severity or any unrevealed finding.\n"
        "Judge whether each turn was a good next step, and whether the trajectory gathered appropriately before "
        "committing. Emphasize:\n"
        "- was the available evidence SUFFICIENT to commit safely, or did the agent stop before resolving "
        "dangerous uncertainty;\n"
        "- could another question MATERIALLY change the disposition;\n"
        "- were any questions REDUNDANT or burdensome;\n"
        "- do NOT penalize a clinically reasonable question ORDER merely because it differs from some optimal "
        "path.\n"
        f"Available actions — order one finding (intent): {', '.join(env.ITEMS)}. "
        f"Commit a disposition (decision): {', '.join(env.ACTIONS)}.\n"
        "Output STRICT JSON only, no prose outside it:\n"
        '{"step_scores":[{"turn":<int>,"useful_information_gain":<-2..2>,"redundancy":<0..2>,'
        '"premature_commitment":<0..2>,"unsafe_clinical_action":<0..2>,"clinical_appropriateness":<-2..2>,'
        '"brief_reason":"<one sentence>"}],"overall_process_score":<-5..5>,"trajectory_rank":null,'
        '"confidence":<0.0..1.0>}'
    )


def render(env, fam, asks, decision, assign):
    lines = [f"Presentation: {env._FAM_DESC[fam]}.", "Chart: no results on file yet.", "",
             "The agent's work-up:"]
    t = 1
    for a in asks:
        it = env.resolve_ask(a)
        res = env._REVEAL[it][assign[env.IDX[it]]] if it else "unrecognized test"
        lines.append(f"  Turn {t}: ordered '{it}' -> Result: {res}.")
        t += 1
    dec = decision if decision in env.ACTIONS else "(no disposition committed)"
    lines.append(f"  Turn {t}: committed disposition '{dec}'.")
    lines += ["", f"Score all {t} turns (each ordered finding, and the final commit)."]
    return "\n".join(lines)


def _content_key(asks, decision):
    return hashlib.sha1(("|".join(asks) + "#" + str(decision)).encode()).hexdigest()[:12]


class RolloutJudge:
    """Caches by (case_id, trajectory-content) so identical rollouts within/across steps aren't re-scored."""
    def __init__(self, env, cache_path, prompt_version="grpo_judge_v1"):
        self.env = env
        self.sys = build_judge_sys(env)
        self.client = TeacherClient(cache_path, prompt_version=prompt_version, max_tokens=3072)
        self.reset_stats()

    def reset_stats(self):
        self.stats = {"calls": 0, "cache_hits": 0, "api_calls": 0, "retries": 0, "parse_fail": 0,
                      "api_latency_s": 0.0}

    def score(self, case, asks, decision):
        """Return the trajectory's overall_process_score in [-5, 5], or None on parse failure.
        Tracks per-call instrumentation (cache hits, live-API latency, retries, parse failures) in self.stats."""
        user = render(self.env, case["family"], asks, decision, case["assign"])
        t0 = time.time()
        rec, from_cache = self.client.label(self.sys, user, case["case_id"], _content_key(asks, decision))
        self.stats["calls"] += 1
        if from_cache:
            self.stats["cache_hits"] += 1
        else:
            self.stats["api_calls"] += 1
            self.stats["api_latency_s"] += time.time() - t0
        if rec["parse_status"] == "ok_retry":
            self.stats["retries"] += 1
        if rec["parse_status"] == "parse_fail":
            self.stats["parse_fail"] += 1
        p = rec["parsed"] or {}
        try:
            return max(-5.0, min(5.0, float(p.get("overall_process_score"))))
        except (TypeError, ValueError):
            return None
