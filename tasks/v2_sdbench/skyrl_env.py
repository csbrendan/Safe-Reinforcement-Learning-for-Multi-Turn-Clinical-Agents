"""SkyRL gym environment for SD-Bench: a thin wrapper around Services + Episode (the same code the TRL path uses).

For the Azure-VM trainer (SkyRL on Ray, see V2/skyrl_plan.md). Nothing here changes the interaction protocol,
the reward or the caches: init() starts the same Episode that env.make_rollout_func would, step() feeds one policy
reply to Episode.step, and the final reward is Services.score(ep)["reward"]. Arms B and C differ only in the env
config file, exactly as in the TRL path.

Differences from the TRL rollout loop, and how each is handled:
  * SkyRL drives one episode per env instance (async, thread pool) instead of turn-batching all episodes.
    The per-episode transcript is identical; only scheduling differs.
  * Context guard: SkyRL stops an over-long episode with reward 0 (no format penalty). We do the TRL loop's check
    ourselves (prompt tokens + turn_max_tokens > max_context_tokens -> overflow, scored with the overflow flag), so
    set SkyRL's generator.max_input_length ABOVE max_context_tokens and the guard here fires first.
    Token counts need the policy tokenizer: config key "skyrl": {"tokenizer": "<path>"}.
  * Truncated replies (hit turn_max_tokens) aren't visible to the env; n_truncated is reported as 0 here and the
    trainer's own response-length metrics stand in for it.
  * Turn limit: Episode already forces a diagnosis after 2 * max_turns agent replies. Set SkyRL's
    generator.max_turns to at least 2 * max_turns + 1 so SkyRL never cuts an episode first.

Config: path from the V2_ENV_CONFIG environment variable (a V2/sdbench_configs/*.json file). SkyRL passes custom
envs an empty env_config, so the file path travels by environment variable (set it in the Ray runtime env).

Dataset rows (build with `python -m tasks.v2_sdbench.skyrl_env build-data`) carry only the prompt, env_class and
case id. The prompt contains the case summary, so files for credentialed datasets stay out of git (results/).
"""
from __future__ import annotations
import argparse
import json
import os
import threading
from typing import Any

try:  # the real base class on the SkyRL VM; a minimal stand-in elsewhere so CPU tests run without SkyRL installed
    from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
except ImportError:  # pragma: no cover - exercised only where skyrl_gym is absent
    class BaseTextEnv:  # type: ignore[no-redef]
        def __init__(self):
            self.turns = 0
            self.max_turns = 1

        def close(self):
            pass

    def BaseTextEnvStepOutput(**kw):  # type: ignore[no-redef]  # noqa: N802
        return dict(kw)

ENV_ID = "sdbench"
CONFIG_ENV_VAR = "V2_ENV_CONFIG"
_SERVICES = None
_TOKENIZER = None
_CASES: dict[str, dict] = {}
_LOCK = threading.Lock()


def services():
    """One Services (clients + JSONL caches) per process, shared by every env instance."""
    global _SERVICES, _TOKENIZER
    with _LOCK:
        if _SERVICES is None:
            from tasks.v2_sdbench.env import Services, load_config
            cfg = load_config(os.environ.get(CONFIG_ENV_VAR))
            tok_path = (cfg.get("skyrl") or {}).get("tokenizer")
            if tok_path:
                from transformers import AutoTokenizer
                _TOKENIZER = AutoTokenizer.from_pretrained(tok_path)
            _SERVICES = Services(cfg, tokenizer=_TOKENIZER)
        return _SERVICES


def set_services(svc, tokenizer=None) -> None:
    """Inject a Services-like object (tests)."""
    global _SERVICES, _TOKENIZER
    with _LOCK:
        _SERVICES, _TOKENIZER = svc, tokenizer
        _CASES.clear()


def _case(svc, case_id: str) -> dict:
    with _LOCK:
        if not _CASES:
            from tasks.v2_sdbench.cases import load_split
            _CASES.update({str(c["case_id"]): c for c in load_split(svc.cfg["dataset"], svc.cfg["split"])})
    return _CASES[str(case_id)]


def n_prompt_tokens(tok, messages: list[dict]) -> int:
    """Same count as grpo_disagg._ids(tok, messages, gen=True)."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tok(text, add_special_tokens=False)["input_ids"])


class SDBenchEnv(BaseTextEnv):
    def __init__(self, env_config: Any = None, extras: dict | None = None):
        super().__init__()
        extras = extras or {}
        if "case_id" not in extras:
            raise ValueError("dataset rows must carry case_id (build them with build-data)")
        self.svc = services()
        cfg = self.svc.cfg
        if extras.get("dataset") not in (None, cfg["dataset"]):
            raise ValueError(f"row dataset {extras.get('dataset')!r} != config dataset {cfg['dataset']!r}")
        self.case_id = str(extras["case_id"])
        self.ep = None
        self.score: dict | None = None

    def init(self, prompt):
        from tasks.v2_sdbench.cases import assert_not_test
        cfg = self.svc.cfg
        assert_not_test(cfg["dataset"], [self.case_id])
        self.ep = self.svc.new_episode(_case(self.svc, self.case_id))
        self.ep.overflow = False
        if [dict(m) for m in prompt] != self.ep.messages:
            raise ValueError(f"case {self.case_id}: dataset prompt differs from the Episode's opening messages "
                             "(stale data file or different policy_format?)")
        return self.ep.messages, {}

    def _overflows(self) -> bool:
        tok = _TOKENIZER
        if tok is None:
            return False
        cfg = self.svc.cfg
        return n_prompt_tokens(tok, self.ep.messages) + cfg["turn_max_tokens"] > cfg["max_context_tokens"]

    def step(self, action: str):
        self.turns += 1
        done = self.ep.step(action)
        if not done and self._overflows():
            self.ep.overflow = True
            self.ep.done = done = True
        if not done:
            return BaseTextEnvStepOutput(observations=[self.ep.messages[-1]], reward=0.0, done=False, metadata={})
        self.score = self.svc.score(self.ep)
        log = self.svc.cfg.get("rollout_log")
        if log:
            with _LOCK, open(log, "a") as f:
                f.write(json.dumps(self.score) + "\n")
        return BaseTextEnvStepOutput(observations=[], reward=float(self.score["reward"]), done=True,
                                     metadata={"case_id": self.case_id})

    def get_metrics(self) -> dict:
        s = self.score or {}
        keys = ("reward", "r_out", "judge_score", "passed", "r_gather", "n_tests", "n_questions", "n_grounded",
                "n_repeat", "n_guarded", "total_cost", "total_cost_corrected", "invalid", "overflow", "forced",
                "n_agent_msgs", "n_runon")
        return {k: float(s[k]) for k in keys if k in s}


def register_env() -> None:
    """Call inside the SkyRL entrypoint task, as SkyRL's own custom-env examples do."""
    from skyrl_gym.envs import register
    register(id=ENV_ID, entry_point="tasks.v2_sdbench.skyrl_env:SDBenchEnv")


def build_rows(cfg: dict) -> list[dict]:
    """SkyRL dataset rows for cfg's dataset/split. Same split checks as env.build_train_prompts."""
    from tasks.v2_sdbench.cases import assert_not_test, load_split
    from tasks.v2_sdbench.env import policy_system_prompt
    from tasks.v2_sdbench.episode import first_user_message
    cases = load_split(cfg["dataset"], cfg["split"])
    assert_not_test(cfg["dataset"], [c["case_id"] for c in cases])
    if cfg["split"] != "train" and cfg["dataset"] != "synthetic":
        raise PermissionError("GRPO may only train on the train split")
    return [{"prompt": [{"role": "system", "content": policy_system_prompt(cfg)},
                        {"role": "user", "content": first_user_message(c)}],
             "env_class": ENV_ID, "case_id": str(c["case_id"]), "dataset": cfg["dataset"]} for c in cases]


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-data", help="write SkyRL JSONL rows for the config's dataset/split")
    b.add_argument("--env-config", required=True)
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    from tasks.v2_sdbench.env import load_config
    rows = build_rows(load_config(a.env_config))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
