"""Offline tests: judge-stability metrics and the agent-side turn-format variants."""
import re

import pytest

from tasks.v2_sdbench import judge_stability as js
from tasks.v2_sdbench import turn_format as tf
from tasks.v2_sdbench.env import load_config, policy_system_prompt
from tasks.v2_sdbench.episode import SYSTEM_PROMPT


def _groups(ra, rb):
    return {"g": [{"r": {"a": x, "b": y}} for x, y in zip(ra, rb)]}


def test_identical_scorings_agree_fully():
    m = js.compare(_groups([0, 0.5, 1], [0, 0.5, 1]), "a", "b")
    assert m["sign_agree"] == 1 and m["sign_flip"] == 0 and m["rank_tau"] == pytest.approx(1)
    assert m["adv_cosine"] == pytest.approx(1) and m["zero_std_change"] == 0


def test_reversed_ranking_flips_signs():
    m = js.compare(_groups([0, 1], [1, 0]), "a", "b")
    assert m["sign_flip"] == 1 and m["rank_tau"] == pytest.approx(-1) and m["adv_cosine"] == pytest.approx(-1)


def test_collapse_to_zero_spread_is_counted():
    m = js.compare(_groups([0, 0.25], [0.25, 0.25]), "a", "b")
    assert m["zero_std_change"] == 1 and m["sign_flip"] == 0 and m["sign_agree"] == 0
    assert m["adv_cosine"] is None


def test_kendall_tau_b_with_ties():
    assert js.kendall_tau_b([0, 0, 1], [0, 0, 1]) == pytest.approx(1)
    assert js.kendall_tau_b([0, 0, 0], [0, 0, 0]) is None


def test_turn_format_default_is_official():
    cfg = load_config(None)
    assert cfg["policy_format"] == "official"
    assert policy_system_prompt(cfg) == SYSTEM_PROMPT
    assert tf.openai_extra("official") == {} and tf.trl_kwargs("official") == {}


def test_turn_format_unknown_variant_rejected(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"policy_format": "loose"}')
    with pytest.raises(ValueError):
        load_config(str(p))


def test_prompt_variant_only_appends():
    s = tf.system_prompt(SYSTEM_PROMPT, "prompt")
    assert s.startswith(SYSTEM_PROMPT.rstrip("\n")) and tf.PROMPT_ADDENDUM in s


@pytest.mark.parametrize("text", [
    "<question>What brings you in?</question>\n<question>Any fever?</question>",
    "Okay.\n\n<test>CBC</test>\n<test>Chest X-ray</test>\n",
    "Based on the results:\n<diagnosis>Whipple disease</diagnosis>",
])
def test_regex_accepts_single_type_turns(text):
    assert re.fullmatch(tf.TURN_REGEX, text)


@pytest.mark.parametrize("text", [
    "<question>Any fever?</question>\n<test>CBC</test>",          # mixed: SD-Bench rejects the turn
    "<test>CBC</test><question>Any fever?</question>",
    "<test>   </test>",                                           # empty tag
    "<diagnosis>A</diagnosis><test>CBC</test>",
])
def test_regex_rejects_mixed_or_empty(text):
    assert not re.fullmatch(tf.TURN_REGEX, text)


def test_constrained_variants_pass_regex_to_both_clients():
    assert tf.openai_extra("constrained") == {"structured_outputs": {"regex": tf.TURN_REGEX}}
    assert tf.trl_kwargs("prompt+constrained") == {"structured_outputs_regex": tf.TURN_REGEX}


# --- leak routes (V2/leak_probe/v2_leak_routes.py) ---
from tasks.v2_sdbench import channel  # noqa: E402


@pytest.mark.parametrize("text", [
    "The fevers resolved within the first week of therapy, and her mental status improved.",
    "She received three cycles of R-CHOP and responded well to treatment.",
    "He underwent a right adrenalectomy and was discharged on hospital day 6.",
    "After starting antibiotics her diarrhea improved.",
    # 2026-11-13 leak-route probe (Qwen gatekeeper answers):
    "The patient's scheduled procedure tomorrow is a random skin biopsy to evaluate the cutaneous findings.",
    "The plan for tomorrow includes a surgical consultation to discuss the upcoming laparoscopic right adrenalectomy.",
    "The care plan includes continuation of the current antibiotic regimen.",
    "He was later managed with phenoxybenzamine and a beta-blocker after adequate alpha-blockade was established.",
])
def test_course_of_care_detected(text):
    assert channel.reveals_course(text)


@pytest.mark.parametrize("text", [
    "She reported a rash with trimethoprim-sulfamethoxazole in the past.",
    "Her medications were lisinopril, levothyroxine, and atorvastatin.",
    "She has no history of chemotherapy or other immunosuppressive therapy.",
    "The fever is intermittent, worse in the evenings, with night sweats.",
    "His father had a thyroid tumor and was treated with thyroid hormone.",
    "He has a history of panic disorder, for which he was treated with sertraline.",
    "She had a cholecystectomy in 2010 and an appendectomy as a child.",
])
def test_history_is_not_course_of_care(text):
    assert not channel.reveals_course(text)
