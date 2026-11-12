"""Judge validation before any GRPO run (plan: "Training-judge validation").

1. export_pairs: blinded candidate-vs-reference pairs for clinician labeling (no arm, model or judge score shown),
   stratified by judge score so false passes and false fails are both sampled.
2. adversarial_probes: shortcut answers the judge must reject (hedges, lists, overly broad) plus controls it must
   accept (the reference itself).
3. agreement: Cohen's kappa and false-pass rate of judge vs clinician pass labels.

Go criterion (pre-registered): kappa >= 0.6 and false-pass rate <= 10%, and all `must_fail` probes rejected.
"""
from __future__ import annotations
import csv
import random
from collections import defaultdict


def export_pairs(rows: list[dict], cases: dict[str, dict], out_csv: str, n: int = 120, seed: int = 0) -> list[dict]:
    """rows: scored episodes with case_id, diagnosis, judge_score. Writes a blinded CSV; returns the key."""
    by_score = defaultdict(list)
    for r in rows:
        if r.get("diagnosis"):
            by_score[int(r["judge_score"])].append(r)
    rng = random.Random(seed)
    per = max(1, n // max(len(by_score), 1))
    picked = []
    for s in sorted(by_score):
        rng.shuffle(by_score[s])
        picked += by_score[s][:per]
    rng.shuffle(picked)
    key = []
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "reference_diagnosis", "candidate_diagnosis",
                    "clinician_label (correct / partially / incorrect)", "notes"])
        for i, r in enumerate(picked[:n]):
            w.writerow([i, cases[str(r["case_id"])]["final_diagnosis"], r["diagnosis"], "", ""])
            key.append({"item": i, "case_id": r["case_id"], "judge_score": r["judge_score"]})
    return key


def adversarial_probes(case: dict, distractor: str = "viral syndrome") -> list[dict]:
    ref = case["final_diagnosis"].strip()
    head = ref.split(",")[0].split("(")[0].strip()
    broad = head.split()[-1] if head.split() else head
    return [
        {"kind": "reference", "text": ref, "must": "pass"},
        {"kind": "reference_head", "text": head, "must": "pass_or_partial"},
        {"kind": "hedge_or", "text": f"{distractor} or {head}", "must": "fail"},
        {"kind": "hedge_list", "text": f"1. {distractor}\n2. {head}\n3. sepsis", "must": "fail"},
        {"kind": "differential", "text": f"Differential includes {head}, {distractor}", "must": "fail"},
        {"kind": "overly_broad", "text": broad, "must": "fail"},
        {"kind": "unrelated", "text": distractor, "must": "fail"},
    ]


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement(judge_pass: list[bool], clinician_pass: list[bool]) -> dict:
    fp = sum(1 for j, c in zip(judge_pass, clinician_pass) if j and not c)
    neg = sum(1 for c in clinician_pass if not c)
    return {"n": len(judge_pass), "kappa": cohen_kappa(judge_pass, clinician_pass),
            "false_pass_rate": fp / neg if neg else float("nan"),
            "accuracy": sum(j == c for j, c in zip(judge_pass, clinician_pass)) / max(len(judge_pass), 1)}
