"""Question channel versus test channel: what each request actually revealed.

The problem (V2/sdbench_env_validation.md, section 1): SD-Bench's official gatekeeper prompt quotes any
patient-question answer that exists in the chart, lab, imaging and pathology results included. Its official cost
counts only rounds that contain a <test>. So an agent can ask "What did your bone-marrow biopsy show?" and get the
biopsy result free. With Qwen3-8B as gatekeeper, every decisive study we probed leaked this way, and some
question-channel answers even stated the diagnosis.

That is two problems:
  * Scoring: official cost undercharges. Fixed for training by `CostModel.corrected_cost`.
  * Information leakage: the agent receives results it never ordered. Fixed for training by the guard below,
    which replaces a test result returned to a patient question with a redirect to order it as a test.

Official SD-Bench behaviour (no guard, official cost) stays available and is the primary evaluation condition, so
benchmark numbers stay comparable. The guard is a training-environment deviation and must be reported as one.

The classifier is deterministic and reads the RESULT text (what was revealed), not the request wording, so the
policy can't avoid it by rephrasing the question. Bedside observations (vitals, examination, mental status) are
allowed through the question channel by SD-Bench's own prompt and stay free. Pre-admission ("historic") studies
are also allowed by SD-Bench's chronology rule, so they pass the guard but are counted (`historic_test`).
"""
from __future__ import annotations
import re

QUESTION_PREFIX = "Question for the patient:"
TEST_PREFIX = "Test to order:"

GUARD_TEXT = ("This is a diagnostic test result, which the patient cannot report. "
              "To obtain it, order it as a test using <test> tags.")
DX_GUARD_TEXT = "The patient cannot tell you the diagnosis. Gather findings with questions and tests."

# Diagnostic studies and specimen work. Matched in the result text.
_STUDY = re.compile(
    r"\b(electrocardiogra\w*|ecg|ekg|echocardiogra\w*|computed tomograph\w*|ct (scan|of|angiogra\w*)|ct\b(?=[\s,.]*(chest|abdomen|head|pelvis|scan|showed|revealed|demonstrated))|"
    r"(chest|head|brain|neck|sinus|spine|spinal|abdominal|abdomen|pelvic|pelvis|cardiac|renal|adrenal) ct\b|"
    r"cta\b|mri\b|magnetic resonance|mra\b|radiograph\w*|x-ray|ultrasound\w*|ultrasonograph\w*|sonogra\w*|doppler|"
    r"pet(-ct)?\b(?=[\s,]*(scan|ct|showed|revealed|demonstrated|imaging))|positron|scintigra\w*|angiogra\w*|"
    r"biops\w*|pathology|pathologist\w*|patholog\w* (examination|specimen|report|evaluation|analysis|findings)|histolog\w*|histopatholog\w*|cytolog\w*|immunohistochem\w*|aspirate|smear|"
    r"flow cytometr\w*|culture[sd]?|pcr|polymerase chain|serolog\w*|antibod\w*|assay|electrophoresis|immunofixation|"
    r"urinalysis|urine analysis|specific gravity|cerebrospinal|csf|lumbar puncture|"
    r"endoscop\w*|colonoscop\w*|bronchoscop\w*|eeg|electroencephalogra\w*|electromyogra\w*|emg\b|nerve conduction|"
    r"genetic test\w*|sequencing|pathogenic variant|karyotype|congo red|staining|stain\b)", re.I)

# Laboratory analytes. Matched in the result text.
_ANALYTE = re.compile(
    r"\b(white[- ](blood[- ])?cell count|wbc|hemoglobin|haemoglobin|hematocrit|platelet count|platelets|"
    r"neutrophil|lymphocyte count|sodium|potassium|chloride|bicarbonate|creatinine|urea nitrogen|bun\b|glucose|"
    r"hemoglobin a1c|hba1c|cholesterol|ldl|hdl|triglyceride|aminotransferase|ast\b|alt\b|alkaline phosphatase|"
    r"bilirubin|albumin|total protein|troponin|natriuretic|bnp|tsh\b|thyrotropin|thyroxine|free t4|ferritin|"
    r"lactate dehydrogenase|ldh\b|c-reactive|crp\b|sedimentation rate|esr\b|procalcitonin|lactate|lipase|amylase|"
    r"inr\b|prothrombin|partial thromboplastin|fibrinogen|d-dimer|metanephrine\w*|normetanephrine|catecholamine\w*|"
    r"cortisol|aldosterone|renin|calcitonin|parathyroid hormone|pth\b|calcium level|magnesium|phosphate|"
    r"immunoglobulin|light chain\w*|complement|ana\b|antinuclear|rheumatoid factor|anca|ammonia|"
    r"kidney function|renal function|liver function|blood count)",
    re.I)

# A number followed by a laboratory unit (not vital-sign units such as mm Hg, bpm, °C, kg).
_LAB_UNIT = re.compile(
    r"\d[\d.,]*\s*(×|x)?\s*(10\s*(\^|⁹|\*\*)?\s*\d*\s*(/|per)\s*(l|liter|µl|ul|mm3)|"
    r"(g|mg|µg|ug|ng|pg|mmol|µmol|umol|nmol|pmol|meq|miu|µiu|uiu|iu|u|mu)\s*/\s*(dl|l|ml|µl|ul|24 ?h))", re.I)

# Pre-admission wording in the REQUEST (chronology rule) and a time-ago marker in the RESULT.
_HIST_REQ = re.compile(r"\b(previous|prior|past|ever had|history of|before (this|the) admission|years? ago|"
                       r"last year|in the past)\b", re.I)
_HIST_RES = re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|several)\s+(years?|months?)\s+"
                       r"(ago|earlier|before|previously|prior)\b|\bin (19|20)\d\d\b", re.I)

_REJECT = re.compile(r"(not available in the patient's current record|would you like to order this test|"
                     r"i can only provide specific test results|please request a specific diagnostic test|"
                     r"^\s*results not available for)", re.I)

# "What is your temperature?: 38.6°C"  -> the policy pasted a previous answer after its question.
_PARROT = re.compile(r"\?\s*:\s.*\Z", re.S)


def is_question(name: str) -> bool:
    return name.startswith(QUESTION_PREFIX)


def body(name: str) -> str:
    for p in (QUESTION_PREFIX, TEST_PREFIX):
        if name.startswith(p):
            return name[len(p):].strip()
    return name.strip()


def canonical_request(name: str) -> str:
    """Request text sent to the gatekeeper. Only change: a question with a pasted answer after "?:" is cut back
    to the question. Multi-part questions ("A? B?") are left whole."""
    if is_question(name):
        b = body(name)
        cut = _PARROT.sub("?", b)
        if cut != b:
            return f"{QUESTION_PREFIX} {cut.strip()}"
    return name


# An analyte name counts only with a value or a result word shortly after it ("sodium intake" and "monitored with
# blood tests" are not results; "creatinine 0.9 mg/dL" and "kidney function is normal" are).
_RESULT_WORD = re.compile(r"\d|\b(normal|abnormal|unremarkable|elevated|raised|increased|decreased|low|high|"
                          r"positive|negative|detected|undetectable|within|above|below|reference)\b", re.I)


def _analyte_result(text: str) -> bool:
    return any(_RESULT_WORD.search(text[m.end():m.end() + 60]) for m in _ANALYTE.finditer(text))


def reveals_test_result(result: str) -> bool:
    return bool(_STUDY.search(result) or _LAB_UNIT.search(result) or _analyte_result(result))


# Words that appear in reference diagnoses but are not specific to one (anatomy, qualifiers, generic nouns).
# Everything else in the reference (lymphoma, pheochromocytoma, amyloidosis, whipple, teratoma, ...) is a key term.
_GENERIC = set("""
a an the of with and or in on to for by from due secondary primary acute chronic left right bilateral syndrome
disease disorder type associated involvement systemic setting arising consistent confirmed multiple variant
pathogenic germline mature cystic receptor infection tumor tumour mass lesion related induced severe mild
positive negative testing test chain light small large cell cells tissue soft soft-tissue central nervous system
nervous-system central-nervous-system cutaneous skin brain lung liver kidney heart cardiac renal hepatic pulmonary
intestinal small-intestinal intestine ovarian ovary bone marrow blood fluid cerebrospinal cerebrospinal-fluid gland
adrenal medulla thyroid immunophenotype germinal center germinal-center non-germinal-center polymerase reaction
chain-reaction polymerase-chain-reaction stage grade status post history partial complete early late recurrent
endocrine metabolic autoimmune inflammatory infectious malignant benign neoplasm
""".split())


def dx_key_terms(reference: str) -> set[str]:
    words = re.findall(r"[a-z][a-z\-]{3,}", (reference or "").lower())
    return {w for w in words if w not in _GENERIC and not all(p in _GENERIC for p in w.split("-"))}


def mentions_dx(text: str, key_terms: set[str]) -> list[str]:
    """Key terms of the reference found in `text`. Long terms also match on a 7-letter stem
    (amyloidosis ~ amyloid, whipple ~ whipplei)."""
    words = set(re.findall(r"[a-z][a-z\-]{3,}", (text or "").lower()))
    hits = []
    for k in key_terms:
        stem = k[:7] if len(k) >= 8 else None
        if k in words or (stem and any(w.startswith(stem) for w in words)):
            hits.append(k)
    return sorted(hits)


def classify(request: str, result: str, dx_terms: set[str] | None = None) -> str:
    """One of: rejected | dx_leak | test_result | historic_test | info.

    For a <test> request, "test_result" just means the test returned something (it's charged as a test anyway).
    For a question:
      dx_leak      the answer names the reference diagnosis (e.g. "What was your final diagnosis?"). Withheld by
                   the guard; never charged (it isn't a test, it's the answer key).
      test_result  a lab/imaging/pathology result; withheld by the guard, charged by the corrected cost.
      info         history, symptoms or bedside observation, which SD-Bench allows for free.
    """
    r = result or ""
    if _REJECT.search(r):
        return "rejected"
    if not is_question(request):
        return "test_result"
    if dx_terms and mentions_dx(r, dx_terms):
        return "dx_leak"
    if not reveals_test_result(r):
        return "info"
    if _HIST_REQ.search(body(request)) and _HIST_RES.search(r):
        return "historic_test"
    return "test_result"


def split_compound(test_body: str) -> list[str]:
    """'CBC, CMP; troponin' -> three tests, so a bundle can't be priced as one item. Splits on commas, semicolons
    and '+'. Parentheses protect their contents ('CSF analysis (cell count, protein)' stays one test)."""
    parts, depth, cur = [], 0, ""
    for ch in test_body:
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and ch in ",;+":
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    out = [p.strip(" .") for p in parts if p.strip(" .")]
    return out or [test_body.strip()]


def question_subject(q: str) -> str:
    """Pricing name for a test obtained through a question: 'What did your fat pad biopsy show?' -> 'fat pad
    biopsy'. Only used to look up a price; the price cache is keyed on this string."""
    s = body(q)
    s = _PARROT.sub("", s).strip()
    s = re.sub(r"\?.*\Z", "", s, flags=re.S)
    s = re.sub(r"^(what|which|how|can you|could you|do you|did you|have you|are|is|were|was)\b\s*", "", s, flags=re.I)
    s = re.sub(r"^(is|are|was|were|did|do|does|tell me|describe|share)\b\s*", "", s, flags=re.I)
    s = re.sub(r"^(the patient's|your|his|her|their|the)\s+", "", s, flags=re.I)
    s = re.sub(r"(\s+(show|showed|reveal|revealed|result|results|level|levels|reading|value|values))+\s*$", "", s,
               flags=re.I)
    return s.strip(" .:") or body(q)
