#!/usr/bin/env python3
"""Bounded AgentClinic-MedQA smoke harness (integration test, NOT a scientific evaluation).

Reproduces AgentClinic's protocol (prompts copied from agentclinic.py at commit b6fbe22) without its
openai==0.28 pin, silent 30x retries, or hard-coded GPT models. Doctor and environment roles can point at
different OpenAI-compatible servers. Doctor sees only Objective_for_Doctor plus returned dialogue; each returned
message is audited for verbatim leakage of the correct diagnosis.
"""
import argparse, json, re, time, urllib.request
from pathlib import Path

GEMMA_STOP, QWEN_STOP = [1, 106], [151643, 151645]

def chat(url, model, system, user, max_tokens, stop_ids=None, stop=None, temperature=0.05, thinking_off=False,
         skip_special=True, timeout=180):
    body = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}] + (user if isinstance(user, list) else [{"role": "user", "content": user}]),
            "skip_special_tokens": skip_special, "seed": 1}
    if stop_ids: body["stop_token_ids"] = stop_ids
    if stop: body["stop"] = stop
    if thinking_off: body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    c = d["choices"][0]
    return {"text": c["message"]["content"] or "", "finish": c["finish_reason"], "stop_reason": c.get("stop_reason"),
            "completion_tokens": d["usage"]["completion_tokens"], "prompt_tokens": d["usage"]["prompt_tokens"],
            "secs": round(time.time() - t, 2)}

def doctor_sys(max_infs, infs, presentation):
    return ("You are a doctor named Dr. Agent who only responds in the form of dialogue. You are inspecting a patient who you will ask questions in order to understand their disease. You are only allowed to ask {} questions total before you must make a decision. You have asked {} questions so far. You can request test results using the format \"REQUEST TEST: [test]\". For example, \"REQUEST TEST: Chest_X-Ray\". Your dialogue will only be 1-3 sentences in length. Once you have decided to make a diagnosis please type \"DIAGNOSIS READY: [diagnosis here]\"".format(max_infs, infs)
            + "\n\nBelow is all of the information you have. {}. \n\n Remember, you must discover their disease by asking them questions. You are also able to provide exams.".format(presentation))

def patient_sys(symptoms):
    return ("You are a patient in a clinic who only responds in the form of dialogue. You are being inspected by a doctor who will ask you questions and will perform exams on you in order to understand your disease. Your answer will only be 1-3 sentences in length."
            + "\n\nBelow is all of your information. {}. \n\n Remember, you must not reveal your disease explicitly but may only convey the symptoms you have in the form of dialogue if you are asked.".format(symptoms))

def meas_sys(info):
    return ("You are an measurement reader who responds with medical test results. Please respond in the format \"RESULTS: [results here]\""
            + "\n\nBelow is all of the information you have. {}. \n\n If the requested results are not in your data then you can respond with NORMAL READINGS.".format(info))

MOD_SYS = "You are responsible for determining if the corrent diagnosis and the doctor diagnosis are the same disease. Please respond only with Yes or No. Nothing else."

def clean_env(t):
    """Strip role labels added by a non-instruct auxiliary model, so the doctor sees only the reply."""
    return re.sub(r"^\s*((Assistant|Patient|Measurement( reader)?|Moderator)\s*:\s*)+", "", t).strip()

def norm(s): return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()
def leaks(dx, text):
    return " ".join(norm(dx)) in " ".join(norm(text))

def load_cases(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def leak_cases(cases):
    out = []
    for i, c in enumerate(cases):
        o = c["OSCE_Examination"]
        if leaks(o["Correct_Diagnosis"], json.dumps(o["Test_Results"])): out.append(i)
    return out

def run_case(idx, case, a):
    o = case["OSCE_Examination"]; dx = o["Correct_Diagnosis"]
    info = dict(o["Physical_Examination_Findings"]); info["tests"] = o["Test_Results"]
    d_stop = GEMMA_STOP if a.stop_mode != "none" else None
    # Doctor text legitimately starts "...\n\nDoctor: <question>" because AgentClinic's prompt ends in "Doctor: ",
    # so only other speakers' labels stop the doctor.
    role_stop = ["Patient:", "Assistant:", "\nUser:", "\nMeasurement"] if a.stop_mode == "tokens+roles" else None
    env_stop = ["Doctor:", "\nAssistant:", "\nUser:"] if a.stop_mode == "tokens+roles" else None
    e_stop = (QWEN_STOP if a.aux_family == "qwen" else GEMMA_STOP) if a.stop_mode != "none" else None
    d_hist = p_hist = m_hist = ""; pi = ""; msgs = []; turns = []; t0 = time.time(); outcome = "turn_limit"; result = {}
    for t in range(a.max_turns):
        if time.time() - t0 > a.case_timeout: outcome = "case_timeout"; break
        if t == a.max_turns - 1: pi += "This is the final question. Please provide a diagnosis.\n"
        if a.doctor_format == "chat":  # real alternating turns; same system prompt and protocol tokens as AgentClinic
            msgs.append({"role": "user", "content": pi or "The patient has arrived. Please begin."})
            d_in = msgs
        else:                          # AgentClinic's own single-user-message history string
            d_in = "\nHere is a history of your dialogue: " + d_hist + "\n Here was the patient response: " + pi + "Now please continue your dialogue\nDoctor: "
        r = chat(a.doctor_url, a.doctor_model, doctor_sys(a.max_turns, t, o["Objective_for_Doctor"]), d_in,
                 a.doctor_max_tokens, d_stop, role_stop, skip_special=not a.show_special)
        txt = r["text"].strip(); d_hist += pi + "\n\n" + txt + "\n\n"; msgs.append({"role": "assistant", "content": txt or "..."})
        rec = {"case": idx, "turn": t, "role": "doctor", **r, "text": txt,
               "role_break": bool(re.search(r"(^|\s)(Patient|Assistant):", txt)),
               "special_leak": bool(re.search(r"<(start|end)_of_turn>|^model\b|\bmodel Doctor", txt)),
               "repeat_of_prev": txt in [x["text"] for x in turns if x["role"] == "doctor"],
               "action": "diagnosis" if "DIAGNOSIS READY" in txt else "test" if "REQUEST TEST" in txt else "question"}
        turns.append(rec)
        if rec["action"] == "diagnosis":
            m = re.search(r"DIAGNOSIS READY:\s*\[?(.*?)\]?\s*$", txt, re.S); stated = (m.group(1) if m else txt).strip()
            j = chat(a.aux_url, a.aux_model, MOD_SYS, "\nHere is the correct diagnosis: " + dx + "\n Here was the doctor dialogue: " + txt + "\nAre these the same?",
                     16, e_stop, None, thinking_off=a.aux_family == "qwen")
            result = {"stated": stated[:200], "judge_raw": j["text"].strip(), "judge_yes": clean_env(j["text"]).lower().startswith("yes"), "judge_secs": j["secs"]}
            outcome = "diagnosed"; break
        if rec["action"] == "test":
            er = chat(a.aux_url, a.aux_model, meas_sys(info), "\nHere is a history of the dialogue: " + m_hist + "\n Here was the doctor measurement request: " + txt,
                      a.env_max_tokens, e_stop, env_stop, thinking_off=a.aux_family == "qwen")
            pi = clean_env(er["text"]); m_hist += txt + "\n\n" + pi + "\n\n"; p_hist += pi + "\n\n"; who = "measurement"
        else:
            er = chat(a.aux_url, a.aux_model, patient_sys(o["Patient_Actor"]), "\nHere is a history of your dialogue: " + p_hist + "\n Here was the doctor response: " + txt + "Now please continue your dialogue\nPatient: ",
                      a.env_max_tokens, e_stop, env_stop, thinking_off=a.aux_family == "qwen")
            pi = clean_env(er["text"]); p_hist += txt + "\n\n" + pi + "\n\n"; m_hist += pi + "\n\n"; who = "patient"
        turns.append({"case": idx, "turn": t, "role": who, **er, "text": pi, "dx_leak": leaks(dx, pi)})
    return {"case": idx, "outcome": outcome, "correct_dx": dx, "wall_secs": round(time.time() - t0, 1),
            "n_doctor_turns": sum(x["role"] == "doctor" for x in turns), "n_tests": sum(x.get("action") == "test" for x in turns),
            "n_questions": sum(x.get("action") == "question" for x in turns), "role_breaks": sum(bool(x.get("role_break")) for x in turns), "doctor_repeats": sum(bool(x.get("repeat_of_prev")) for x in turns),
            "doctor_hit_token_cap": sum(x["role"] == "doctor" and x["finish"] == "length" for x in turns),
            "doctor_tokens_mean": round(sum(x["completion_tokens"] for x in turns if x["role"] == "doctor") / max(1, sum(x["role"] == "doctor" for x in turns)), 1),
            "dx_leaks_to_doctor": sum(bool(x.get("dx_leak")) for x in turns), **result}, turns

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(Path.home() / "AgentClinic/agentclinic_medqa_extended.jsonl"))
    p.add_argument("--doctor-url", default="http://127.0.0.1:8100/v1"); p.add_argument("--doctor-model", default="medgemma4b")
    p.add_argument("--aux-url", default="http://192.168.100.2:8101/v1"); p.add_argument("--aux-model", default="aux")
    p.add_argument("--aux-family", default="qwen", choices=["qwen", "gemma"])
    p.add_argument("--cases", type=int, nargs="+", default=None); p.add_argument("--n-clean", type=int, default=3)
    p.add_argument("--max-turns", type=int, default=12); p.add_argument("--doctor-max-tokens", type=int, default=200)
    p.add_argument("--env-max-tokens", type=int, default=120); p.add_argument("--case-timeout", type=int, default=600)
    p.add_argument("--stop-mode", default="tokens", choices=["none", "tokens", "tokens+roles"])
    p.add_argument("--doctor-format", default="agentclinic", choices=["agentclinic", "chat"]); p.add_argument("--show-special", action="store_true"); p.add_argument("--out", required=True)
    a = p.parse_args()
    cases = load_cases(a.data); bad = set(leak_cases(cases))
    ids = a.cases if a.cases is not None else [i for i in range(len(cases)) if i not in bad][: a.n_clean]
    print(json.dumps({"total_cases": len(cases), "leaky_case_ids": sorted(bad), "n_leaky": len(bad), "selected": ids, "args": vars(a)}))
    with open(a.out, "w") as f:
        for i in ids:
            s, tr = run_case(i, cases[i], a)
            for x in tr: f.write(json.dumps(x) + "\n")
            f.write(json.dumps({"summary": s}) + "\n"); f.flush(); print(json.dumps(s))
