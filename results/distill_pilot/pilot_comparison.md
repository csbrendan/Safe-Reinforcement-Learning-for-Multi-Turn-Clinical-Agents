# Phase-0 pilot — Claude teacher vs analytic oracle (diagnostics, student-visible only)

Claude sees ONLY the visible transcript + action schema + rubric. Oracle is computed privileged-side for comparison only. `MedGemma action` is not populated in this CPU pilot (no student rollout) — shown as `—`.

| # | case | revealed so far | MedGemma | Claude action | oracle action | agree | risk/conf | leak |
|---|---|---|---|---|---|---|---|---|
| 0 | dx-0 chest_pain | (none) | — | ask:ecg_ischemia | ask:troponin_high | ~type | moderate/0.7 |  |
| 1 | dx-0 chest_pain | ecg_ischemia | — | ask:troponin_high | ask:troponin_high | ✓ | moderate/0.7 |  |
| 2 | dx-0 chest_pain | ecg_ischemia, troponin_high | — | commit:admit | ask:radiation_arm | ✗ | high/0.8 |  |
| 3 | dx-1 abd_pain | (none) | — | ask:abd_rigidity | ask:abd_rigidity | ✓ | moderate/0.6 |  |
| 4 | dx-1 abd_pain | abd_rigidity | — | ask:rebound_tender | ask:fever_high | ~type | moderate/0.6 |  |
| 5 | dx-1 abd_pain | abd_rigidity, rebound_tender | — | ask:lactate_high | ask:fever_high | ~type | moderate/0.6 |  |
| 6 | dx-1 abd_pain | abd_rigidity, rebound_tender, lactate_high | — | ask:ct_free_air | ask:fever_high | ~type | high/0.62 |  |
| 7 | dx-1 abd_pain | abd_rigidity, rebound_tender, lactate_high, ct_free_air | — | ask:fever_high | ask:fever_high | ✓ | moderate/0.6 |  |
| 8 | dx-1 abd_pain | abd_rigidity, rebound_tender, lactate_high, ct_free_air, fever_high | — | ask:wbc_high | ask:wbc_high | ✓ | moderate/0.6 |  |
| 9 | dx-1 abd_pain | abd_rigidity, rebound_tender, lactate_high, ct_free_air, fever_high, wbc_high | — | commit:admit | commit:symptomatic_tx | ~type | moderate/0.7 |  |
| 10 | dx-2 chest_pain | (none) | — | ask:ecg_ischemia | ask:troponin_high | ~type | moderate/0.7 |  |
| 11 | dx-2 chest_pain | ecg_ischemia | — | ask:troponin_high | ask:troponin_high | ✓ | high/0.7 |  |
| 12 | dx-2 chest_pain | ecg_ischemia, troponin_high | — | commit:emergent_tx | ask:dyspnea_severe | ✗ | high/0.92 |  |
| 13 | dx-3 abd_pain | (none) | — | ask:abd_rigidity | ask:abd_rigidity | ✓ | moderate/0.6 |  |
| 14 | dx-3 abd_pain | abd_rigidity | — | ask:ct_free_air | ask:rebound_tender | ~type | high/0.7 |  |
| 15 | dx-3 abd_pain | abd_rigidity, ct_free_air | — | ask:lactate_high | ask:rebound_tender | ~type | high/0.6 |  |
| 16 | dx-3 abd_pain | abd_rigidity, ct_free_air, lactate_high | — | ask:rebound_tender | ask:rebound_tender | ✓ | moderate/0.6 |  |
| 17 | dx-3 abd_pain | abd_rigidity, ct_free_air, lactate_high, rebound_tender | — | ask:guarding | ask:guarding | ✓ | moderate/0.6 |  |
| 18 | dx-3 abd_pain | abd_rigidity, ct_free_air, lactate_high, rebound_tender, guarding | — | commit:observe | commit:discharge | ~type | moderate/0.68 |  |
| 19 | dx-16 chest_pain | (none) | — | ask:ecg_ischemia | ask:troponin_high | ~type | moderate/0.6 |  |
| 20 | dx-16 chest_pain | ecg_ischemia | — | ask:troponin_high | ask:troponin_high | ✓ | high/0.8 |  |
| 21 | dx-16 chest_pain | ecg_ischemia, troponin_high | — | commit:admit | ask:pleuritic_pain | ✗ | high/0.8 |  |
| 22 | dx-31 abd_pain | (none) | — | ask:abd_rigidity | ask:abd_rigidity | ✓ | moderate/0.6 |  |
| 23 | dx-31 abd_pain | abd_rigidity | — | ask:ct_free_air | ask:rebound_tender | ~type | high/0.7 |  |
| 24 | dx-31 abd_pain | abd_rigidity, ct_free_air | — | ask:lactate_high | ask:rebound_tender | ~type | high/0.6 |  |
| 25 | dx-31 abd_pain | abd_rigidity, ct_free_air, lactate_high | — | commit:emergent_tx | ask:rebound_tender | ✗ | high/0.78 |  |

## Rationales (full text)

**[0] dx-0 chest_pain** — Claude `ask:ecg_ischemia` vs oracle `ask:troponin_high`  
  revealed: (none)  
  rationale: Acute chest pain requires immediate ECG to evaluate for ischemia, which is the most decision-relevant initial finding for triage and disposition.

**[1] dx-0 chest_pain** — Claude `ask:troponin_high` vs oracle `ask:troponin_high`  
  revealed: ecg_ischemia  
  rationale: With chest pain and a non-ischemic ECG, a troponin is essential to rule out myocardial injury before considering disposition.

**[2] dx-0 chest_pain** — Claude `commit:admit` vs oracle `ask:radiation_arm`  
  revealed: ecg_ischemia, troponin_high  
  rationale: Elevated troponin in the setting of acute chest pain indicates myocardial injury requiring inpatient monitoring and further cardiac workup, even without ECG ischemia.

**[3] dx-1 abd_pain** — Claude `ask:abd_rigidity` vs oracle `ask:abd_rigidity`  
  revealed: (none)  
  rationale: Acute abdominal pain with no findings yet; assessing for peritoneal signs like rigidity is the highest-yield next step to detect a surgical abdomen.

**[4] dx-1 abd_pain** — Claude `ask:rebound_tender` vs oracle `ask:fever_high`  
  revealed: abd_rigidity  
  rationale: Acute abdominal pain with a soft abdomen is reassuring, but I need to check for peritoneal signs like rebound tenderness before deciding on disposition.

**[5] dx-1 abd_pain** — Claude `ask:lactate_high` vs oracle `ask:fever_high`  
  revealed: abd_rigidity, rebound_tender  
  rationale: Abdomen is soft without rebound, which is reassuring, but a lactate level would help detect occult bowel ischemia or systemic hypoperfusion not evident on exam before deciding disposition.

**[6] dx-1 abd_pain** — Claude `ask:ct_free_air` vs oracle `ask:fever_high`  
  revealed: abd_rigidity, rebound_tender, lactate_high  
  rationale: Elevated lactate despite a soft abdomen without rebound raises concern for occult ischemia or perforation; CT for free air/mesenteric findings is decision-relevant before disposition.

**[7] dx-1 abd_pain** — Claude `ask:fever_high` vs oracle `ask:fever_high`  
  revealed: abd_rigidity, rebound_tender, lactate_high, ct_free_air  
  rationale: Elevated lactate with a benign abdominal exam (soft, no rebound) and no free air is concerning for occult ischemia or early sepsis; checking for fever helps distinguish an infectious/systemic process needing admission.

**[8] dx-1 abd_pain** — Claude `ask:wbc_high` vs oracle `ask:wbc_high`  
  revealed: abd_rigidity, rebound_tender, lactate_high, ct_free_air, fever_high  
  rationale: Elevated lactate raises concern for occult ischemia or early sepsis despite benign exam and no free air; WBC helps stratify inflammatory/infectious severity before disposition.

**[9] dx-1 abd_pain** — Claude `commit:admit` vs oracle `commit:symptomatic_tx`  
  revealed: abd_rigidity, rebound_tender, lactate_high, ct_free_air, fever_high, wbc_high  
  rationale: Soft abdomen with no rebound and no free air on CT argues against surgical peritonitis, but elevated lactate and elevated WBC signal a potentially serious evolving process warranting inpatient monitoring and workup.

**[10] dx-2 chest_pain** — Claude `ask:ecg_ischemia` vs oracle `ask:troponin_high`  
  revealed: (none)  
  rationale: Acute chest pain requires immediate ECG to evaluate for ischemia, the highest-yield and time-critical first test.

**[11] dx-2 chest_pain** — Claude `ask:troponin_high` vs oracle `ask:troponin_high`  
  revealed: ecg_ischemia  
  rationale: ECG ischemia is concerning for acute coronary syndrome; troponin will confirm myocardial injury and guide between admission and emergent intervention.

**[12] dx-2 chest_pain** — Claude `commit:emergent_tx` vs oracle `ask:dyspnea_severe`  
  revealed: ecg_ischemia, troponin_high  
  rationale: ECG shows ischemia and troponin is elevated, together indicating acute coronary syndrome (likely STEMI/NSTEMI) requiring immediate intervention.

**[13] dx-3 abd_pain** — Claude `ask:abd_rigidity` vs oracle `ask:abd_rigidity`  
  revealed: (none)  
  rationale: Acute abdominal pain with no findings yet; checking for abdominal rigidity is a high-yield initial step to detect peritonitis or surgical abdomen.

**[14] dx-3 abd_pain** — Claude `ask:ct_free_air` vs oracle `ask:rebound_tender`  
  revealed: abd_rigidity  
  rationale: Abdominal rigidity raises concern for peritonitis or a perforated viscus; CT for free air would confirm perforation and dictate whether emergent surgical intervention is needed.

**[15] dx-3 abd_pain** — Claude `ask:lactate_high` vs oracle `ask:rebound_tender`  
  revealed: abd_rigidity, ct_free_air  
  rationale: Abdominal rigidity suggests peritonitis, but CT shows no free air; lactate would help detect bowel ischemia or systemic hypoperfusion driving an acute abdomen.

**[16] dx-3 abd_pain** — Claude `ask:rebound_tender` vs oracle `ask:rebound_tender`  
  revealed: abd_rigidity, ct_free_air, lactate_high  
  rationale: Abdominal rigidity suggests peritoneal irritation, but no free air on CT and normal lactate are reassuring against perforation or ischemia; assessing for rebound tenderness clarifies whether true peritonitis is present.

**[17] dx-3 abd_pain** — Claude `ask:guarding` vs oracle `ask:guarding`  
  revealed: abd_rigidity, ct_free_air, lactate_high, rebound_tender  
  rationale: Abdominal rigidity is concerning, but CT shows no free air, lactate is normal, and no rebound, which lowers likelihood of surgical catastrophe; checking for guarding helps clarify whether true peritoneal signs are present.

**[18] dx-3 abd_pain** — Claude `commit:observe` vs oracle `commit:discharge`  
  revealed: abd_rigidity, ct_free_air, lactate_high, rebound_tender, guarding  
  rationale: Abdominal rigidity is present but CT shows no free air, lactate is normal, and there is no rebound or guarding, making surgical catastrophe unlikely; observation allows monitoring for evolving signs.

**[19] dx-16 chest_pain** — Claude `ask:ecg_ischemia` vs oracle `ask:troponin_high`  
  revealed: (none)  
  rationale: Acute chest pain requires immediate risk stratification for acute coronary syndrome; an ECG is the fastest, highest-yield initial test to detect ischemia.

**[20] dx-16 chest_pain** — Claude `ask:troponin_high` vs oracle `ask:troponin_high`  
  revealed: ecg_ischemia  
  rationale: ECG shows ischemia, raising concern for acute coronary syndrome; troponin is essential to confirm myocardial injury and guide whether emergent intervention versus admission is needed.

**[21] dx-16 chest_pain** — Claude `commit:admit` vs oracle `ask:pleuritic_pain`  
  revealed: ecg_ischemia, troponin_high  
  rationale: ECG demonstrates ischemia, which alone indicates acute coronary syndrome requiring admission and cardiac monitoring, even though the initial troponin is normal (early presentation may precede troponin rise).

**[22] dx-31 abd_pain** — Claude `ask:abd_rigidity` vs oracle `ask:abd_rigidity`  
  revealed: (none)  
  rationale: Acute abdominal pain with no findings yet; checking for abdominal rigidity is a key exam step to detect peritonitis or surgical abdomen and guide urgency.

**[23] dx-31 abd_pain** — Claude `ask:ct_free_air` vs oracle `ask:rebound_tender`  
  revealed: abd_rigidity  
  rationale: Abdominal rigidity raises concern for peritonitis, possibly from a perforated viscus; CT for free air would confirm a surgical emergency and guide disposition.

**[24] dx-31 abd_pain** — Claude `ask:lactate_high` vs oracle `ask:rebound_tender`  
  revealed: abd_rigidity, ct_free_air  
  rationale: Abdominal rigidity suggests peritonitis, but CT shows no free air; an elevated lactate would help identify bowel ischemia or evolving sepsis requiring emergent intervention.

**[25] dx-31 abd_pain** — Claude `commit:emergent_tx` vs oracle `ask:rebound_tender`  
  revealed: abd_rigidity, ct_free_air, lactate_high  
  rationale: Abdominal rigidity with elevated lactate suggests peritonitis and possible ischemia (e.g., mesenteric); absence of free air on CT does not exclude a surgical emergency, warranting emergent evaluation and treatment.
