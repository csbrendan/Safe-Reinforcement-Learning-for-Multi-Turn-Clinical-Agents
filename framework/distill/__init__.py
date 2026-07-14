"""Distilling Step-by-Step (Hsieh et al., 2023) — black-box rationale distillation from a Claude Opus teacher
into the MedGemma student. Sequence-level (word/reasoning) supervision only: the teacher supplies target
ACTION strings and RATIONALE strings; the student is trained with ordinary cross-entropy to generate them.
No teacher logits / soft labels are used (impossible via the API and not required by the method).

Phase 0 (pilot): vet teacher rationale quality, JSON reliability, hidden-state leakage, and teacher
consistency on ~20 diagnostics states BEFORE generating a full dataset or training anything.
"""
