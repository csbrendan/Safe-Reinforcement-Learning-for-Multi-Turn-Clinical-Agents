# Research thesis — when does RL add value over imitation in clinical AI?

## Canonical claim
> **Among *clinically verifiable* tasks, RL provides the greatest value when (1) the demonstrations
> *under-specify* the optimal policy (low demonstrability) AND (2) the model already possesses *sufficient
> latent task support* to be elicited.**
>
> Equivalently: **verifiability enables RL; demonstrability determines how much RL is needed; latent
> capability determines whether RL can actually deliver it.**

It is a *conditional* about when RL helps — not a promise that RL wins. Real clinical practice is routinely
**under-specified** by the demonstrations available to train a model (edge cases, hidden variables, unstated
trade-offs), which is exactly the regime where imitation falls short and RL *may* help.

## The two axes (each measured a priori, before any RL)
- **Demonstration under-specification — the BC-gap.** value(oracle adaptive policy) − value(best
  *demonstrable* policy, i.e. the best fixed checklist / behavior-cloneable policy). Under-specification is
  not "too little data": demonstrations that always order every lab, or always act immediately, can be
  plentiful yet never reveal *when* selective, adaptive information-gathering is correct.
- **Latent support — pass@k.** Whether the base/SFT model can already *sample* reward-bearing (oracle-grade)
  behavior. RL elicits and sharpens capability that exists; it does not conjure capability that is absent.

## Gates and falsifiability
Pre-RL **gates**: run RL only when the BC-gap is meaningfully positive **and** latent support is non-trivial.
The framework is **predictive** (allocates RL compute only where the axes say it can matter) and
**falsifiable**: if RL fails to beat a strong supervised baseline when both gates pass *and the policy
genuinely moves*, the thesis is wrong for that task. A flat RL result is only thesis-relevant once the policy
has actually moved (e.g. KL well above ~1e-3) on cases with non-zero advantage.

## Verifiability is the scope, not the variable
We restrict to verifiable tasks (computable, hack-resistant reward) precisely so RL's contribution is cleanly
attributable. Most medical-RLVR work asks *"can we construct a verifiable reward?"*; we ask *"given one, is RL
even needed, or did SFT already capture it?"*

## Honest-framing guardrails
- Report the demonstrability level (BC-gap) alongside any RL gain; it is the independent variable.
- Don't weaken the SFT baseline to manufacture an RL win; compare against a *strong* warmup.
- A reward gain is only a win if it comes from the **intended behavior** (adaptive gathering), not ask-cost
  exploitation / blind commits / over-escalation — verify with subset + trajectory audits, not mean reward.
- Screening (small-n / short-budget) allocates compute; it does not answer the paper-level question.
