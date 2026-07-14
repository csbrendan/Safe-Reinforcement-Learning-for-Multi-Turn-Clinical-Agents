"""Claude process-reward GRPO — Claude Opus as an off-policy PROCESS-REWARD JUDGE whose trajectory-level score
augments the analytic training reward: training_reward = analytic_terminal_reward + alpha * claude_process_score.

Phase 0b (this package): VALIDATE the judge before any RL. Construct archetypal candidate trajectories per case
(oracle / premature-commit / under-gather / over-gather / reordered / unsafe-commit), have Claude score them
from student-visible info ONLY, and check whether Claude's process scores rank trajectories the way the analytic
verifier does — especially on the high-risk-safety contrasts. No GPU, no GRPO. Stops at the validation report.
"""
