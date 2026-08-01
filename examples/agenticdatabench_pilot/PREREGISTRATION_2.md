# Pre-registration: input previews and verified ensemble

Fixed before running, per the rule that cost 120 runs to learn.

## Set

40 tasks, stratified proportionally by baseline score band (14 zero, 18
partial, 8 perfect) and spread across all 15 domains, seed 20260801.
Baseline mean on this set is 0.4404 against 0.4349 on all 246, so it is
representative.

## Arms

All at temperature 0, max_turns 25, timeout 900, MiniMax M3.

- **baseline** — reuse of the completed 246-task run, restricted to these 40.
- **previews** — deterministic schema preview (columns, dtypes, 3 rows) of
  every tabular input injected into the prompt by the harness.
- **previews+verify** — previews, plus two blind solves into separate
  directories, file-level agreement, and a third reconciliation solve on
  disagreement.

## Hypotheses

1. Previews cut input tokens. Trace analysis of the 246-task run: 3.72
   leading turns per task do nothing but discover columns and dtypes (30% of
   all turns), and input is 96.2% of token volume because the transcript is
   resent every turn. Expect a token reduction; score effect unknown and not
   the primary claim.
2. Verify raises score by resolving run-to-run flakiness. Four identical runs
   on 10 tasks gave mean 0.520 against an oracle best-of-4 of 0.700, so
   +0.180 is recoverable in principle; a real selector captures part of it.

## Decision rule

Paired difference against baseline on the same 40 tasks, 95% CI from the
paired SE.

- **Adopt** if the CI excludes zero and the mean is positive.
- **Adopt previews on cost alone** if tokens drop at least 20% and the score
  CI does not exclude zero in the negative direction (i.e. no evidence of harm).
- **Reject** if the mean is negative and the CI excludes zero.
- **Not measured** otherwise, and say so rather than reporting a direction.

## Power, stated in advance

Paired per-task SD on this benchmark is about 0.29, so at n=40 the paired SE
is roughly 0.046 and the detectable effect is about 0.09. The oracle headroom
is 0.18; if verify captures more than half of it the design will see it,
otherwise the honest outcome is "not measured" and the decision moves to the
full 246 where SE is about 0.019.

No post-hoc subgroup will be used to rescue a null result.
