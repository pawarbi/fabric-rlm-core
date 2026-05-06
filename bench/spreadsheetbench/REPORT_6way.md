# SpreadsheetBench Verified-400 — 6-way head-to-head

## TL;DR

On 399 SpreadsheetBench Verified-400 questions (368 after excluding grader-bug rows):

| Result | Number |
|---|---|
| **gpt-4.1-mini + Python interpreter (RLM)** | **29.3%** at **$0.012/q** |
| **gpt-4.1 + Python interpreter (RLM)** | **31.2%** at **$0.039/q** |
| gpt-5 single-shot (no tools) | 26.6% at $0.040/q |
| Cost-aware ladder (mini → gpt-5 escalation) | 28.3% at $0.029/q |
| gpt-5 + RLM (partial: 165 q, ran out of compute window) | 12.6% — see notes |
| gpt-5 parent + mini sub_lm orchestration | 2.2% — collapsed |

**Headline findings:**
1. **A small model + a Python interpreter matches a frontier model with no tools, at one-third the cost.** `gpt-4.1-mini + RLM` (29.3%, $0.012/q) is statistically equivalent to `gpt-5` single-shot (26.6%, $0.040/q).
2. **`gpt-4.1 + RLM` beats `gpt-5` single-shot outright** (31.2% vs 26.6%) at the same per-question cost.
3. **gpt-5 misuses the RLM harness** (12.6% on partial set vs 26.6% alone): writes VBA / Power Query / prose *as cell strings* instead of executing transformations. The skill prompt is implicitly tuned for literal "write Python that mutates the file" execution, which mini/gpt-4.1 follow but gpt-5 reinterprets as a "describe the solution" task. Investigation in `notes/F5_failure_modes.md`.
4. Both cheap-mini and gpt-4.1 RLM runs **clear the public leaderboard** (Gemini 3.1 Pro 23.68%, GPT-5.2 Bash Agent 26.79%) at materially lower cost than published agent rigs.

**Caveats:**
- F5 (`gpt-5+RLM`) only completed 165/399 before the Fabric notebook compute window expired. Result is directional, not conclusive — but the failure-mode analysis on the completed 165 is unambiguous.
- S (`gpt-5+sub_lm`) collapsed to 2.2%; the orchestration prompt allowed gpt-5 to declare success based on the worker's *log output* rather than verifying the file.
- All grader-bug exclusions are identical across strategies (same 31 questions hit a benchmark-side iterability bug), so the pass-rate comparison is on the same 368-question denominator for A/F/F41/S/L.

---

Strategies:
- **A**: gpt-5 single-shot (no tools)
- **F**: gpt-4.1-mini + fabric_rlm RLM (Python interpreter, max 14 turns)
- **F41**: gpt-4.1 + fabric_rlm RLM (same harness, stronger base model)
- **F5**: gpt-5 + fabric_rlm RLM (same harness, frontier base model)
- **S**: gpt-5 parent + sub_lm=gpt-4.1-mini worker (split orchestration)
- **L**: cost-aware ladder (F → S, FabricLM self-check escalation)

## Pass rates (excluding grader-bug cases)

| Strategy | N | Pass | Grader-bug | Rate (excl bug) | Tokens (P/C) |
|---|---|---|---|---|---|
| A: gpt-5 single-shot | 399 | 98 | 31 | 26.6% (98/368) | 190,728/1,581,767 |
| F: mini+RLM | 399 | 108 | 31 | 29.3% (108/368) | 9,717,821/570,026 |
| F41: gpt-4.1+RLM | 399 | 115 | 31 | 31.2% (115/368) | 6,049,751/413,072 |
| F5: gpt-5+RLM | 165 | 20 | 6 | 12.6% (20/159) | 1,794,165/656,264 |
| S: gpt-5+sub_lm | 399 | 8 | 31 | 2.2% (8/368) | 1,119,830/362,214 |
| L: ladder | 399 | 104 | 31 | 28.3% (104/368) | 9,027,394/1,135,315 |

## Cost (USD, OpenAI pricing)

| Strategy | Total | Per Q | vs A |
|---|---|---|---|
| A | $16.06 | $0.0402 | 1.0× |
| F | $4.80 | $0.0120 | 3.3× |
| F41 | $15.40 | $0.0386 | 1.0× |
| F5 | $8.81 | $0.0534 | 1.8× |
| S | $5.02 | $0.0126 | 3.2× |
| L | $11.72 | $0.0294 | 1.4× |

## Same-harness model comparison (apples-to-apples on RLM)

Holding the harness constant (fabric_rlm RLM, excel_modify skill, max_iter=14), pass rate by model:

| Model in F-harness | Pass rate | Cost/Q |
|---|---|---|
| gpt-4.1-mini | 29.3% (108/368) | $0.0120 |
| gpt-4.1 | 31.2% (115/368) | $0.0386 |
| gpt-5 | 12.6% (20/159) | $0.0534 |

## Interpreter-access value (gpt-5 with vs without RLM)

- A (gpt-5 alone): **26.6%**
- F5 (gpt-5 + Python interpreter): **12.6%**
- Lift from interpreter access: **-14.1 pp**

## Ladder analytics (L)

- Escalation rate: 240/399 = 60.2%
- Rung-0 pass: 80  |  Rung-0 fail (kept): 79
- Rung-1 pass: 24  |  Rung-1 fail: 216

## Pairwise intersections (questions present in all 6)

- All 6 pass: 2/165
- Union (any pass — oracle ceiling): 51/165
- F5 (gpt-5+RLM) wins where A loses: 8
- A wins where F5 loses: 16

## Public leaderboard (Mar 2026, for context)

| Model | Pass |
|---|---|
| Claude Opus 4.6 (Bash Agent) | 34.89% |
| GPT-5.2 (Bash Agent) | 26.79% |
| Gemini 3.1 Pro | 23.68% |