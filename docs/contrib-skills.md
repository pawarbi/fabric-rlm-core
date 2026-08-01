# Contributed skills

Skills that are not part of the installed package. They live here because they are
narrower than the bundled playbooks, or because the evidence behind them is thinner
than a default install should carry.

Nothing in this directory ships with `pip install fabric-rlm`. To use one, copy the
file somewhere the notebook can read and point a `SkillLoader` at it:

```python
from fabric_rlm import RLM, File, SkillLoader

loader = SkillLoader(skill_dir="contrib-skills")

result = RLM.task(
    task="What was Boeing's FY2022 core operating loss? Report the figure with its sign.",
    inputs={"filing": File("BOEING_2022_10K.pdf")},
    outputs=["answer"],
    skills=["pdf_document_analysis", "financial_documents"],
    skill_loader=loader,
).run()
```

A custom directory layers over the packaged skills rather than replacing them, so
the bundled playbooks stay available and can be named in the same `skills=[...]`
list. In a Fabric notebook, a Lakehouse `Files` path works:
`SkillLoader(skill_dir="/lakehouse/default/Files/skills")`.

## financial_documents

Reporting conventions for 10-K, 10-Q, annual reports and earnings releases:
parentheses as negative, scale stated in a header rather than beside the number,
fiscal year against calendar year, adjacent period columns, subtotal and contra
rows, restatements and non-GAAP measures.

Load it alongside `pdf_document_analysis`, which covers getting text and tables off
the page. This playbook covers what the numbers on that page mean.

It is deliberately domain-scoped. Parentheses mark a subsection in a legal document
and a citation or uncertainty bound in a scientific paper, so the same rule applied
to a non-financial PDF would corrupt figures rather than correct them. Do not load
it for general document work.

### What it was measured on

A 40-question set built from tables in 24 SEC filings. Each question names a table
by its column heading and first row label, then asks which row holds the largest or
smallest value in that column. Half ask for the smallest value, so grabbing the
biggest number on the page does not score. Twenty of the forty have a negative in
the candidate set. Gold answers were computed from the parsed tables and verified
against the source PDFs, and grading is deterministic with no judge.

    pdf_document_analysis alone                     32 / 40
    plus financial_documents                        35 / 40    5 fixed, 2 broke
    subset with a negative candidate value          16 / 20 baseline, 19 / 20 with

The gain concentrates on questions where a value is parenthesised, and reading the
traces confirms the mechanism. The baseline picked "Boeing Capital $199 million" as
the smallest value in a column that contained "(231)", and picked "Global Ventures
1.7%" as the smallest against "(24.2)%". It was reading bracketed negatives as
positive numbers.

### What the numbers do not say

The result is not statistically significant: two-sided McNemar on the paired runs
gives p = 0.453. Forty questions cannot separate a modest effect from noise, and
two questions that passed without the skill failed with it.

The set was built to concentrate sign handling, so the three-question gain does not
transfer to a general accuracy figure. On FinanceBench, where roughly one in five
extraction errors turns on a sign, the expected effect is closer to one question in
150.

That is why this skill sits here rather than in the package. It encodes conventions
that hold independently of any benchmark, and it fixed every sign error it was shown
without breaking the negative-value subgroup, but the evidence is not strong enough
to change what a default install puts in front of every task.

A generic "enumerate the candidates and compute the extreme" rule was tested on the
same set and rejected: 32 / 40, three fixed and three broken. It made the model cast
a wider net and pull rows in from neighbouring tables. That rule is not in this
directory and is not shipped anywhere.

## Not measured (withheld): output_contract

A playbook for treating a task statement's output specification as a strict
contract (exact headers, positional metric-to-column mapping, implied row
filters, stated formats and sort orders, reload-and-verify before submit). It
was drafted after three AgenticDataBench failures where the analysis was right
and the output shape was wrong, then measured on ten held-out tasks under a
decision rule fixed before the run
(`examples/agenticdatabench_pilot/PREREGISTRATION.md`).

    baseline run 1                0.551
    plus output_contract          0.496    1 up, 3 down, 6 unchanged
    baseline run 2 (control)      0.450    same config, no skill

The control run is the finding. Two identical baseline runs differ by 0.101 and
agree on only 6 of 10 tasks, while the skill effect under test was 0.055. The
noise floor is about twice the effect, so the comparison resolves nothing in
either direction. The regressions that a first draft attributed to the skill
reproduce without it, so that mechanism story was fitted to noise and has been
withdrawn.

The skill stays out of the directory because nothing shows it helps and the
directory exists to keep unmeasured guidance out of a default install. But it
was not shown to hurt, and the honest label is "not measured" rather than
"rejected on evidence". Re-testing it needs a pinned temperature, a larger
sample, and a same-config control; see the pre-registration for the full
account.

Nothing derived from inspecting gold files was put into the skill or the
prompt. Per-data-source hints would have raised the score and measured
nothing, since the facts in them came from the answer key. That constraint is
specific to benchmarking: in production, house metric definitions stored beside
the data are exactly what a skill should carry.

## driver_analysis

A method for answering "why did this metric move?" from data: validate the premise
before explaining it, frame an explicit gap against a stated baseline (trend and
seasonality adjusted where the metric has a pattern), locate the move in time and
classify its shape (step change, drift, one-period dip, or in line with the
pattern), decompose through four lenses (dimensional contribution, rate times
volume, mix versus within-segment, population change), and reconcile named drivers
plus a residual to the gap. The report contract requires a premise check, a ranked driver table with
signed contributions, an explicit residual, and a "not checked" section.

It is deliberately domain generic. Sales, scrap rate, churn, conversion, margin and
cycle time decompose identically; the playbook carries the method, and house metric
definitions belong in a separate lakehouse-specific skill layered on top. Load it
with `data_exploration` (declared as a dependency, so the router co-loads it).

The verifier is structural and permissive: it checks that a submitted report has
the required sections and is quantified, not that the drivers are correct. There is
no gold answer for an insight, so correctness has to come from the method, and the
verifier only enforces that the method's artifacts are present.

### What it was measured on

Nothing yet. The playbook encodes standard analyst practice (contribution
decomposition, mix-shift analysis, like-for-like splits) that holds independently of
any benchmark, but no paired runs exist to show it changes outcomes. The natural
measurement is DataClawBench (arXiv 2605.02503), whose exploratory financial tasks
match this skill's scope; until such a run exists, treat the skill as method
documentation, not as a measured improvement. Per the house rule, the eval and its
decision rule should be registered before the skill is tuned against any failures.

### Portable variant

`contrib-skills/driver-analysis/SKILL.md` is the same method in the Agent Skills
format used by Claude Code, Cowork, and GitHub Copilot. Its primary intended use
is Cowork with the Fabric IQ integration against Power BI semantic models, and it
carries a section on semantic-model mechanics: refresh freshness, filter context
and row-level security, quoting the measure's DAX instead of re-deriving it, and
testing additivity before reconciling contributions. To install it:

- Claude Code: copy the `driver-analysis/` folder into `.claude/skills/` in a
  project, or `~/.claude/skills/` for all projects.
- Cowork: package the folder as a `.skill` file and save it, or add it to the
  session's skills.
- GitHub Copilot: copy the folder into `.github/skills/` in the repository.

The two copies must be kept in sync by hand; the fabric variant additionally
carries router frontmatter and the SUBMIT verifier, which the portable format has
no runtime for.
