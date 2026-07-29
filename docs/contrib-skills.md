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
