---
applies_when:
  keywords:
    - insight
    - deep business insight
    - hidden subgroup
    - hidden pattern
    - emerging trend
    - identify kpi
    - cohort
    - anomaly
    - concentration
    - mix shift
    - synthesize signal
  output_fields:
    - insights
excludes: []
depends_on: []
specificity: domain
---
# deep_insight_discovery

Summary: Source-agnostic discovery of decision-grade KPIs, hidden patterns, subgroups, trends, anomalies, and synthesized business insights with independently recomputed evidence.

## Purpose

Discover material signals a human analyst may miss, then turn only source-derived and independently verified candidates into prioritized, decision-grade insights.

## Contract: output fields

When this skill is active, submit an `insights: list[dict]`. Each insight must
contain:

- **title: str** - concise and distinct from every other title.
- **statement: str** - measured fact with population, period, comparison, unit,
  and effect size; no unsupported causal language.
- **interpretation: str** - why the measured fact matters, separated from the
  fact itself.
- **competing_explanations: list[str]** - plausible alternative mechanisms when
  causality is not established.
- **action: dict** - non-empty `owner`, `segment`, `decision`, `target`, and
  `time_horizon`.
- **priority: dict** - `impact`, `urgency`, and integer `rank`.
- **confidence: dict** - `level` and a `reason` grounded in coverage, sample
  size, reconciliation, and robustness.
- **limitations: list[str]** - what the data cannot establish.
- **verification: dict** - `method`, source-derived `expression`, and non-empty
  alias-to-source `sources`.
- **causal_evidence: dict | omitted** - required only for affirmative causal
  claims; when present it needs `design`, `result`, and `limitations`.

The portable verifier below checks structure, obvious self-verification,
lineage, duplication, and causal restraint. The host task must execute each
verification expression through the bound source and assert that its result
equals the submitted metric. Only call `SUBMIT(...)` after both checks pass.

## Required verifier

```python
def verify(payload):
    import re

    def strip_comments(text):
        text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
        return re.sub(r"--[^\n]*", " ", text)

    def constant_expression(expression):
        text = expression.strip()
        if re.search(r"(?:\*\s*0\b|\b0\s*\*)", text):
            return True
        text = re.sub(r"\b[a-z_]\w*\s*(?=\()", " ", text, flags=re.I)
        text = re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", " 0 ", text)
        text = re.sub(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", " 0 ", text, flags=re.I)
        text = re.sub(
            r"\b(null|true|false|cast|try_cast|as|decimal|numeric|double|precision|"
            r"real|integer|int|bigint|smallint|date|timestamp|interval)\b",
            " ",
            text,
            flags=re.I,
        )
        text = re.sub(r"[\s0()+\-*/%.,:]+", "", text)
        return not text

    def simple_source_metric(expression):
        identifier = (
            r'(?:[A-Za-z_]\w*|"[^"]+"|\[[^\]]+\]|`[^`]+`)'
            r'(?:\s*\.\s*(?:[A-Za-z_]\w*|"[^"]+"|\[[^\]]+\]|`[^`]+`))*'
        )
        aggregate = (
            rf"(?:count|sum|avg|min|max|median|stddev|stddev_pop|stddev_samp|"
            rf"var_pop|var_samp)\s*\(\s*(?:distinct\s+)?(?:\*|{identifier})\s*\)"
        )
        return bool(
            re.fullmatch(rf"(?:{identifier}|{aggregate})", expression.strip(), flags=re.I)
        )

    def sql_select_blocks(sql):
        text = strip_comments(sql)
        lower = text.lower()
        blocks = []

        def depth_before(end):
            depth = 0
            quote = None
            for index, char in enumerate(text[:end]):
                if quote:
                    if char == quote:
                        quote = None
                    continue
                if char in "'\"":
                    quote = char
                elif char == "(":
                    depth += 1
                elif char == ")":
                    depth = max(0, depth - 1)
            return depth

        for match in re.finditer(r"\bselect\b", lower):
            start = match.end()
            base_depth = depth_before(match.start())
            depth = base_depth
            quote = None
            index = start
            select_end = None
            block_end = len(text)
            while index < len(text):
                char = text[index]
                if quote:
                    if char == quote:
                        if index + 1 < len(text) and text[index + 1] == quote:
                            index += 2
                            continue
                        quote = None
                    index += 1
                    continue
                if char in "'\"":
                    quote = char
                elif char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth < base_depth:
                        block_end = index
                        break
                elif char == ";" and depth == base_depth:
                    block_end = index
                    break
                elif (
                    select_end is None
                    and depth == base_depth
                    and re.match(r"from\b", lower[index:])
                ):
                    select_end = index
                index += 1
            select_end = block_end if select_end is None else select_end
            blocks.append((text[start:select_end], text[match.start():block_end]))
        return blocks

    def split_projections(select_list):
        parts = []
        start = 0
        depth = 0
        quote = None
        for index, char in enumerate(select_list):
            if quote:
                if char == quote:
                    quote = None
                continue
            if char in "'\"":
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth = max(0, depth - 1)
            elif char == "," and depth == 0:
                parts.append(select_list[start:index])
                start = index + 1
        parts.append(select_list[start:])
        return parts

    def sql_metric_blocks(sql):
        metric_blocks = []
        metric_name = r"(?:metric_value|current_value)"
        quoted_metric_name = (
            rf"(?:{metric_name}|\"{metric_name}\"|\[{metric_name}\]|`{metric_name}`)"
        )
        alias = re.compile(
            rf"^(.*?)\s+(?:as\s+)?{quoted_metric_name}\s*$",
            re.I | re.DOTALL,
        )
        for select_list, block in sql_select_blocks(sql):
            for projection in split_projections(select_list):
                clean = projection.strip()
                if re.fullmatch(quoted_metric_name, clean, flags=re.I):
                    metric_blocks.append((clean, sql_relations(block)))
                    continue
                match = alias.match(clean)
                if match:
                    metric_blocks.append((match.group(1).strip(), sql_relations(block)))
        return metric_blocks

    def normalize_identifier(value):
        return re.sub(r"[\s\"'`\[\]]+", "", str(value)).lower()

    def sql_relations(sql):
        text = strip_comments(sql)
        text = re.sub(r"'(?:''|[^'])*'", "''", text)
        relation = re.compile(
            r"\b(?:from|join)\s+"
            r"((?:\"[^\"]+\"|\[[^\]]+\]|`[^`]+`|[\w-]+)"
            r"(?:\s*\.\s*(?:\"[^\"]+\"|\[[^\]]+\]|`[^`]+`|[\w-]+))*)",
            re.I,
        )
        return {normalize_identifier(match.group(1)) for match in relation.finditer(text)}

    def sql_has_comma_join(sql):
        text = strip_comments(sql)
        depth = 0
        quote = None
        in_from = False
        index = 0
        terminal_clauses = re.compile(
            r"(?:where\b|group\s+by\b|having\b|qualify\b|"
            r"window\s+(?:\"[^\"]+\"|\[[^\]]+\]|`[^`]+`|[\w-]+)\s+as\s*\(|"
            r"order\s+by\b|limit\b|offset\b|fetch\b|union\b|intersect\b|except\b)",
            re.I,
        )
        while index < len(text):
            char = text[index]
            if quote:
                if quote == "]":
                    if char == "]":
                        quote = None
                elif char == quote:
                    if index + 1 < len(text) and text[index + 1] == quote:
                        index += 2
                        continue
                    quote = None
                index += 1
                continue
            if char in "'\"`":
                quote = char
            elif char == "[":
                quote = "]"
            elif char == "(":
                depth += 1
            elif char == ")":
                depth = max(0, depth - 1)
            elif depth == 0:
                previous = text[:index].rstrip()[-1:]
                token_boundary = (
                    (index == 0 or not re.match(r"[\w$]", text[index - 1]))
                    and previous != "."
                )
                if (
                    token_boundary
                    and not in_from
                    and re.match(r"from\b", text[index:], flags=re.I)
                ):
                    in_from = True
                    index += len("from")
                    continue
                terminal = (
                    terminal_clauses.match(text[index:])
                    if token_boundary and in_from
                    else None
                )
                if terminal:
                    remainder = text[index + len(terminal.group(0)):].lstrip()
                    if remainder.startswith(","):
                        return True
                    if re.match(
                        r"(?:indexed\s+by|not\s+indexed)\b",
                        remainder,
                        flags=re.I,
                    ):
                        index += 1
                        continue
                    return False
                if in_from and char == ",":
                    return True
            index += 1
        return False

    def sql_cte_names(sql):
        text = strip_comments(sql)
        pattern = re.compile(
            r"(?:\bwith(?:\s+recursive)?\b|,)\s*"
            r"(\"[^\"]+\"|\[[^\]]+\]|`[^`]+`|[\w-]+)"
            r"(?:\s*\([^)]*\))?\s+as\s*\(",
            re.I,
        )
        return {normalize_identifier(match.group(1)) for match in pattern.finditer(text)}

    def expression_tokens(text):
        text = strip_comments(text)
        text = re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", " ", text)
        return set(re.findall(r"\b[a-z_]\w*\b", text.lower()))

    insights = payload.get("insights")
    assert isinstance(insights, list) and insights, "insights must be a non-empty list"
    seen = set()
    causal = re.compile(r"\b(caused|causes|driven by|due to|resulted in|leads? to)\b", re.I)
    disclaimer = re.compile(
        r"\b(cannot|does not|did not|insufficient to|unable to)\b.{0,40}"
        r"\b(establish|prove|infer|determine)\b.{0,40}\b(caus|whether)\b",
        re.I,
    )
    negated_causal = re.compile(
        r"\b(?:not|never)\b.{0,24}\b(?:caused|causes|driven by|due to|resulted in|leads? to)\b",
        re.I,
    )
    sentinel = {"none", "n/a", "na", "unknown", "not applicable", "null"}

    for index, insight in enumerate(insights, start=1):
        assert isinstance(insight, dict), f"insight {index} must be an object"
        for field in (
            "title", "statement", "interpretation", "competing_explanations",
            "action", "priority", "confidence", "limitations", "verification",
        ):
            assert field in insight, f"insight {index} missing {field}"

        assert isinstance(insight["title"], str), f"insight {index} title must be text"
        assert isinstance(insight["statement"], str), f"insight {index} statement must be text"
        assert isinstance(insight["interpretation"], str), (
            f"insight {index} interpretation must be text"
        )
        title = insight["title"].strip()
        statement = insight["statement"].strip()
        interpretation = insight["interpretation"].strip()
        assert title and statement and interpretation
        title_fingerprint = re.sub(r"\W+", " ", title.lower()).strip()
        statement_fingerprint = re.sub(r"\W+", " ", statement.lower()).strip()
        assert title_fingerprint not in seen, f"duplicate insight title: {title}"
        assert statement_fingerprint not in seen, f"duplicate insight statement: {title}"
        seen.update((title_fingerprint, statement_fingerprint))

        competing = insight["competing_explanations"]
        assert (
            isinstance(competing, list)
            and competing
            and all(
                isinstance(item, str)
                and item.strip()
                and item.strip().lower() not in sentinel
                for item in competing
            )
        ), f"insight {index} competing_explanations must be non-empty text"

        causal_clauses = [
            clause.strip()
            for clause in re.split(
                r"(?<=[.!?])\s+|[,;]\s*(?:although|but|however)\s+",
                f"{statement} {interpretation}",
                flags=re.I,
            )
            if causal.search(clause)
        ]
        unsupported_causal = [
            clause
            for clause in causal_clauses
            if not disclaimer.search(clause) and not negated_causal.search(clause)
        ]
        if unsupported_causal:
            evidence = insight.get("causal_evidence")
            assert isinstance(evidence, dict), (
                f"insight {index} uses causal language without structured causal evidence"
            )
            for field in ("design", "result", "limitations"):
                value = str(evidence.get(field, "")).strip()
                assert value and value.lower() not in sentinel, (
                    f"insight {index} causal evidence missing {field}"
                )

        action = insight["action"]
        assert isinstance(action, dict), f"insight {index} action must be structured"
        for field in ("owner", "segment", "decision", "target", "time_horizon"):
            assert isinstance(action.get(field), str) and action[field].strip(), (
                f"insight {index} action missing {field}"
            )

        priority = insight["priority"]
        assert isinstance(priority, dict), f"insight {index} priority must be structured"
        for field in ("impact", "urgency"):
            assert isinstance(priority.get(field), str) and priority[field].strip(), (
                f"insight {index} priority missing {field}"
            )
        assert (
            isinstance(priority.get("rank"), int)
            and not isinstance(priority["rank"], bool)
            and priority["rank"] >= 1
        ), f"insight {index} priority rank must be a positive integer"

        confidence = insight["confidence"]
        assert isinstance(confidence, dict), f"insight {index} confidence must be structured"
        assert isinstance(confidence.get("level"), str) and confidence["level"].strip()
        assert isinstance(confidence.get("reason"), str) and confidence["reason"].strip()
        assert (
            isinstance(insight["limitations"], list)
            and insight["limitations"]
            and all(
                isinstance(item, str)
                and item.strip()
                and item.strip().lower() not in sentinel
                for item in insight["limitations"]
            )
        ), (
            f"insight {index} limitations must be non-empty text"
        )

        verification = insight["verification"]
        assert isinstance(verification, dict), f"insight {index} verification must be structured"
        method = str(verification.get("method", "")).strip().lower()
        expression = str(verification.get("expression", "")).strip()
        sources = verification.get("sources")
        assert method in {"sql", "dax", "python", "api"}, (
            f"insight {index} verification method is unsupported"
        )
        assert expression, f"insight {index} verification expression is required"
        assert isinstance(sources, dict) and sources, (
            f"insight {index} verification sources are required"
        )
        assert all(
            isinstance(alias, str) and alias.strip()
            and isinstance(source, str) and source.strip()
            for alias, source in sources.items()
        ), f"insight {index} verification sources are invalid"

        if method == "sql":
            sql_without_literals = re.sub(
                r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"",
                " ",
                strip_comments(expression),
            )
            assert (
                len(re.findall(r"\bselect\b", sql_without_literals, flags=re.I)) == 1
                and not re.search(r"\bwith(?:\s+recursive)?\b", sql_without_literals, flags=re.I)
                and ";" not in sql_without_literals.rstrip().rstrip(";")
            ), (
                f"insight {index} verification must recompute metric_value from "
                "a declared source using one flat SELECT"
            )
            assert not sql_has_comma_join(expression), (
                f"insight {index} verification must use explicit JOIN syntax; "
                "comma joins are not supported"
            )
            metric_blocks = sql_metric_blocks(expression)
            assert metric_blocks, (
                f"insight {index} verification must produce metric_value"
            )
            assert all(not constant_expression(item[0]) for item in metric_blocks), (
                f"insight {index} verification must recompute metric_value from source data"
            )
            assert all(simple_source_metric(item[0]) for item in metric_blocks), (
                f"insight {index} verification must recompute metric_value with "
                "a source column or one aggregate; verify derived metrics as components"
            )
            declared = {
                normalize_identifier(value)
                for pair in sources.items()
                for value in pair
            }
            assert all(
                relations and relations <= declared for _, relations in metric_blocks
            ), (
                f"insight {index} verification does not reference a declared source"
            )
        elif method == "dax":
            row_metrics = re.findall(
                r"\"(?:metric_value|current_value)\"\s*,\s*([^,)]+)",
                strip_comments(expression),
                flags=re.I,
            )
            assert row_metrics and all(
                not constant_expression(item) for item in row_metrics
            ), f"insight {index} verification must recompute metric_value from source data"
            declared_tokens = {
                normalize_identifier(value).split(".")[-1]
                for pair in sources.items()
                for value in pair
            }
            metric_tokens = set().union(*(expression_tokens(item) for item in row_metrics))
            assert metric_tokens & declared_tokens, (
                f"insight {index} verification does not reference a declared source"
            )
        else:
            assignments = re.findall(
                r"\b(?:metric_value|current_value)\s*=\s*([^\n;]+)",
                strip_comments(expression),
                flags=re.I,
            )
            assert assignments and all(
                not constant_expression(item) for item in assignments
            ), f"insight {index} verification must recompute metric_value from source data"
            declared_tokens = {
                normalize_identifier(value).split(".")[-1]
                for pair in sources.items()
                for value in pair
            }
            metric_tokens = set().union(*(expression_tokens(item) for item in assignments))
            assert metric_tokens & declared_tokens, (
                f"insight {index} verification does not reference a declared source"
            )
```

## Tripwires

- Run `3c5f29ba59e449a3aa04c9eff8f06c33` passed because its SQL selected model-supplied constants instead of recomputing metrics.
- An all-time active-status share was labeled churn even though no cohort transition or time-based churn denominator was calculated.
- Aggregate trends were presented without searching interactions or subgroups that could reverse the conclusion.
- Interpretations used `driven by` and `caused` when the data supported association only.
- Generic actions such as "monitor closely" passed despite having no owner, target, or time horizon.

## Invariants

- `insights` is a non-empty list and every item satisfies the output contract.
- Every verification expression computes `metric_value` or `current_value` from
  a declared source; selecting or assigning constants is invalid.
- The host independently executes verification and matches metric, comparison,
  period, population, denominator, units, and sample size.
- Affirmative causal language requires structured causal evidence; explicit
  causal disclaimers do not.
- Every action has an owner, segment, decision, measurable target, and time horizon.
- Duplicate findings are rejected; synthesis must combine rather than repeat signals.

## Procedure

### 1. Map the business system

Infer the provisional business model and industry from source names, columns,
values, and relationships. Identify entities, events, facts, dimensions, keys,
time grains, currencies, lifecycle stages, date coverage, missingness, and
population changes. Separate observed schema facts from assumptions.

### 2. Construct a use-case KPI tree

Build outcomes -> drivers -> diagnostics across value, retention, adoption,
efficiency, customer experience, and risk. Classify each KPI as **computable**,
**partially computable**, or **not computable**. Never silently substitute a
proxy: a status snapshot is not churn, activity is not retention, and compute
consumption is not customer value.

### 3. Search systematically

Use bounded code or queries to evaluate KPI x time, subgroup, cohort, lifecycle
stage, and two-dimensional interaction. Search seasonality, acceleration,
change points, concentration, tail risk, mix shifts, anomalies, lead/lag
signals, reconciliation failures, and aggregate/subgroup reversals. Record
sample size, baseline, effect size, absolute impact, persistence, and
sensitivity to reasonable windows and denominators.

### 4. Select non-redundant candidates

Score materiality, surprise, persistence, confidence, actionability, novelty,
and strategic relevance. Apply a non-redundancy check and reject obvious chart
captions, tiny immaterial groups, and several variants of the same signal.

### 5. Verify independently

Recompute each candidate through the bound source using SQL, DAX, bounded API
aggregation, or code over a bound file. Check source lineage, join
multiplicity, null treatment, numerator, denominator, population, period,
units, and sample size. Reconcile against an alternate calculation where
practical. Portable SQL verification uses one flat aggregate `SELECT`; split a
complex metric into multiple simple component queries and let the host combine
and cross-check them. Run the required verifier and the host equality checks
before submission.

### 6. Calibrate interpretation and action

Keep measured fact, interpretation, competing explanations, causal evidence,
action, confidence, and limitations separate. Prefer "is associated with,"
"is consistent with," or "may reflect" unless the design supports causality.

### 7. Synthesize

Group findings into two to four decision themes. Explain how signals reinforce
or contradict one another, what the aggregate masked, the best-fitting and
competing explanations, the decision justified now, and the next analysis that
would discriminate between explanations. Synthesis must add information beyond
concatenating statements.
