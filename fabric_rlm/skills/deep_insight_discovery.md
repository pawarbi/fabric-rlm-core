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

When this skill is active, submit:

- **contract_version: int** - baseline runs emit version `2`. Evidence-closure
  runs emit version `3`. Omission defaults to legacy version `1` solely for
  compatibility with existing consumers. Versions `2` and `3` require typed
  diagnostics for every insight; unsupported versions fail verification.
- **analysis_plan: dict** - source-derived `business_context`, a non-empty
  `kpi_map`, and `search_space`. Each KPI records `kpi`, `computability`
  (`computable`, `partially_computable`, or `not_computable`), and `reason`.
  Search space records lists of `dimensions_available`, `dimensions_deferred`
  (`dimension` plus substantive `reason`), `time_grains_available`, and
  non-empty `populations`. Empty dimension and time lists are valid when the
  source genuinely has neither.
- **candidates: list[dict]** - the auditable search ledger. Every entry records
  `candidate`, `dimensions_tested`, `disposition` (`promoted` or `rejected`),
  and a substantive `reason`; promoted candidates also identify the exact
  insight `title` in `promoted_as`. Every submitted insight must trace to
  exactly one promoted candidate, and its discovery dimensions must match that
  candidate's tested dimensions. Rejected candidates also record
  `rejection_type`: quantitative rejections require source-derived
  `rejection_evidence` for effect, baseline, and sample size; `not_computable`
  rejections identify the missing fields; `redundant` rejections identify the
  promoted insight they duplicate. Do not invent rejected candidates merely
  to increase the count.
- **insights: list[dict]** - the promoted findings described below.

Each insight must contain:

- **title: str** - concise and distinct from every other title. It must not
  claim more than `evidence_tier` and `decision_readiness` permit.
  Investigate-first titles use calibrated terms such as `associated`,
  `observed`, `possible`, `signal`, or `warrants investigation`, never
  `affects`, `drives`, `causes`, `root cause`, `primary lever`, or `failure`.
- **statement: str** - exactly one primary measured fact with population,
  period, comparison, unit, and effect size; no unsupported causal language.
  Opaque forms such as `the primary measured difference is X` are invalid.
- **interpretation: str** - why the measured fact matters, separated from the
  fact itself and containing no additional quantitative claims. It must not
  convert an investigate-first observation into a planning baseline, dominant
  business driver, or counterfactual operational conclusion.
- **competing_explanations: list[str]** - plausible alternative mechanisms when
  causality is not established.
- **diagnostic_measurability: str | omitted** - typed-contract marker:
  `measurable`, `not_measurable`, or `mixed`. When present,
  `diagnostic_assessment` is required. Current-version insights must provide
  both fields. Only version `1` legacy insights may omit both.
- **diagnostic_assessment: dict | omitted** - required when
  `diagnostic_measurability` is declared. Contains `decision_readiness`
  (`act_ready` or `investigate_first`) and `explanations`, which must cover the
  `competing_explanations` exactly. Each explanation has `explanation`,
  boolean `measurable`, and `disposition` (`ruled_out`, `weakened`,
  `unresolved`, `supported`, or `not_measurable`). A measurable `ruled_out`,
  `weakened`, or `supported` item also has finite numeric `expected_value` and
  independent `verification`. A non-measurable item uses only
  `not_measurable` and supplies a substantive `limitation`. Version `3` also
  requires a unique stable `explanation_id` and `closure_status` (`pending`,
  `weakened`, `ruled_out`, `supported`, or `unresolvable`). Every measurable
  explanation supplies a substantive `required_check`. Closure status must
  agree with disposition: pending/unresolved, weakened/weakened,
  ruled_out/ruled_out, supported/supported, or
  unresolvable/not_measurable.
- **action: dict** - non-empty `owner`, `segment`, `decision`, `target`, and
  `time_horizon`. Typed diagnostics additionally require `kind`
  (`diagnostic` or `program`).
- **priority: dict** - `impact`, `urgency`, and integer `rank`.
- **confidence: dict** - `level` and a `reason` grounded in coverage, sample
  size, evidence design, reconciliation, and robustness.
- **evidence_tier: str** - `descriptive`, `associational`, or `causal`.
  Causal language and causal tier both require structured causal evidence.
- **limitations: list[str]** - what the data cannot establish.
- **temporal_context: dict | omitted** - required for titles framed as current,
  latest, recent, today, or this period. It declares `time_basis`, `timezone`,
  `requested_as_of`, `data_as_of`, `trustworthy_through`,
  `latest_complete_period`, `current_window`, `comparators`,
  `partial_period_policy`, `completeness_basis`, `recency_status`, and boolean
  `supports_current_action`, plus `evidence_fingerprints` linking the temporal
  status to deterministic coverage and window results. Status is `current_change`, `current_level`,
  `persistent`, `recurring_seasonal`, `historical`, `stale`, or
  `not_applicable`. A current change requires a complete current window and at
  least one comparator. Only current-change, current-level, and persistent
  evidence may support a current program action.
- **verification: dict | omitted** - legacy simple-metric form containing
  `method`, source-derived `expression`, and non-empty alias-to-source
  `sources`. Omit it when `metric_spec` supplies independently verified
  components; every claim must provide one form or the other.
- **metric_spec: dict | omitted** - additive, discriminated metric metadata for
  the primary claim. `type` is `value`, `count`, `amount`, `average`, `rate`,
  `share`, `delta`, `rate_of_change`, `decomposition`, or `correlation`;
  `expected_value` is finite numeric; and `components` contains unique, named,
  source-derived values with `role`, finite numeric `expected_value`, and
  independent `verification`. Simple types use role `value`; every count also declares
  `comparison.kind` as `none` or `cross_period`. Rates and shares use
  `numerator` and `denominator`; deltas and rates of change use `current` and
  `comparison`. Arithmetic is `numerator / denominator`, `current -
  comparison`, and `(current - comparison) / abs(comparison)`, respectively,
  compared with `rel_tol=1e-9` and `abs_tol=1e-9`. A decomposition uses one `total_delta`, one or more
  `contribution` components, and exactly one explicit `residual`.
  Correlation uses complete cases and independently verified `pair_count`,
  `sum_x`, `sum_y`, `sum_x_squared`, `sum_y_squared`, and `sum_xy` components,
  plus named `x`/`y` variables and an explicit population. Every correlation
  component repeats the same `variables`, `population`, and
  `pairwise_missing_policy="complete_cases"` metadata so the sufficient
  statistics cannot be assembled from inconsistent samples.
- **supporting_claims: list[dict]** - zero or more secondary quantitative facts,
  each with `claim` text, `expected_value`, and either its own legacy
  `verification` or `metric_spec`. A supporting metric spec's expected value
  must match the claim's `expected_value`.
  The host executes every component expression and compares its result with
  that component's expected value. Never hide secondary numbers in
  interpretation.
- **discovery: dict** - `pattern_type` (`portfolio_trend`, `subgroup`,
  `cohort_transition`, or `interaction`), non-empty `dimensions_tested`,
  `population`, positive integer `sample_size`, and non-empty
  `robustness_checks`. Interactions must test at least two dimensions; cohort
  transitions must check denominator or population-composition stability.
  Claimed interactions additionally require `interaction_evidence` with at
  least two labeled cells, numeric effects and sample sizes, a baseline effect,
  and a substantive explanation of the observed heterogeneity.
- **causal_evidence: dict | omitted** - required only for affirmative causal
  claims; when present it needs `design`, `result`, and `limitations`.

For a cross-period `count`, add `comparison: {"kind": "cross_period",
"population": ...}`. A `variable` population requires `current`,
`comparison`, `current_denominator`, and `comparison_denominator` components,
plus `current_rate` and `comparison_rate`; the verifier recomputes both rates.
A genuinely `stable` or `exhaustive` population instead requires independently
verified `current_denominator` and `comparison_denominator` components. The
portable verifier asserts that both are positive and equal within tolerance;
a model-supplied stability flag is not evidence.

The portable verifier below checks structure, component arithmetic, obvious
self-verification, lineage, duplication, and causal restraint. Derived results
are recomputed from independently verified component expected values; a
model-supplied derived number is never trusted. Division by zero is invalid.
The host task must execute each verification expression through the bound
source and assert that its result equals the submitted component. Only call
`SUBMIT(...)` after both checks pass.

If any measurable explanation remains `unresolved` or is supported by closure
evidence, confidence cannot be `high`, urgency cannot be `critical`,
`decision_readiness` must be `investigate_first`, and `action.kind` must be
`diagnostic`. This structural gate does not infer measurability from prose. The model's
declarations and source checks are auditable inputs; a semantic critic remains
responsible for challenging false `not_measurable` declarations, but cannot
waive current-version structure.

Critic-driven closure may append a new explanation only when it is a bounded
follow-up analysis for a specific material or blocking challenge. The follow-up
must be added to both `competing_explanations` and the typed assessment. If the
frozen sources lack the requested field, period, denominator, or comparator,
record `closure_status: unresolvable` and `disposition: not_measurable` with a
substantive limitation; an empty population must never be reported as a
measured zero.

## Required verifier

```python
def verify(payload):
    import hashlib
    import json
    import math
    import re
    from datetime import date, datetime, timedelta
    from decimal import Decimal, localcontext
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    def collect_strings(value):
        strings = []
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                strings.extend(collect_strings(child))
        elif isinstance(value, list):
            for child in value:
                strings.extend(collect_strings(child))
        return strings

    def strip_comments(text):
        text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
        return re.sub(r"--[^\n]*", " ", text)

    def constant_expression(expression):
        text = expression.strip()
        if re.search(r"(?:\*\s*0\b|\b0\s*\*)", text):
            return True
        text = re.sub(r"\b[a-z_]\w*\s*(?=\()", " ", text, flags=re.I)
        text = re.sub(r"'(?:''|[^'])*'", " 0 ", text)
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
        aggregate_name = (
            r"(?:count|sum|avg|min|max|median|stddev|stddev_pop|stddev_samp|"
            r"var_pop|var_samp|quantile_cont|quantile_disc)"
        )
        text = expression.strip()
        if re.fullmatch(identifier, text, flags=re.I):
            return True
        if re.fullmatch(
            rf"count\s*\(\s*(?:distinct\s+)?\*\s*\)"
            rf"(?:\s*::\s*(?:double|real|decimal|numeric|"
            rf"integer|int|bigint|smallint))?",
            text,
            flags=re.I,
        ):
            return True
        aggregate_calls = re.findall(
            rf"\b{aggregate_name}\s*\(",
            text,
            flags=re.I,
        )
        if len(aggregate_calls) != 1 or constant_expression(text):
            return False
        without_literals = re.sub(r"'(?:''|[^'])*'", " ", text)
        function_names = {
            name.lower()
            for name in re.findall(
                r"\b([A-Za-z_]\w*)\s*\(",
                without_literals,
            )
        }
        tokens = set(re.findall(r"\b[A-Za-z_]\w*\b", without_literals))
        keywords = {
            "as", "case", "cast", "distinct", "else", "end", "false",
            "interval", "null", "then", "true", "when",
        }
        return bool(
            {
                token.lower()
                for token in tokens
                if token.lower() not in function_names
                and token.lower() not in keywords
            }
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

    def dax_references(text):
        clean = strip_comments(text)
        measures = {
            normalize_identifier(match.group(1))
            for match in re.finditer(
                r"(?<![\w'\]])\[([^\]]+)\]",
                clean,
            )
        }
        qualified = {
            normalize_identifier(
                f"{table_name}.{match.group(3)}"
            ).replace(".", "")
            for match in re.finditer(
                r"(?:'([^']+)'|([A-Za-z_]\w*))\s*\[([^\]]+)\]",
                clean,
            )
            for table_name in [match.group(1) or match.group(2)]
        }
        return measures, qualified

    def dax_row_metrics(expression):
        text = strip_comments(expression)
        metrics = []
        pattern = re.compile(
            r"\"(?:metric_value|current_value)\"\s*,",
            flags=re.I,
        )
        for match in pattern.finditer(text):
            start = match.end()
            depth = 0
            quote = None
            index = start
            while index < len(text):
                char = text[index]
                if quote is not None:
                    if char == quote:
                        if index + 1 < len(text) and text[index + 1] == quote:
                            index += 2
                            continue
                        quote = None
                    index += 1
                    continue
                if char in {"'", '"'}:
                    quote = char
                elif char == "(":
                    depth += 1
                elif char == ")":
                    if depth == 0:
                        break
                    depth -= 1
                elif char == "," and depth == 0:
                    break
                index += 1
            metric = text[start:index].strip()
            if metric:
                metrics.append(metric)
        return metrics

    def validate_verification(verification, label):
        assert isinstance(verification, dict), f"{label} verification must be structured"
        method = str(verification.get("method", "")).strip().lower()
        expression = str(verification.get("expression", "")).strip()
        sources = verification.get("sources")
        assert method in {"sql", "dax", "python", "api"}, (
            f"{label} verification method is unsupported"
        )
        assert expression, f"{label} verification expression is required"
        assert isinstance(sources, dict) and sources, (
            f"{label} verification sources are required"
        )
        assert all(
            isinstance(alias, str) and alias.strip()
            and isinstance(source, str) and source.strip()
            for alias, source in sources.items()
        ), f"{label} verification sources are invalid"

        if method == "sql":
            sql_without_literals = re.sub(
                r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"",
                " ",
                strip_comments(expression),
            )
            assert (
                re.search(r"\bselect\b", sql_without_literals, flags=re.I)
                and not re.search(
                    r"\bwith\s+recursive\b", sql_without_literals, flags=re.I
                )
                and ";" not in sql_without_literals.rstrip().rstrip(";")
            ), (
                f"{label} verification must recompute metric_value from "
                "a declared source using one read-only SELECT"
            )
            assert not sql_has_comma_join(expression), (
                f"{label} verification must use explicit JOIN syntax; "
                "comma joins are not supported"
            )
            metric_blocks = sql_metric_blocks(expression)
            assert metric_blocks, f"{label} verification must produce metric_value"
            assert all(
                not constant_expression(item[0])
                or re.fullmatch(
                    r"count\s*\(\s*(?:distinct\s+)?\*\s*\)"
                    r"(?:\s*::\s*(?:double|real|decimal|numeric|"
                    r"integer|int|bigint|smallint))?",
                    item[0].strip(),
                    flags=re.I,
                )
                for item in metric_blocks
            ), (
                f"{label} verification must recompute metric_value from source data"
            )
            assert all(simple_source_metric(item[0]) for item in metric_blocks), (
                f"{label} verification must recompute metric_value with "
                "a source column or one aggregate; verify derived metrics as components"
            )
            declared = {
                normalize_identifier(value)
                for pair in sources.items()
                for value in pair
            }
            cte_names = sql_cte_names(expression)
            assert not (cte_names & declared), (
                f"{label} verification CTE cannot shadow a declared source"
            )
            allowed_relations = declared | cte_names
            all_relations = sql_relations(expression)
            assert (
                all_relations
                and all_relations <= allowed_relations
                and bool(all_relations & declared)
            ), f"{label} verification does not reference a declared source"
            assert all(
                relations and relations <= allowed_relations
                for _, relations in metric_blocks
            ), f"{label} verification does not reference a declared source"
        elif method == "dax":
            row_metrics = dax_row_metrics(expression)
            assert row_metrics and all(
                not constant_expression(item) for item in row_metrics
            ), f"{label} verification must recompute metric_value from source data"
            declared_references = {
                normalize_identifier(value).replace(".", "")
                for pair in sources.items()
                for value in pair
            }
            measure_references, _ = dax_references(expression)
            qualified_references = set()
            for item in row_metrics:
                _, item_qualified = dax_references(item)
                qualified_references.update(item_qualified)
            if measure_references:
                valid_lineage = measure_references <= declared_references
            else:
                valid_lineage = bool(
                    qualified_references & declared_references
                )
            assert valid_lineage, (
                f"{label} verification does not reference a declared source"
            )
        else:
            assignments = re.findall(
                r"\b(?:metric_value|current_value)\s*=\s*([^\n;]+)",
                strip_comments(expression),
                flags=re.I,
            )
            assert assignments and all(
                not constant_expression(item) for item in assignments
            ), f"{label} verification must recompute metric_value from source data"
            declared_tokens = {
                normalize_identifier(value).split(".")[-1]
                for pair in sources.items()
                for value in pair
            }
            metric_tokens = set().union(
                *(expression_tokens(item) for item in assignments)
            )
            assert metric_tokens & declared_tokens, (
                f"{label} verification does not reference a declared source"
            )

    def validate_metric_spec(metric_spec, label, claim_expected=None):
        assert isinstance(metric_spec, dict), f"{label} metric_spec must be structured"
        metric_type = metric_spec.get("type")
        simple_types = {"value", "count", "amount", "average"}
        derived_types = {
            "rate", "share", "delta", "rate_of_change", "decomposition",
            "correlation",
        }
        assert metric_type in simple_types | derived_types, (
            f"{label} metric_spec type is invalid"
        )

        def finite_numeric(value, value_label):
            assert (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            ), f"{value_label} must be finite numeric"
            try:
                numeric = float(value)
            except (OverflowError, TypeError, ValueError):
                raise AssertionError(
                    f"{value_label} must be finite numeric"
                ) from None
            assert math.isfinite(numeric), (
                f"{value_label} must be finite numeric"
            )
            return value

        def close(left, right):
            return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)

        expected = finite_numeric(
            metric_spec.get("expected_value"),
            f"{label} metric_spec expected_value",
        )
        if claim_expected is not None:
            submitted = finite_numeric(claim_expected, f"{label} expected_value")
            assert close(expected, submitted), (
                f"{label} metric_spec expected_value does not match claim"
            )

        components = metric_spec.get("components")
        assert isinstance(components, list) and components, (
            f"{label} metric_spec components are required"
        )
        names = set()
        by_role = {}
        component_contexts = []
        for component_index, component in enumerate(components, start=1):
            assert isinstance(component, dict), (
                f"{label} metric component {component_index} must be structured"
            )
            name = component.get("name")
            role = component.get("role")
            assert isinstance(name, str) and name.strip() and name not in names, (
                f"{label} metric component names must be unique non-empty text"
            )
            assert isinstance(role, str) and role.strip(), (
                f"{label} metric component {name!r} role is required"
            )
            names.add(name)
            value = finite_numeric(
                component.get("expected_value"),
                f"{label} metric component {name!r} expected_value",
            )
            by_role.setdefault(role, []).append(value)
            component_contexts.append(component)
            validate_verification(
                component.get("verification"),
                f"{label} metric component {name!r}",
            )

        def exactly_one(role):
            values = by_role.get(role, [])
            assert len(values) == 1, (
                f"{label} metric_spec requires exactly one {role} component"
            )
            return values[0]

        count_comparison = None
        if metric_type == "count":
            count_comparison = metric_spec.get("comparison")
            assert (
                isinstance(count_comparison, dict)
                and count_comparison.get("kind") in {"none", "cross_period"}
            ), f"{label} count comparison metadata is required"

        if metric_type in {"rate", "share"}:
            numerator = exactly_one("numerator")
            denominator = exactly_one("denominator")
            assert denominator != 0, f"{label} metric_spec has a zero denominator"
            recomputed = numerator / denominator
        elif metric_type == "delta":
            recomputed = exactly_one("current") - exactly_one("comparison")
        elif metric_type == "rate_of_change":
            current = exactly_one("current")
            comparison_value = exactly_one("comparison")
            assert comparison_value != 0, (
                f"{label} metric_spec has a zero denominator"
            )
            recomputed = (current - comparison_value) / abs(comparison_value)
        elif metric_type == "decomposition":
            total_delta = exactly_one("total_delta")
            residual = exactly_one("residual")
            contributions = by_role.get("contribution", [])
            assert contributions, (
                f"{label} decomposition requires contribution components"
            )
            assert close(sum(contributions) + residual, total_delta), (
                f"{label} decomposition components do not reconcile to total delta"
            )
            recomputed = total_delta
        elif metric_type == "correlation":
            variables = metric_spec.get("variables")
            assert (
                isinstance(variables, dict)
                and set(variables) == {"x", "y"}
                and all(
                    isinstance(value, str) and value.strip()
                    for value in variables.values()
                )
            ), f"{label} correlation requires named x and y variables"
            population = metric_spec.get("population")
            assert isinstance(population, str) and population.strip(), (
                f"{label} correlation population is required"
            )
            assert metric_spec.get("pairwise_missing_policy") == "complete_cases", (
                f"{label} correlation requires a complete-case missing policy"
            )
            assert all(
                component.get("variables") == variables
                for component in component_contexts
            ), f"{label} correlation components must use the same variables"
            assert all(
                component.get("population") == population
                for component in component_contexts
            ), f"{label} correlation components must use the same population"
            assert all(
                component.get("pairwise_missing_policy") == "complete_cases"
                for component in component_contexts
            ), (
                f"{label} correlation components must use the same "
                "complete-case missing policy"
            )
            pair_count = exactly_one("pair_count")
            assert pair_count >= 2 and close(pair_count, round(pair_count)), (
                f"{label} correlation pair_count must be an integer of at least 2"
            )
            sum_x = exactly_one("sum_x")
            sum_y = exactly_one("sum_y")
            sum_x_squared = exactly_one("sum_x_squared")
            sum_y_squared = exactly_one("sum_y_squared")
            sum_xy = exactly_one("sum_xy")
            with localcontext() as context:
                context.prec = 50
                n_decimal = Decimal(str(pair_count))
                sum_x_decimal = Decimal(str(sum_x))
                sum_y_decimal = Decimal(str(sum_y))
                x_variance_term = (
                    n_decimal * Decimal(str(sum_x_squared))
                    - sum_x_decimal * sum_x_decimal
                )
                y_variance_term = (
                    n_decimal * Decimal(str(sum_y_squared))
                    - sum_y_decimal * sum_y_decimal
                )
                covariance_term = (
                    n_decimal * Decimal(str(sum_xy))
                    - sum_x_decimal * sum_y_decimal
                )
            assert x_variance_term > 0 and y_variance_term > 0, (
                f"{label} correlation requires positive variance in both variables"
            )
            recomputed = float(
                covariance_term / (x_variance_term * y_variance_term).sqrt()
            )
            tolerance = 1e-9
            assert -1.0 - tolerance <= recomputed <= 1.0 + tolerance, (
                f"{label} correlation must be between -1 and 1"
            )
            recomputed = min(1.0, max(-1.0, recomputed))
        elif metric_type == "count" and count_comparison["kind"] == "cross_period":
            comparison = count_comparison
            recomputed = exactly_one("current")
            exactly_one("comparison")
            population = comparison.get("population")
            assert population in {"variable", "stable", "exhaustive"}, (
                f"{label} count comparison requires denominator integrity metadata"
            )
            if population == "variable":
                current_denominator = exactly_one("current_denominator")
                comparison_denominator = exactly_one("comparison_denominator")
                assert current_denominator != 0 and comparison_denominator != 0, (
                    f"{label} count comparison has a zero denominator"
                )
                current_rate = finite_numeric(
                    comparison.get("current_rate"),
                    f"{label} count comparison current rate",
                )
                comparison_rate = finite_numeric(
                    comparison.get("comparison_rate"),
                    f"{label} count comparison comparison rate",
                )
                assert close(recomputed / current_denominator, current_rate), (
                    f"{label} count comparison current rate does not reconcile"
                )
                prior_count = by_role["comparison"][0]
                assert close(
                    prior_count / comparison_denominator,
                    comparison_rate,
                ), f"{label} count comparison comparison rate does not reconcile"
            else:
                current_denominator = exactly_one("current_denominator")
                comparison_denominator = exactly_one("comparison_denominator")
                assert (
                    current_denominator > 0 and comparison_denominator > 0
                ), f"{label} stable population denominators must be positive"
                assert close(current_denominator, comparison_denominator), (
                    f"{label} stable population denominators must be equal"
                )
        else:
            recomputed = exactly_one("value")

        assert close(recomputed, expected), (
            f"{label} metric_spec expected_value does not reconcile with components"
        )

    def validate_diagnostic_assessment(insight, label):
        declaration = insight.get("diagnostic_measurability")
        assessment = insight.get("diagnostic_assessment")
        assert declaration in {"measurable", "not_measurable", "mixed"}, (
            f"{label} diagnostic_measurability is invalid"
        )
        assert isinstance(assessment, dict), (
            f"{label} diagnostic_assessment is required"
        )
        readiness = assessment.get("decision_readiness")
        assert readiness in {"act_ready", "investigate_first"}, (
            f"{label} diagnostic decision_readiness is invalid"
        )
        explanations = assessment.get("explanations")
        assert isinstance(explanations, list) and explanations, (
            f"{label} diagnostic explanations are required"
        )
        assessed_text = []
        measured_states = []
        explanation_ids = set()
        unresolved_measurable = False
        for explanation_index, explanation in enumerate(explanations, start=1):
            item_label = f"{label} diagnostic explanation {explanation_index}"
            assert isinstance(explanation, dict), (
                f"{item_label} must be structured"
            )
            text = explanation.get("explanation")
            assert isinstance(text, str) and text.strip(), (
                f"{item_label} text is required"
            )
            assessed_text.append(text.strip())
            measurable = explanation.get("measurable")
            assert isinstance(measurable, bool), (
                f"{item_label} measurable must be boolean"
            )
            measured_states.append(measurable)
            allowed_dispositions = {
                "ruled_out", "weakened", "unresolved", "not_measurable",
            }
            if contract_version >= 3:
                allowed_dispositions.add("supported")
            disposition = explanation.get("disposition")
            assert disposition in allowed_dispositions, (
                f"{item_label} disposition is invalid"
            )
            if contract_version >= 3:
                explanation_id = explanation.get("explanation_id")
                assert (
                    isinstance(explanation_id, str)
                    and re.fullmatch(r"[a-z][a-z0-9_-]{2,63}", explanation_id)
                ), f"{item_label} explanation_id is invalid"
                assert explanation_id not in explanation_ids, (
                    f"{label} diagnostic explanation_id values must be unique"
                )
                explanation_ids.add(explanation_id)
                closure_status = explanation.get("closure_status")
                assert closure_status in {
                    "pending", "weakened", "ruled_out", "supported", "unresolvable",
                }, f"{item_label} closure_status is invalid"
                expected_disposition = {
                    "pending": "unresolved",
                    "weakened": "weakened",
                    "ruled_out": "ruled_out",
                    "supported": "supported",
                    "unresolvable": "not_measurable",
                }[closure_status]
                assert disposition == expected_disposition, (
                    f"{item_label} closure_status and disposition are inconsistent"
                )
                if measurable:
                    required_check = explanation.get("required_check")
                    assert (
                        isinstance(required_check, str)
                        and required_check.strip()
                        and required_check.strip().lower() not in sentinel
                    ), f"{item_label} required_check is required"
                    assert closure_status != "unresolvable", (
                        f"{item_label} measurable explanation cannot be unresolvable"
                    )
                else:
                    assert closure_status == "unresolvable", (
                        f"{item_label} non-measurable explanation must be unresolvable"
                    )
            if measurable:
                assert disposition != "not_measurable", (
                    f"{item_label} measurable explanation cannot be not_measurable"
                )
                if disposition in {"ruled_out", "weakened", "supported"}:
                    expected_value = explanation.get("expected_value")
                    assert (
                        isinstance(expected_value, (int, float))
                        and not isinstance(expected_value, bool)
                        and math.isfinite(expected_value)
                    ), f"{item_label} expected_value must be finite numeric"
                    validate_verification(
                        explanation.get("verification"),
                        item_label,
                    )
                if disposition in {"unresolved", "supported"}:
                    unresolved_measurable = True
            else:
                assert disposition == "not_measurable", (
                    f"{item_label} non-measurable explanation must be not_measurable"
                )
                limitation = explanation.get("limitation")
                assert (
                    isinstance(limitation, str)
                    and limitation.strip()
                    and limitation.strip().lower() not in sentinel
                ), f"{item_label} not_measurable limitation is required"

        competing_text = [
            explanation.strip()
            for explanation in insight["competing_explanations"]
        ]
        assert sorted(assessed_text) == sorted(competing_text), (
            f"{label} diagnostic explanations must match competing_explanations"
        )
        if declaration == "measurable":
            assert all(measured_states), (
                f"{label} diagnostic measurability declaration is inconsistent"
            )
        elif declaration == "not_measurable":
            assert not any(measured_states), (
                f"{label} diagnostic measurability declaration is inconsistent"
            )
        else:
            assert any(measured_states) and not all(measured_states), (
                f"{label} mixed diagnostic measurability requires both states"
            )

        action_kind = insight["action"].get("kind")
        assert action_kind in {"diagnostic", "program"}, (
            f"{label} typed diagnostics require action kind"
        )
        if unresolved_measurable:
            assert insight["confidence"]["level"].strip().lower() != "high", (
                f"{label} unresolved measurable explanation forbids high confidence"
            )
            assert insight["priority"]["urgency"].strip().lower() != "critical", (
                f"{label} unresolved measurable explanation forbids critical urgency"
            )
            assert readiness == "investigate_first", (
                f"{label} unresolved measurable explanation requires investigate_first"
            )
            assert action_kind == "diagnostic", (
                f"{label} investigate_first requires a diagnostic action"
            )

    def validate_temporal_context(context, label):
        assert isinstance(context, dict), (
            f"{label} temporal_context must be structured"
        )
        for field in (
            "time_basis",
            "timezone",
            "requested_as_of",
            "data_as_of",
            "trustworthy_through",
            "partial_period_policy",
            "completeness_basis",
            "recency_status",
        ):
            assert isinstance(context.get(field), str) and context[field].strip(), (
                f"{label} temporal_context {field} is required"
            )
        try:
            requested_as_of = date.fromisoformat(context["requested_as_of"])
        except ValueError:
            raise AssertionError(
                f"{label} temporal_context requested_as_of must be an ISO date"
            ) from None
        try:
            source_timezone = ZoneInfo(context["timezone"])
        except ZoneInfoNotFoundError:
            raise AssertionError(
                f"{label} temporal_context timezone must be a valid IANA timezone"
            ) from None

        def parse_instant(field):
            text = context[field]
            try:
                instant = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                raise AssertionError(
                    f"{label} temporal_context {field} must be an ISO timestamp"
                ) from None
            assert instant.tzinfo is not None, (
                f"{label} temporal_context {field} must include a timezone"
            )
            return instant

        data_as_of = parse_instant("data_as_of")
        trustworthy_through = parse_instant("trustworthy_through")
        assert trustworthy_through <= data_as_of, (
            f"{label} temporal_context trustworthy_through exceeds data_as_of"
        )
        requested_exclusive = datetime.combine(
            requested_as_of + timedelta(days=1),
            datetime.min.time(),
            tzinfo=source_timezone,
        )
        assert data_as_of <= requested_exclusive, (
            f"{label} temporal_context data_as_of exceeds requested_as_of"
        )
        assert context["partial_period_policy"] in {"exclude", "include_flagged"}, (
            f"{label} temporal_context partial_period_policy is invalid"
        )
        status = context["recency_status"]
        current_statuses = {"current_change", "current_level", "persistent"}
        assert status in current_statuses | {
            "recurring_seasonal",
            "historical",
            "stale",
            "not_applicable",
        }, f"{label} temporal_context recency_status is invalid"
        supports_current_action = context.get("supports_current_action")
        assert isinstance(supports_current_action, bool), (
            f"{label} temporal_context supports_current_action must be boolean"
        )
        assert supports_current_action == (status in current_statuses), (
            f"{label} temporal_context action support contradicts recency_status"
        )
        evidence_fingerprints = context.get("evidence_fingerprints")
        assert isinstance(evidence_fingerprints, dict), (
            f"{label} temporal_context evidence_fingerprints are required"
        )
        assert re.fullmatch(
            r"[0-9a-f]{64}",
            str(evidence_fingerprints.get("coverage", "")),
        ), f"{label} temporal_context coverage fingerprint is invalid"
        window_fingerprint = evidence_fingerprints.get("window")
        assert window_fingerprint is None or re.fullmatch(
            r"[0-9a-f]{64}",
            str(window_fingerprint),
        ), f"{label} temporal_context window fingerprint is invalid"
        fingerprint_payloads = context.get("evidence_fingerprint_payloads")
        assert isinstance(fingerprint_payloads, dict), (
            f"{label} temporal_context evidence_fingerprint_payloads are required"
        )

        def evidence_fingerprint(value):
            canonical = json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            encoded = (
                "fabric-rlm.analysis.fingerprint.v1\0" + canonical
            ).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()

        assert evidence_fingerprint(
            fingerprint_payloads.get("coverage")
        ) == evidence_fingerprints["coverage"], (
            f"{label} temporal_context coverage fingerprint does not match evidence"
        )
        coverage_payload = fingerprint_payloads.get("coverage")
        assert isinstance(coverage_payload, dict), (
            f"{label} temporal_context coverage evidence must be structured"
        )
        payload_requested_as_of = coverage_payload.get("requested_as_of")
        if payload_requested_as_of is None:
            payload_requested_as_of = (
                coverage_payload.get("diagnostics", {})
                .get("watermarks", {})
                .get("requested_as_of")
            )
        assert payload_requested_as_of == context["requested_as_of"], (
            f"{label} temporal_context coverage requested_as_of does not match"
        )
        window_payload = fingerprint_payloads.get("window")
        assert (
            window_fingerprint is None and window_payload is None
        ) or (
            window_fingerprint is not None
            and evidence_fingerprint(window_payload) == window_fingerprint
        ), f"{label} temporal_context window fingerprint does not match evidence"
        if status == "current_change":
            assert window_fingerprint is not None and isinstance(
                window_payload,
                dict,
            ), (
                f"{label} temporal_context current_change requires linked "
                "window evidence"
            )
            assert (
                window_payload.get("coverage_fingerprint")
                == evidence_fingerprints["coverage"]
            ), (
                f"{label} temporal_context window evidence does not reference "
                "coverage"
            )

        def period_start(value, grain):
            if grain == "day":
                return value
            if grain == "week":
                return value - timedelta(days=value.weekday())
            if grain == "month":
                return value.replace(day=1)
            return value.replace(
                month=3 * ((value.month - 1) // 3) + 1,
                day=1,
            )

        def next_period(value, grain):
            if grain == "day":
                return value + timedelta(days=1)
            if grain == "week":
                return value + timedelta(days=7)
            months = 1 if grain == "month" else 3
            month_index = value.year * 12 + value.month - 1 + months
            return date(month_index // 12, month_index % 12 + 1, 1)

        def period_label(value, grain):
            if grain == "day":
                return value.isoformat()
            if grain == "week":
                iso_year, iso_week, _ = value.isocalendar()
                return f"{iso_year}-W{iso_week:02d}"
            if grain == "month":
                return value.strftime("%Y-%m")
            return f"{value.year}-Q{((value.month - 1) // 3) + 1}"

        def validate_period(period, field, require_periods=False):
            assert isinstance(period, dict), (
                f"{label} temporal_context {field} must be structured"
            )
            assert period.get("grain") in {"day", "week", "month", "quarter"}, (
                f"{label} temporal_context {field} grain is invalid"
            )
            try:
                start = date.fromisoformat(period.get("start", ""))
                end = date.fromisoformat(period.get("end", ""))
            except ValueError:
                raise AssertionError(
                    f"{label} temporal_context {field} requires ISO dates"
                ) from None
            assert start <= end, (
                f"{label} temporal_context {field} start exceeds end"
            )
            grain = period["grain"]
            assert start == period_start(start, grain), (
                f"{label} temporal_context {field} start is not grain-aligned"
            )
            assert end == next_period(period_start(end, grain), grain) - timedelta(days=1), (
                f"{label} temporal_context {field} end is not grain-aligned"
            )
            periods = period.get("periods")
            if require_periods:
                assert (
                    isinstance(periods, list)
                    and periods
                    and all(isinstance(item, str) and item for item in periods)
                ), f"{label} temporal_context {field} periods are invalid"
                expected_periods = []
                cursor = start
                while cursor <= end:
                    assert len(expected_periods) < 10000, (
                        f"{label} temporal_context {field} period range is too large"
                    )
                    expected_periods.append(period_label(cursor, grain))
                    cursor = next_period(cursor, grain)
                assert periods == expected_periods, (
                    f"{label} temporal_context {field} periods do not match "
                    "the declared range"
                )
            return start, end

        if status in current_statuses:
            latest_complete = context.get("latest_complete_period")
            assert isinstance(latest_complete, dict), (
                f"{label} temporal_context current status requires "
                "latest_complete_period"
            )
            _, complete_end = validate_period(
                latest_complete,
                "latest_complete_period",
            )
            complete_exclusive = datetime.combine(
                complete_end + timedelta(days=1),
                datetime.min.time(),
                tzinfo=source_timezone,
            )
            assert trustworthy_through >= complete_exclusive, (
                f"{label} temporal_context latest_complete_period exceeds "
                "trustworthy coverage"
            )
        comparators = context.get("comparators")
        assert isinstance(comparators, list), (
            f"{label} temporal_context comparators must be a list"
        )
        if status == "current_change":
            current_window = context.get("current_window")
            assert isinstance(current_window, dict), (
                f"{label} temporal_context current_change requires current_window"
            )
            current_start, current_end = validate_period(
                current_window,
                "current_window",
                require_periods=True,
            )
            assert current_window["grain"] == latest_complete["grain"], (
                f"{label} temporal_context current_window grain must match "
                "latest_complete_period"
            )
            assert current_end == complete_end, (
                f"{label} temporal_context current_window must end at "
                "latest_complete_period"
            )
            assert comparators, (
                f"{label} temporal_context current_change requires a comparator"
            )
            assert window_payload.get("current_window") == current_window, (
                f"{label} temporal_context current_window does not match "
                "window evidence"
            )
            assert len(comparators) == 1, (
                f"{label} temporal_context current_change requires exactly "
                "one comparator"
            )
            assert window_payload.get("comparator") == comparators[0], (
                f"{label} temporal_context comparator does not match "
                "window evidence"
            )
            for comparator_index, comparator in enumerate(comparators, start=1):
                assert isinstance(comparator, dict), (
                    f"{label} temporal_context comparator {comparator_index} "
                    "must be structured"
                )
                assert isinstance(comparator.get("kind"), str) and comparator["kind"], (
                    f"{label} temporal_context comparator {comparator_index} "
                    "kind is required"
                )
                comparator_start, comparator_end = validate_period(
                    comparator,
                    f"comparator {comparator_index}",
                    require_periods=True,
                )
                assert comparator["grain"] == current_window["grain"], (
                    f"{label} temporal_context comparator {comparator_index} "
                    "grain must match current_window"
                )
                kind = comparator["kind"]
                assert kind in {
                    "previous_window",
                    "same_period_prior_year",
                }, (
                    f"{label} temporal_context comparator {comparator_index} "
                    "kind is invalid"
                )
                assert len(comparator["periods"]) == len(
                    current_window["periods"]
                ), (
                    f"{label} temporal_context comparator period count must "
                    "match current_window"
                )
                if kind == "previous_window":
                    assert comparator_end + timedelta(days=1) == current_start, (
                        f"{label} temporal_context previous_window comparator "
                        "must immediately precede current_window"
                    )
                else:
                    if current_window["grain"] == "week":
                        current_iso = current_start.isocalendar()
                        comparator_iso = comparator_start.isocalendar()
                        same_prior_period = (
                            comparator_iso.year == current_iso.year - 1
                            and comparator_iso.week == current_iso.week
                        )
                    else:
                        same_prior_period = (
                            comparator_start.month == current_start.month
                            and comparator_start.day == current_start.day
                            and comparator_start.year == current_start.year - 1
                        )
                    assert same_prior_period, (
                        f"{label} temporal_context same_period_prior_year "
                        "comparator is not calendar-aligned"
                    )
        complete_periods = context.get("complete_periods")
        assert isinstance(complete_periods, list) and all(
            isinstance(item, str) and item for item in complete_periods
        ), f"{label} temporal_context complete_periods must be a list of text"
        complete_set = set(complete_periods)
        persistence_periods = context.get("persistence_periods")
        seasonal_cycles = context.get("seasonal_cycles")
        assert isinstance(persistence_periods, list), (
            f"{label} temporal_context persistence_periods must be a list"
        )
        assert isinstance(seasonal_cycles, list), (
            f"{label} temporal_context seasonal_cycles must be a list"
        )
        if status == "persistent":
            assert (
                len(persistence_periods) >= 3
                and len(set(persistence_periods)) == len(persistence_periods)
                and set(persistence_periods) <= complete_set
            ), (
                f"{label} temporal_context persistent status requires at "
                "least three distinct complete persistence_periods"
            )
        if status == "recurring_seasonal":
            assert len(seasonal_cycles) >= 2, (
                f"{label} temporal_context recurring_seasonal status requires "
                "at least two seasonal_cycles"
            )
            normalized_cycles = []
            used_periods = set()
            for cycle_index, cycle in enumerate(seasonal_cycles, start=1):
                assert (
                    isinstance(cycle, list)
                    and cycle
                    and len(set(cycle)) == len(cycle)
                    and set(cycle) <= complete_set
                ), (
                    f"{label} temporal_context seasonal cycle {cycle_index} "
                    "must contain distinct complete periods"
                )
                normalized = tuple(cycle)
                assert normalized not in normalized_cycles, (
                    f"{label} temporal_context seasonal_cycles must be distinct"
                )
                assert not (used_periods & set(cycle)), (
                    f"{label} temporal_context seasonal_cycles must not overlap"
                )
                normalized_cycles.append(normalized)
                used_periods.update(cycle)
        return status, supports_current_action

    sentinel = {"none", "n/a", "na", "unknown", "not applicable", "null"}
    contract_version = payload.get("contract_version", 1)
    assert (
        isinstance(contract_version, int)
        and not isinstance(contract_version, bool)
        and contract_version in {1, 2, 3}
    ), "contract_version is unsupported"
    analysis_plan = payload.get("analysis_plan")
    assert isinstance(analysis_plan, dict), "analysis_plan is required"
    business_context = analysis_plan.get("business_context")
    assert isinstance(business_context, str) and business_context.strip(), (
        "analysis_plan business_context is required"
    )
    kpi_map = analysis_plan.get("kpi_map")
    assert isinstance(kpi_map, list) and kpi_map, (
        "analysis_plan kpi_map is required"
    )
    computability_levels = {
        "computable", "partially_computable", "not_computable",
    }
    for kpi_index, kpi_entry in enumerate(kpi_map, start=1):
        assert isinstance(kpi_entry, dict), (
            f"analysis_plan KPI {kpi_index} must be structured"
        )
        assert isinstance(kpi_entry.get("kpi"), str) and kpi_entry["kpi"].strip(), (
            f"analysis_plan KPI {kpi_index} name is required"
        )
        assert kpi_entry.get("computability") in computability_levels, (
            f"analysis_plan KPI {kpi_index} computability is invalid"
        )
        reason = kpi_entry.get("reason")
        assert (
            isinstance(reason, str)
            and reason.strip()
            and reason.strip().lower() not in sentinel
        ), f"analysis_plan KPI {kpi_index} reason is required"
    assert any(
        entry["computability"] in {"computable", "partially_computable"}
        for entry in kpi_map
    ), "analysis_plan must identify at least one computable KPI"

    search_space = analysis_plan.get("search_space")
    assert isinstance(search_space, dict), (
        "analysis_plan search_space is required"
    )
    for field in (
        "dimensions_available", "dimensions_deferred",
        "time_grains_available", "populations",
    ):
        values = search_space.get(field)
        assert isinstance(values, list), (
            f"analysis_plan search_space {field} must be a list"
        )
        if field != "dimensions_deferred":
            assert all(
                isinstance(value, str) and value.strip() for value in values
            ), f"analysis_plan search_space {field} must be a list of text"
    assert search_space["populations"], (
        "analysis_plan search_space populations must be non-empty"
    )

    insights = payload.get("insights")
    assert isinstance(insights, list) and insights, "insights must be a non-empty list"
    seen = set()
    causal = re.compile(
        r"\b(caused|causes|driven by|due to|resulted in|leads? to|"
        r"led to|drives?|drove|triggered|because(?: of)?|leading indicator|"
        r"signals?|implies|confirms?|flowing into|behind)\b",
        re.I,
    )
    disclaimer = re.compile(
        r"\b(cannot|does not|did not|insufficient to|unable to)\b.{0,40}"
        r"\b(establish|prove|infer|determine)\b.{0,40}\b(caus|whether)\b",
        re.I,
    )
    negated_causal = re.compile(
        r"\b(?:not|never)\b.{0,24}\b(?:caused|causes|driven by|due to|"
        r"resulted in|leads? to|led to|drives?|drove|triggered|because(?: of)?|"
        r"leading indicator|signals?|implies|confirms?|flowing into|behind)\b",
        re.I,
    )

    for index, insight in enumerate(insights, start=1):
        assert isinstance(insight, dict), f"insight {index} must be an object"
        for field in (
            "title", "statement", "interpretation", "competing_explanations",
            "action", "priority", "confidence", "limitations",
            "supporting_claims", "discovery", "evidence_tier",
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
        assert len(re.findall(r"[.!?](?:\s|$)", statement)) <= 1, (
            f"insight {index} statement must contain one primary measured claim"
        )
        assert ";" not in statement, (
            f"insight {index} statement must contain one primary measured claim"
        )
        packed_claim = re.search(
            r"(?:\d|%|\$).{0,160}\b(?:while|whereas|and|as|with|alongside)\b"
            r".{0,160}(?:\d|%|\$)",
            statement,
            flags=re.I,
        )
        assert not packed_claim, (
            f"insight {index} statement must contain one primary measured claim"
        )
        interpretation_without_periods = re.sub(
            r"\b(?:19|20)\d{2}-\d{2}(?:-\d{2})?\b|"
            r"\b(?:in|during|since|through|year|fy)\s+(?:19|20)\d{2}\b|"
            r"\b(?:19|20)\d{2}(?:\s+\w+){0,2}\s+"
            r"(?:cohort|period|year|quarter)\b|\bQ[1-4]\b",
            " ",
            interpretation,
            flags=re.I,
        )
        assert not re.search(
            r"\d+(?:[,.]\d+)*\s*%?|\$\s*\d",
            interpretation_without_periods,
        ), (
            f"insight {index} quantitative facts belong in verified supporting_claims"
        )
        assert not re.search(
            r"\b(?:"
            r"half\s+(?:of\s+)?(?:the\s+)?|"
            r"a\s+(?:third|quarter)\s+of\s+|"
            r"one\s+in\s+(?:two|three|four|five|six|seven|eight|nine|ten)\b|"
            r"(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+dozen\b"
            r")",
            interpretation_without_periods,
            flags=re.I,
        ), (
            f"insight {index} quantitative facts belong in verified supporting_claims"
        )
        title_fingerprint = re.sub(r"\W+", " ", title.lower()).strip()
        statement_fingerprint = re.sub(r"\W+", " ", statement.lower()).strip()
        assert title_fingerprint not in seen, f"duplicate insight title: {title}"
        assert statement_fingerprint not in seen, f"duplicate insight statement: {title}"
        seen.update((title_fingerprint, statement_fingerprint))
        current_claim_text = " ".join(
            [
                title,
                statement,
                interpretation,
                *collect_strings(insight.get("supporting_claims", [])),
                *collect_strings(insight.get("action", {})),
            ]
        )
        current_claim = re.search(
            r"\b(current|currently|latest|recent|today|"
            r"this\s+(?:day|week|month|quarter|year))\b",
            current_claim_text,
            flags=re.I,
        )
        temporal_context = insight.get("temporal_context")
        if current_claim:
            assert temporal_context is not None, (
                f"insight {index} current claim requires temporal_context"
            )
        temporal_status = None
        supports_current_action = None
        if temporal_context is not None:
            temporal_status, supports_current_action = validate_temporal_context(
                temporal_context,
                f"insight {index}",
            )
            if current_claim:
                assert temporal_status in {
                    "current_change",
                    "current_level",
                    "persistent",
                }, (
                    f"insight {index} current claim is not supported by its "
                    "temporal status"
                )

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

        evidence_tier = insight["evidence_tier"]
        assert evidence_tier in {"descriptive", "associational", "causal"}, (
            f"insight {index} evidence tier is invalid"
        )
        if unsupported_causal:
            assert evidence_tier == "causal", (
                f"insight {index} causal language requires causal evidence tier"
            )
        if evidence_tier == "causal":
            causal_evidence = insight.get("causal_evidence")
            assert isinstance(causal_evidence, dict), (
                f"insight {index} causal evidence is required for causal tier"
            )
            for field in ("design", "result", "limitations"):
                value = str(causal_evidence.get(field, "")).strip()
                assert value and value.lower() not in sentinel, (
                    f"insight {index} causal evidence missing {field}"
                )
        overclaim = re.search(
            r"\b(proven|proves?|definitive(?:ly)?|conclusive(?:ly)?)\b",
            f"{statement} {interpretation}",
            flags=re.I,
        )
        assert not overclaim or evidence_tier == "causal", (
            f"insight {index} overclaiming language exceeds its evidence tier"
        )

        action = insight["action"]
        assert isinstance(action, dict), f"insight {index} action must be structured"
        for field in ("owner", "segment", "decision", "target", "time_horizon"):
            assert isinstance(action.get(field), str) and action[field].strip(), (
                f"insight {index} action missing {field}"
            )
        if action.get("kind") == "program":
            assert temporal_context is not None, (
                f"insight {index} program action requires temporal_context"
            )
        if (
            temporal_context is not None
            and action.get("kind") == "program"
            and not supports_current_action
        ):
            raise AssertionError(
                f"insight {index} {temporal_status} temporal evidence "
                "cannot support a program action"
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
        confidence_reason = confidence["reason"].strip()
        assert not re.fullmatch(
            r"(?:based on\s+)?(?:the\s+)?(?:available\s+|observed\s+)?"
            r"(?:data|analysis|results?)[.!]?",
            confidence_reason,
            flags=re.I,
        ), f"insight {index} confidence reason must name its evidence basis"
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

        has_diagnostic_declaration = "diagnostic_measurability" in insight
        has_diagnostic_assessment = "diagnostic_assessment" in insight
        if contract_version >= 2:
            assert has_diagnostic_declaration and has_diagnostic_assessment, (
                f"insight {index} current contract requires typed diagnostic fields"
            )
        assert has_diagnostic_declaration == has_diagnostic_assessment, (
            f"insight {index} diagnostic_assessment and "
            "diagnostic_measurability must be provided together"
        )
        if has_diagnostic_declaration:
            validate_diagnostic_assessment(insight, f"insight {index}")

        if "metric_spec" in insight:
            validate_metric_spec(insight["metric_spec"], f"insight {index}")
        else:
            assert "verification" in insight, f"insight {index} missing verification"
            validate_verification(insight["verification"], f"insight {index}")

        supporting_claims = insight["supporting_claims"]
        assert isinstance(supporting_claims, list), (
            f"insight {index} supporting_claims must be a list"
        )
        for claim_index, claim in enumerate(supporting_claims, start=1):
            assert isinstance(claim, dict), (
                f"insight {index} supporting claim {claim_index} must be structured"
            )
            assert isinstance(claim.get("claim"), str) and claim["claim"].strip(), (
                f"insight {index} supporting claim {claim_index} needs claim text"
            )
            assert "expected_value" in claim and claim["expected_value"] is not None, (
                f"insight {index} supporting claim {claim_index} expected_value "
                "is required"
            )
            if "metric_spec" in claim:
                validate_metric_spec(
                    claim["metric_spec"],
                    f"insight {index} supporting claim {claim_index}",
                    claim["expected_value"],
                )
            else:
                validate_verification(
                    claim.get("verification"),
                    f"insight {index} supporting claim {claim_index}",
                )

        readiness = (
            insight["diagnostic_assessment"].get("decision_readiness")
            if has_diagnostic_assessment
            else None
        )
        if readiness == "investigate_first":
            assert not re.search(
                r"\baffects?\b"
                r"|(?<!usb )(?<!hard )(?<!disk )(?<!flash )"
                r"\bdrives?\b(?![- ]time\b)"
                r"|\bcauses?\b(?!\s+(?:for|analysis|investigation)\b)"
                r"|\bcaused\b|\bfailure tail\b|\broot cause\b",
                title,
                flags=re.I,
            ), f"insight {index} title overclaims investigate-first evidence"
            assert not re.search(
                r"primary lever|intervention target|will improve|failure tail"
                r"|planning baseline|economics (?:are|is) dominated"
                r"|\b(?:is|are) dominated by\b|\bwould (?:over|under)shoot\b"
                r"|\banchor(?:s|ed|ing)? capacity planning\b"
                r"|\bmost associated\b"
                r"|\b(?:revenue|growth|outcomes?|performance|retention|conversion"
                r"|sales|demand|profitability|economics?) depends on\b"
                r"|\bbehaves as (?:a )?one[- ]shot\b"
                r"|\bacquisition economics dominate\b"
                r"|\bconcentrat(?:e|es) leverage\b",
                interpretation,
                flags=re.I,
            ), f"insight {index} interpretation overclaims investigate-first evidence"

        assert not re.fullmatch(
            r"\s*the primary measured (?:difference|count|rate|share|value)"
            r" is [-+]?\d[\d,]*(?:\.\d+)?%?\.?\s*",
            statement,
            flags=re.I,
        ), f"insight {index} statement lacks decision context"

        if re.search(r"\b(?:step[- ]change|level shift)\b", title, flags=re.I):
            assert re.search(
                r"\b(?:possible|potential|candidate|signal|suggests?|may)\b",
                title,
                flags=re.I,
            ), (
                f"insight {index} unverified level shift title must say "
                "possible or candidate"
            )

        supporting_text = " ".join(collect_strings(supporting_claims))
        if re.search(
            r"heav(?:y|ily) (?:right[- ]skewed|tail)|long (?:failure )?tail",
            f"{title} {interpretation}",
            flags=re.I,
        ):
            assert re.search(
                r"\bp(?:95|99)\b|\b0\.(?:95|99)\b",
                supporting_text,
                flags=re.I,
            ), f"insight {index} heavy-tail language requires P95 or P99 evidence"

        governed_basis = bool(re.search(
            r"\b(?:approved|governed|policy[- ]owned|contractual|sla)\b"
            r".{0,40}\b(?:benchmark|target|threshold)\b"
            r"|\b(?:benchmark|target|threshold)\b.{0,40}"
            r"\b(?:approved|governed|policy[- ]owned|contractual|sla)\b",
            supporting_text,
            flags=re.I,
        ))

        if re.search(
            r"\b(?:extreme|severe|unusually|exceptionally|nearly absent|rare"
            r"|heav(?:y|ily))\b",
            title,
            flags=re.I,
        ) or re.search(r"\bhigh\b.{0,30}\bconcentrat", title, flags=re.I):
            assert governed_basis, (
                f"insight {index} qualitative severity requires a governed benchmark"
            )

        if re.search(r"\bconcentrat", title, flags=re.I):
            assert re.search(
                r"\b(?:distinct|total|active|eligible)\s+\w*\s*"
                r"(?:seller|customer|account|product|supplier|franchise)s?\b",
                supporting_text,
                flags=re.I,
            ) and re.search(
                r"\b\d[\d,]*(?:\.\d+)?\s+(?:of|out of)\s+"
                r"\d[\d,]*(?:\.\d+)?\b",
                title,
                flags=re.I,
            ), (
                f"insight {index} concentration requires its eligible population "
                "in its title"
            )
            if insight["action"]["kind"] == "program":
                assert governed_basis, (
                    f"insight {index} concentration program requires a governed threshold"
                )

        assessment_text = " ".join(
            collect_strings(insight.get("diagnostic_assessment", {}))
        )
        if re.search(
            r"\bunder[- ]?merg(?:e|es|ed|ing)\b.{0,100}"
            r"\btrue\b.{0,30}\b(?:higher|larger|greater)\b",
            assessment_text,
            flags=re.I,
        ):
            assert not re.search(r"\bupper bound\b", interpretation, flags=re.I), (
                f"insight {index} has reversed bound direction"
            )
        if re.search(
            r"\bover[- ]?merg(?:e|es|ed|ing)\b.{0,100}"
            r"\btrue\b.{0,30}\b(?:lower|smaller|less)\b",
            assessment_text,
            flags=re.I,
        ):
            assert not re.search(r"\blower bound\b", interpretation, flags=re.I), (
                f"insight {index} has reversed bound direction"
            )

        discovery = insight["discovery"]
        assert isinstance(discovery, dict), (
            f"insight {index} discovery must be structured"
        )
        assert discovery.get("pattern_type") in {
            "portfolio_trend", "subgroup", "cohort_transition", "interaction",
        }, f"insight {index} discovery pattern_type is invalid"
        assert (
            isinstance(discovery.get("dimensions_tested"), list)
            and all(
                isinstance(item, str) and item.strip()
                for item in discovery["dimensions_tested"]
            )
        ), f"insight {index} discovery dimensions_tested are required"
        if search_space["dimensions_available"]:
            assert discovery["dimensions_tested"], (
                f"insight {index} discovery dimensions_tested are required"
            )
        assert isinstance(discovery.get("population"), str) and discovery["population"].strip(), (
            f"insight {index} discovery population is required"
        )
        assert (
            isinstance(discovery.get("sample_size"), int)
            and not isinstance(discovery["sample_size"], bool)
            and discovery["sample_size"] > 0
        ), f"insight {index} discovery sample_size must be a positive integer"
        assert (
            isinstance(discovery.get("robustness_checks"), list)
            and discovery["robustness_checks"]
            and all(
                isinstance(item, str) and item.strip()
                for item in discovery["robustness_checks"]
            )
        ), f"insight {index} discovery robustness_checks are required"
        robustness_text = " ".join(discovery["robustness_checks"]).lower()
        if discovery["pattern_type"] == "cohort_transition":
            assert re.search(
                r"\b(denominator|population|composition|entry|mix)\b",
                robustness_text,
            ), (
                f"insight {index} cohort transition requires a denominator "
                "or population-composition robustness check"
            )
        if discovery["pattern_type"] == "interaction":
            assert len(set(discovery["dimensions_tested"])) >= 2, (
                f"insight {index} interaction must test at least two dimensions"
            )
            interaction_evidence = discovery.get("interaction_evidence")
            assert isinstance(interaction_evidence, dict), (
                f"insight {index} interaction requires effect heterogeneity evidence"
            )
            cells = interaction_evidence.get("cells")
            assert isinstance(cells, list) and len(cells) >= 2, (
                f"insight {index} interaction requires at least two evidence cells"
            )
            effects = []
            for cell_index, cell in enumerate(cells, start=1):
                assert isinstance(cell, dict), (
                    f"insight {index} interaction cell {cell_index} must be structured"
                )
                assert isinstance(cell.get("cell"), str) and cell["cell"].strip(), (
                    f"insight {index} interaction cell {cell_index} label is required"
                )
                effect = cell.get("effect")
                assert (
                    isinstance(effect, (int, float))
                    and not isinstance(effect, bool)
                ), f"insight {index} interaction cell {cell_index} effect must be numeric"
                effects.append(float(effect))
                assert (
                    isinstance(cell.get("sample_size"), int)
                    and not isinstance(cell["sample_size"], bool)
                    and cell["sample_size"] > 0
                ), (
                    f"insight {index} interaction cell {cell_index} sample_size "
                    "must be a positive integer"
                )
            assert len(set(effects)) >= 2, (
                f"insight {index} interaction cell effects must differ"
            )
            heterogeneity = interaction_evidence.get("heterogeneity")
            assert (
                isinstance(heterogeneity, str)
                and heterogeneity.strip()
                and heterogeneity.strip().lower() not in sentinel
            ), f"insight {index} interaction heterogeneity explanation is required"
            baseline_effect = interaction_evidence.get("baseline_effect")
            assert (
                isinstance(baseline_effect, (int, float))
                and not isinstance(baseline_effect, bool)
            ), f"insight {index} interaction baseline_effect must be numeric"

    if len(insights) >= 8:
        pattern_types = {
            insight["discovery"]["pattern_type"] for insight in insights
        }
        assert len(pattern_types) >= 3, (
            "deep analysis pattern diversity requires at least three pattern types"
        )

    candidates = payload.get("candidates")
    assert isinstance(candidates, list) and candidates, (
        "candidates ledger is required"
    )
    available_dimensions = set(search_space["dimensions_available"])
    insight_titles = {insight["title"] for insight in insights}
    insight_by_title = {insight["title"]: insight for insight in insights}
    promoted_titles = set()
    for candidate_index, candidate in enumerate(candidates, start=1):
        assert isinstance(candidate, dict), (
            f"candidate {candidate_index} must be structured"
        )
        assert isinstance(candidate.get("candidate"), str) and candidate["candidate"].strip(), (
            f"candidate {candidate_index} name is required"
        )
        assert (
            isinstance(candidate.get("dimensions_tested"), list)
            and all(
                isinstance(item, str) and item.strip()
                for item in candidate["dimensions_tested"]
            )
        ), f"candidate {candidate_index} dimensions_tested must be a list"
        assert candidate.get("disposition") in {"promoted", "rejected"}, (
            f"candidate {candidate_index} disposition is invalid"
        )
        reason = candidate.get("reason")
        assert (
            isinstance(reason, str)
            and reason.strip()
            and reason.strip().lower() not in sentinel
        ), f"candidate {candidate_index} reason is required"
        if candidate["disposition"] == "promoted":
            promoted_as = candidate.get("promoted_as")
            assert promoted_as in insight_titles, (
                f"candidate {candidate_index} promoted_as references an unknown insight"
            )
            assert promoted_as not in promoted_titles, (
                f"candidate {candidate_index} duplicates a promoted insight"
            )
            insight_dimensions = set(
                insight_by_title[promoted_as]["discovery"]["dimensions_tested"]
            )
            candidate_dimensions = set(candidate["dimensions_tested"])
            assert insight_dimensions == candidate_dimensions, (
                f"insight {promoted_as!r} discovery dimensions do not match "
                "promoted candidate"
            )
            promoted_titles.add(promoted_as)
        else:
            assert candidate.get("promoted_as") in {None, ""}, (
                f"rejected candidate {candidate_index} cannot be promoted"
            )
            rejection_type = candidate.get("rejection_type")
            assert rejection_type in {
                "quantitative", "not_computable", "redundant",
            }, f"rejected candidate {candidate_index} rejection_type is invalid"
            if rejection_type == "quantitative":
                evidence = candidate.get("rejection_evidence")
                assert isinstance(evidence, dict), (
                    f"candidate {candidate_index} quantitative rejection evidence "
                    "is required"
                )
                expected_components = {
                    "effect_value": evidence.get("effect_value"),
                    "baseline_value": evidence.get("baseline_value"),
                    "sample_size": evidence.get("sample_size"),
                }
                assert all(
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                    for value in expected_components.values()
                ), (
                    f"candidate {candidate_index} quantitative rejection evidence "
                    "must include numeric effect, baseline, and sample size"
                )
                assert evidence["sample_size"] > 0, (
                    f"candidate {candidate_index} quantitative rejection evidence "
                    "sample_size must be positive"
                )
                rejection_verification = evidence.get("verification")
                validate_verification(
                    rejection_verification,
                    f"candidate {candidate_index} quantitative rejection",
                )
                components = rejection_verification.get("components")
                assert isinstance(components, list) and components, (
                    f"candidate {candidate_index} quantitative rejection evidence "
                    "components are required"
                )
                components_by_name = {
                    component.get("name"): component
                    for component in components
                    if isinstance(component, dict)
                }
                assert set(expected_components) <= set(components_by_name), (
                    f"candidate {candidate_index} quantitative rejection evidence "
                    "must verify effect, baseline, and sample size"
                )
                for component_name, expected_value in expected_components.items():
                    component = components_by_name[component_name]
                    assert component.get("expected_value") == expected_value, (
                        f"candidate {candidate_index} quantitative rejection "
                        f"{component_name} does not match its verified component"
                    )
                    component_verification = component.get("verification")
                    if not isinstance(component_verification, dict):
                        component_verification = {
                            "method": rejection_verification["method"],
                            "expression": component.get("expression"),
                            "sources": component.get("sources"),
                        }
                    validate_verification(
                        component_verification,
                        (
                            f"candidate {candidate_index} quantitative rejection "
                            f"{component_name}"
                        ),
                    )
            elif rejection_type == "not_computable":
                missing_fields = candidate.get("missing_fields")
                assert (
                    isinstance(missing_fields, list)
                    and missing_fields
                    and all(
                        isinstance(field, str) and field.strip()
                        for field in missing_fields
                    )
                ), (
                    f"candidate {candidate_index} not_computable rejection must "
                    "identify missing_fields"
                )
                incorrectly_missing = available_dimensions & set(missing_fields)
                assert not incorrectly_missing, (
                    f"candidate {candidate_index} not_computable rejection names "
                    f"an available field: {sorted(incorrectly_missing)}"
                )
            else:
                duplicate_of = candidate.get("duplicate_of")
                assert duplicate_of in insight_titles, (
                    f"candidate {candidate_index} redundant rejection must identify "
                    "a promoted insight in duplicate_of"
                )
    missing_promotions = insight_titles - promoted_titles
    assert not missing_promotions, (
        f"insight not promoted from a candidate: {sorted(missing_promotions)}"
    )

    deferred_dimensions = set()
    for deferred_index, deferred in enumerate(
        search_space["dimensions_deferred"],
        start=1,
    ):
        assert isinstance(deferred, dict), (
            f"deferred dimension {deferred_index} must be structured"
        )
        dimension = deferred.get("dimension")
        reason = deferred.get("reason")
        assert isinstance(dimension, str) and dimension.strip(), (
            f"deferred dimension {deferred_index} name is required"
        )
        assert (
            isinstance(reason, str)
            and reason.strip()
            and reason.strip().lower() not in sentinel
        ), f"deferred dimension {deferred_index} reason is required"
        assert dimension in available_dimensions, (
            f"deferred dimension {dimension} is not available"
        )
        assert dimension not in deferred_dimensions, (
            f"deferred dimension {dimension} is duplicated"
        )
        deferred_dimensions.add(dimension)

    tested_dimensions = {
        dimension
        for insight in insights
        for dimension in insight["discovery"]["dimensions_tested"]
    }
    tested_dimensions.update(
        dimension
        for candidate in candidates
        for dimension in candidate["dimensions_tested"]
    )
    unsearched_dimensions = (
        available_dimensions - tested_dimensions - deferred_dimensions
    )
    assert not unsearched_dimensions, (
        f"unsearched dimension: {sorted(unsearched_dimensions)}"
    )
```

## Tripwires

- Run `3c5f29ba59e449a3aa04c9eff8f06c33` passed because its SQL selected model-supplied constants instead of recomputing metrics.
- An all-time active-status share was labeled churn even though no cohort transition or time-based churn denominator was calculated.
- Aggregate trends were presented without searching interactions or subgroups that could reverse the conclusion.
- An aggregate change was accepted without separating volume, per-unit rate,
  and population-mix effects.
- A retention or engagement rate silently excluded churned or lapsed members,
  so it measured survivors rather than the original cohort.
- Interpretations used `driven by` and `caused` when the data supported association only.
- Generic actions such as "monitor closely" passed despite having no owner, target, or time horizon.

## Invariants

- `insights` is a non-empty list and every item satisfies the output contract.
- Baseline submissions use `contract_version: 2`; evidence-closure submissions
  use version `3`; omitted version is a legacy-only compatibility path, and
  unsupported versions are rejected.
- Every verification expression computes `metric_value` or `current_value` from
  a declared source; selecting or assigning constants is invalid.
- Derived metric arithmetic is deterministic over independently verified,
  finite numeric components; zero denominators and booleans are invalid.
- Cross-period counts prove denominator integrity with verified denominators
  and recomputed rates, or equal independently verified denominators for a
  stable/exhaustive population.
- Decomposition contributions plus an explicit residual reconcile to the
  independently verified total delta.
- The host independently executes verification and matches metric, comparison,
  period, population, denominator, units, and sample size.
- Affirmative causal language requires structured causal evidence; explicit
  causal disclaimers do not.
- Typed diagnostic assessments exactly cover declared competing explanations.
  Tested dispositions have independent verification; unresolved measurable
  alternatives gate confidence, urgency, readiness, and action kind.
- Every action has an owner, segment, decision, measurable target, and time horizon.
- Every secondary quantitative fact is an independently verified supporting
  claim; interpretation contains no hidden numeric evidence.
- Discovery provenance records the tested dimensions, population, sample size,
  pattern type, and robustness checks.
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

Start with business-defined dimensions present in the source before creating
derived groupings. For every aggregate change, separately test whether it is
explained by population volume, per-unit rate, or subgroup mix. Before
accepting an aggregate conclusion, test at least two relevant dimensions for
subgroup reversals and record them in `discovery.dimensions_tested`.

Before querying, submit the source-derived analysis plan. Test or explicitly
defer every available dimension; a deferral requires a concrete source
limitation. Keep a candidate ledger throughout the search, including candidates
that were rejected for immateriality, instability, duplication, weak evidence,
or lack of actionability. The ledger is an audit trail, not a quota.

### 4. Select non-redundant candidates

Score materiality, surprise, persistence, confidence, actionability, novelty,
and strategic relevance. Apply a non-redundancy check and reject obvious chart
captions, tiny immaterial groups, and several variants of the same signal.
Promote only candidates that survive the applicable robustness checks, and
link each promoted candidate to exactly one final insight.

### 5. Verify independently

Recompute each candidate through the bound source using SQL, DAX, bounded API
aggregation, or code over a bound file. Prefer rates, shares, changes, and
reconciled decompositions over raw counts when populations can vary. Check source lineage, join
multiplicity, null treatment, numerator, denominator, population, period,
units, and sample size. Reconcile against an alternate calculation where
practical. Portable SQL verification uses one flat aggregate `SELECT` per component;
split a complex metric into simple component queries and let the portable
verifier combine their expected values while the host verifies every source
result. For cross-period counts, verify denominators and rates unless the
population is stable or exhaustive; in that case, verify both period
denominators and require them to be equal. Run the required verifier and the
host equality checks before submission.

### 6. Calibrate interpretation and action

Keep measured fact, interpretation, competing explanations, causal evidence,
action, confidence, and limitations separate. Baseline runs emit
`contract_version: 2`; evidence-closure runs emit version `3`. Both declare
diagnostic measurability and use the typed assessment for every insight.
Version `1` omission remains valid only for existing-consumer compatibility.
Prefer "is associated with," "is consistent
with," or "may reflect" unless the design supports causality.

Do not promote averages-only findings. Group comparisons must report explicit
group sizes and denominators, median and distribution tails, and sensitivity to
skew. Test censoring, selection effects, obvious confounders, and comparable
populations. Normalize accumulating lifecycle or activity measures by exposure;
when these checks are unavailable or materially unresolved, use
`investigate_first` with a diagnostic action or reject the candidate.

Calibrate every title and interpretation to readiness. Investigate-first
language must remain observational. A concentration title states its total
eligible population; a heavy-tail claim includes P95 or P99; and a level-shift
or step-change title says `possible` or `candidate` unless a formal change-point
diagnostic, explicit windows, and partial-period policy are verified.
Qualitative severity such as `extreme`, `severe`, `unusually high`,
`nearly absent`, `rare`, `high concentration`, or `heavy` requires an approved
benchmark or governed threshold. Statements must retain decision context and
must not collapse into opaque forms such as `the primary measured difference
is X`.
Concentration-based program actions require a governed threshold, and the title
states the selected count out of the total eligible population. Bound wording
must follow the competing explanation: under-merging that can hide repeat
entities makes the observed rate a lower bound, not an upper bound.

### 7. Synthesize

Group findings into two to four decision themes. Explain how signals reinforce
or contradict one another, what the aggregate masked, the best-fitting and
competing explanations, the decision justified now, and the next analysis that
would discriminate between explanations. Synthesis must add information beyond
concatenating statements.
