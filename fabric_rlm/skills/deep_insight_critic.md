---
applies_when:
  keywords:
    - adversarial analytics critic
    - adversarial review
    - insight critic
    - critique insights
    - challenge analytics
    - decision quality review
    - audit insight recommendations
  output_fields:
    - reviewed_insights
    - synthesis_manifest
excludes: []
depends_on: []
specificity: domain
---
# deep_insight_critic

Summary: Adversarial, source-agnostic criticism of discovery outputs that produces an exhaustive, machine-enforceable decision disposition ledger.

## Purpose and boundary

This is the critic between search and synthesis:

`search -> adversarial critic -> synthesis`

It challenges whether findings deserve to affect a decision. It does not
stylistically rewrite findings or re-run the discovery verifier. This phase
does not synthesize prose. It cannot repair evidence by assertion. Semantic truth remains the
responsibility of source-aware host/model analysis; the portable verifier
enforces completeness, provenance, and legal state transitions.

## Critic procedure

1. **Inspect the source payload.** Bind the exact contract-version 2 discovery
   payload with the orchestration-supplied fingerprint and inventory. Trace
   every insight by exact title and rank.
2. **Independently test** cross-insight tensions, denominators, metric
   definitions and populations, targets and benchmarks, grains and joins,
   instrumentation alternatives, and competing explanations. Evidence may be
   SQL, Python, notebook cells, files, source-payload paths, or other
   independently addressable checks.
3. **Challenge obviousness** relative to the decision baseline and ask whether
   cross-domain depth, actionability, or a materially different decision is
   present. Do not reward a well-audited inventory of univariate restatements.
4. **Challenge statistical and operational fitness.** Reject averages-only
   findings unless the denominator, sample size, median or distribution tail,
   skew sensitivity, and relevant censoring are addressed or explicitly gated.
   Test obvious confounders and population comparability. Require
   exposure-normalized comparisons for measures that accumulate with lifecycle
   age or time at risk. Treat model-proposed thresholds and external benchmarks
   as ungoverned until an approved source is evidenced.
5. **Submit** the self-contained critic contract only after every source
   insight and every taxonomy check has a disposition.

## Output contract: `critic_version: 1`

Submit a dictionary with:

- `critic_version: 1`, `source_contract_version` (`2` or `3`), and a non-empty
  orchestration-supplied `source_fingerprint`.
- `source_inventory`: the exact source insight `title` and positive integer
  `rank`, each once. Optional `action_kind` is `program` or `diagnostic`;
  optional `decision_readiness` is `act_ready` or `investigate_first`.
- `reviewed_insights`: exactly one record per inventory entry, in inventory
  order. Each has exact `title` and `rank`; `verdict` (`approve`, `revise`, or
  `reject`); substantive `decision_effect`; at least one `challenge`;
  `required_changes`; boolean `synthesis_eligible`; and `resolutions`.

Each challenge has a globally unique stable `id`, a `type`, substantive
`assessment`, `severity` (`blocking`, `material`, or `minor`), and unique,
non-empty `evidence_refs`. Insight challenge types are:

`obviousness`, `cross_domain_depth`, `contradiction`,
`denominator_integrity`, `metric_definition`, `alternative_explanation`,
`target_basis`, `benchmark_basis`, `causal_overclaim`, `grain_or_join`,
`headline_consistency`, and `actionability`.

The high-risk types `contradiction`, `denominator_integrity`,
`grain_or_join`, `headline_consistency`, and `causal_overclaim` may not be
minor. Approval must be earned by a recorded challenge, has no required
changes or blocking challenge, and is synthesis eligible. Revision requires
one or more `required_changes`; rejection requires a substantive rejection
change, is never synthesis eligible, and cannot enter synthesis.

A required change is `{"change": <substantive string>, "gate": "none" |
"investigate_first"}`. A resolution binds `challenge_index` (zero-based) and
the matching `challenge_type` to `status` (`resolved`, `downgraded`, or
`gated`), substantive `rationale`, and evidence. Only one resolution may bind
a challenge. Blocking challenges cannot be downgraded. A synthesis-eligible
revision must resolve every material/blocking challenge or explicitly gate
the unresolved work with an `investigate_first` required change. `gated`
means investigate-first/diagnostic-only: it cannot preserve a program action.

Material or blocking revisions must be executable as bounded follow-up
analyses or explicitly identified as unavailable in the frozen source. Distinct
challenges require distinct evidence unless the critic explicitly consolidates
them. A zero count does not prove that a requested comparator period exists;
missing periods, fields, denominators, and populations remain unresolvable and
cannot support synthesis.

- `portfolio_challenges`: cross-insight challenges with the same challenge
  fields plus `affected_insight_titles`. It additionally permits
  `coverage_gap` and `cross_insight_tension`. At least one is required for two
  or more source insights; affected titles must exist.
- `checks_performed`: exactly one entry for every base taxonomy category.
  Each has `type`, `status` (`tested`, `not_applicable`, or `deferred`),
  substantive `rationale`, and evidence. A deferred check also declares
  `severity` and `affected_insight_titles`; deferred material/blocking checks
  bar those titles from program-action synthesis.
- `synthesis_manifest`: exact ordered `approved`, `revised`, and `rejected`
  title lists partitioned by verdict, plus disjoint
  `program_action_titles` and `diagnostic_only_titles`, both subsets of
  synthesis-eligible titles. Gated items are diagnostic-only. No approval
  quota exists: rejecting every source insight is valid.
- `quality_summary`: finite numeric 0-10 `process_rigor`,
  `analytical_depth`, and `decision_quality`; substantive
  `overall_assessment`; and `blocking_issues`. Each blocking issue contains
  the exact stable `challenge_id`, substantive `summary`, and evidence.
  Its IDs exactly cover unresolved or gated insight blocking challenges and
  all portfolio blocking challenges.

Sentinel assertions such as `TBD`, `unknown`, `none`, `n/a`, `not assessed`,
`no issues`, and `looks good` are invalid. The verifier intentionally does
not infer semantic truth from prose.

## Required verifier

```python
def verify(payload):
    import math

    taxonomy = (
        "obviousness",
        "cross_domain_depth",
        "contradiction",
        "denominator_integrity",
        "metric_definition",
        "alternative_explanation",
        "target_basis",
        "benchmark_basis",
        "causal_overclaim",
        "grain_or_join",
        "headline_consistency",
        "actionability",
    )
    high_risk = {
        "contradiction",
        "denominator_integrity",
        "grain_or_join",
        "headline_consistency",
        "causal_overclaim",
    }
    sentinels = {
        "",
        "tbd",
        "unknown",
        "none",
        "n/a",
        "na",
        "not assessed",
        "no issues",
        "looks good",
    }

    def is_int(value):
        return isinstance(value, int) and not isinstance(value, bool)

    def substantive(value, label):
        assert isinstance(value, str), f"{label} must be a string"
        normalized = " ".join(value.strip().lower().rstrip(".").split())
        assert normalized not in sentinels, f"{label} is boilerplate"
        assert len(value.strip()) >= 12 and len(value.split()) >= 3, (
            f"{label} must be substantive"
        )
        return value

    def evidence_refs(value, label):
        assert isinstance(value, list) and value, f"{label} evidence_refs required"
        assert all(isinstance(ref, str) and ref.strip() for ref in value), (
            f"{label} evidence_refs must be non-empty strings"
        )
        assert len(value) == len(set(value)), f"{label} evidence_refs must be unique"
        for ref in value:
            assert ref.strip().lower() not in sentinels and len(ref.strip()) >= 4, (
                f"{label} evidence reference is boilerplate"
            )

    def string_list(value, label, allow_empty=True):
        assert isinstance(value, list), f"{label} must be a list"
        if not allow_empty:
            assert value, f"{label} must not be empty"
        assert all(isinstance(item, str) and item.strip() for item in value), (
            f"{label} must contain non-empty strings"
        )
        assert len(value) == len(set(value)), f"{label} must be unique"

    def challenge(record, allowed_types, label, ids):
        assert isinstance(record, dict), f"{label} must be an object"
        challenge_id = record.get("id")
        assert isinstance(challenge_id, str) and challenge_id.strip(), (
            f"{label} id required"
        )
        assert challenge_id not in ids, f"{label} id must be globally unique"
        ids.add(challenge_id)
        challenge_type = record.get("type")
        assert challenge_type in allowed_types, f"{label} unsupported type"
        severity = record.get("severity")
        assert severity in {"blocking", "material", "minor"}, (
            f"{label} unsupported severity"
        )
        assert not (challenge_type in high_risk and severity == "minor"), (
            f"{label} high-risk type cannot be minor"
        )
        substantive(record.get("assessment"), f"{label} assessment")
        evidence_refs(record.get("evidence_refs"), label)

    assert isinstance(payload, dict), "critic payload must be an object"
    assert is_int(payload.get("critic_version")) and payload["critic_version"] == 1, (
        "unsupported critic_version"
    )
    assert (
        is_int(payload.get("source_contract_version"))
        and payload["source_contract_version"] in {2, 3}
    ), "unsupported source_contract_version"
    fingerprint = payload.get("source_fingerprint")
    assert isinstance(fingerprint, str) and fingerprint.strip(), (
        "source_fingerprint required"
    )

    inventory = payload.get("source_inventory")
    assert isinstance(inventory, list), "source_inventory must be a list"
    inventory_pairs = []
    inventory_by_title = {}
    ranks = set()
    for index, item in enumerate(inventory):
        assert isinstance(item, dict), "source_inventory entries must be objects"
        title = item.get("title")
        rank = item.get("rank")
        assert isinstance(title, str) and title.strip(), "inventory title required"
        assert is_int(rank) and rank > 0, "inventory rank must be a positive integer"
        assert title not in inventory_by_title, "duplicate inventory title"
        assert rank not in ranks, "duplicate inventory rank"
        action_kind = item.get("action_kind")
        readiness = item.get("decision_readiness")
        assert action_kind is None or action_kind in {"program", "diagnostic"}, (
            "unsupported inventory action_kind"
        )
        assert readiness is None or readiness in {"act_ready", "investigate_first"}, (
            "unsupported inventory decision_readiness"
        )
        inventory_pairs.append((title, rank))
        inventory_by_title[title] = item
        ranks.add(rank)

    reviewed = payload.get("reviewed_insights")
    assert isinstance(reviewed, list), "reviewed_insights must be a list"
    reviewed_pairs = []
    reviewed_titles = set()
    all_ids = set()
    gated_titles = set()
    unresolved_or_gated_blocking = set()
    eligible_titles = set()
    verdict_lists = {"approve": [], "revise": [], "reject": []}

    for insight_index, item in enumerate(reviewed):
        label = f"reviewed_insights[{insight_index}]"
        assert isinstance(item, dict), f"{label} must be an object"
        title = item.get("title")
        rank = item.get("rank")
        assert isinstance(title, str) and title.strip(), f"{label} title required"
        assert is_int(rank) and rank > 0, f"{label} rank must be a positive integer"
        assert title not in reviewed_titles, "duplicate reviewed title"
        assert rank not in {pair[1] for pair in reviewed_pairs}, (
            "duplicate reviewed rank"
        )
        reviewed_titles.add(title)
        reviewed_pairs.append((title, rank))
        verdict = item.get("verdict")
        assert verdict in verdict_lists, f"{label} unsupported verdict"
        verdict_lists[verdict].append(title)
        substantive(item.get("decision_effect"), f"{label} decision_effect")

        challenges = item.get("challenges")
        assert isinstance(challenges, list) and challenges, (
            f"{label} requires at least one challenge"
        )
        for challenge_index, entry in enumerate(challenges):
            challenge(
                entry,
                set(taxonomy),
                f"{label}.challenges[{challenge_index}]",
                all_ids,
            )

        changes = item.get("required_changes")
        assert isinstance(changes, list), f"{label} required_changes must be a list"
        for change_index, change in enumerate(changes):
            assert isinstance(change, dict), f"{label} required change must be an object"
            substantive(
                change.get("change"),
                f"{label}.required_changes[{change_index}].change",
            )
            assert change.get("gate") in {"none", "investigate_first"}, (
                f"{label} required change has unsupported gate"
            )
        eligible = item.get("synthesis_eligible")
        assert isinstance(eligible, bool), f"{label} synthesis_eligible must be bool"

        resolutions = item.get("resolutions")
        assert isinstance(resolutions, list), f"{label} resolutions must be a list"
        resolution_by_index = {}
        for resolution_index, resolution in enumerate(resolutions):
            rlabel = f"{label}.resolutions[{resolution_index}]"
            assert isinstance(resolution, dict), f"{rlabel} must be an object"
            bound_index = resolution.get("challenge_index")
            assert is_int(bound_index) and 0 <= bound_index < len(challenges), (
                f"{rlabel} resolution points to a missing challenge"
            )
            assert bound_index not in resolution_by_index, (
                f"{label} duplicate resolution for challenge"
            )
            bound_challenge = challenges[bound_index]
            assert resolution.get("challenge_type") == bound_challenge["type"], (
                f"{rlabel} resolution challenge_type mismatch"
            )
            status = resolution.get("status")
            assert status in {"resolved", "downgraded", "gated"}, (
                f"{rlabel} unsupported resolution status"
            )
            assert not (
                bound_challenge["severity"] == "blocking" and status == "downgraded"
            ), "blocking challenge cannot be downgraded"
            substantive(resolution.get("rationale"), f"{rlabel} rationale")
            evidence_refs(resolution.get("evidence_refs"), rlabel)
            resolution_by_index[bound_index] = status
            if status == "gated":
                gated_titles.add(title)

        has_change_gate = any(
            change.get("gate") == "investigate_first" for change in changes
        )
        if eligible and has_change_gate:
            gated_titles.add(title)
        for challenge_index, entry in enumerate(challenges):
            status = resolution_by_index.get(challenge_index)
            if entry["severity"] == "blocking" and status != "resolved":
                unresolved_or_gated_blocking.add(entry["id"])
            if (
                verdict == "revise"
                and eligible
                and entry["severity"] in {"material", "blocking"}
            ):
                assert status in {"resolved", "gated"} or has_change_gate, (
                    f"{label} synthesis-eligible revision has unresolved "
                    f"{entry['severity']} challenge"
                )

        if verdict == "approve":
            assert eligible, "approve verdict must be synthesis eligible"
            assert not changes, "approve verdict cannot have required_changes"
            assert not any(c["severity"] == "blocking" for c in challenges), (
                "approve verdict cannot retain a blocking challenge"
            )
            for challenge_index, entry in enumerate(challenges):
                if entry["severity"] == "material":
                    assert resolution_by_index.get(challenge_index) in {
                        "resolved",
                        "downgraded",
                    }, "approve verdict cannot retain an unresolved material challenge"
        elif verdict == "revise":
            assert changes, "revise verdict requires required_changes"
        else:
            assert changes, "reject verdict requires a substantive rejection reason"
            assert not eligible, "reject verdict cannot be synthesis eligible"
        if eligible:
            eligible_titles.add(title)

    assert reviewed_pairs == inventory_pairs, (
        "reviewed insight coverage must exactly match source_inventory"
    )

    portfolio = payload.get("portfolio_challenges")
    assert isinstance(portfolio, list), "portfolio_challenges must be a list"
    if len(inventory) >= 2:
        assert portfolio, "portfolio challenge required for multiple insights"
    portfolio_types = set(taxonomy) | {"coverage_gap", "cross_insight_tension"}
    portfolio_blocked_titles = set()
    for index, entry in enumerate(portfolio):
        label = f"portfolio_challenges[{index}]"
        challenge(entry, portfolio_types, label, all_ids)
        affected = entry.get("affected_insight_titles")
        string_list(affected, f"{label} affected titles", allow_empty=False)
        assert set(affected) <= set(inventory_by_title), (
            f"{label} affected title is unknown"
        )
        if entry["severity"] == "blocking":
            unresolved_or_gated_blocking.add(entry["id"])
            portfolio_blocked_titles.update(affected)

    checks = payload.get("checks_performed")
    assert isinstance(checks, list), "checks_performed must be a list"
    check_types = []
    deferred_action_bars = set()
    for index, check in enumerate(checks):
        label = f"checks_performed[{index}]"
        assert isinstance(check, dict), f"{label} must be an object"
        check_type = check.get("type")
        assert check_type in taxonomy, f"{label} unsupported type"
        check_types.append(check_type)
        status = check.get("status")
        assert status in {"tested", "not_applicable", "deferred"}, (
            f"{label} unsupported status"
        )
        substantive(check.get("rationale"), f"{label} rationale")
        evidence_refs(check.get("evidence_refs"), label)
        if status == "deferred":
            severity = check.get("severity")
            assert severity in {"blocking", "material", "minor"}, (
                f"{label} deferred check requires severity"
            )
            affected = check.get("affected_insight_titles", list(inventory_by_title))
            string_list(affected, f"{label} affected titles")
            assert set(affected) <= set(inventory_by_title), (
                f"{label} affected title is unknown"
            )
            if severity in {"blocking", "material"}:
                deferred_action_bars.update(affected)
    assert check_types == list(taxonomy), (
        "checks_performed must cover every taxonomy category exactly once and in order"
    )

    manifest = payload.get("synthesis_manifest")
    assert isinstance(manifest, dict), "synthesis_manifest must be an object"
    for key in (
        "approved",
        "revised",
        "rejected",
        "program_action_titles",
        "diagnostic_only_titles",
    ):
        string_list(manifest.get(key), f"manifest {key}")
    assert manifest["approved"] == verdict_lists["approve"], (
        "manifest approved list drift"
    )
    assert manifest["revised"] == verdict_lists["revise"], (
        "manifest revised list drift"
    )
    assert manifest["rejected"] == verdict_lists["reject"], (
        "manifest rejected list drift"
    )
    partition = (
        manifest["approved"] + manifest["revised"] + manifest["rejected"]
    )
    assert set(partition) == set(inventory_by_title) and len(partition) == len(inventory), (
        "manifest verdict lists must exactly partition inventory"
    )
    program = set(manifest["program_action_titles"])
    diagnostic = set(manifest["diagnostic_only_titles"])
    assert not (program & diagnostic), "manifest action subsets must be disjoint"
    assert program | diagnostic <= eligible_titles, (
        "manifest actions must be synthesis-eligible"
    )
    assert program | diagnostic == eligible_titles, (
        "every synthesis-eligible title needs exactly one action disposition"
    )
    assert not (gated_titles & program), "gated title cannot be a program action"
    assert gated_titles <= diagnostic, "gated title must be diagnostic-only"
    assert not (deferred_action_bars & program), (
        "deferred material check prevents program action"
    )
    assert not (portfolio_blocked_titles & program), (
        "portfolio blocking challenge prevents program action"
    )
    for title in program:
        source = inventory_by_title[title]
        assert source.get("action_kind", "program") == "program", (
            "program action conflicts with source action_kind"
        )
        assert source.get("decision_readiness", "act_ready") == "act_ready", (
            "program action conflicts with source decision readiness"
        )

    summary = payload.get("quality_summary")
    assert isinstance(summary, dict), "quality_summary must be an object"
    for field in ("process_rigor", "analytical_depth", "decision_quality"):
        score = summary.get(field)
        assert isinstance(score, (int, float)) and not isinstance(score, bool), (
            f"{field} must be numeric, not bool"
        )
        assert math.isfinite(score) and 0 <= score <= 10, (
            f"{field} must be finite and between 0 and 10"
        )
    substantive(summary.get("overall_assessment"), "overall_assessment")
    blocking = summary.get("blocking_issues")
    assert isinstance(blocking, list), "blocking_issues must be a list"
    blocking_ids = []
    for index, issue in enumerate(blocking):
        label = f"blocking_issues[{index}]"
        assert isinstance(issue, dict), f"{label} must be an object"
        challenge_id = issue.get("challenge_id")
        assert isinstance(challenge_id, str) and challenge_id.strip(), (
            f"{label} challenge_id required"
        )
        blocking_ids.append(challenge_id)
        substantive(issue.get("summary"), f"{label} summary")
        evidence_refs(issue.get("evidence_refs"), label)
    assert len(blocking_ids) == len(set(blocking_ids)), (
        "blocking issue challenge IDs must be unique"
    )
    assert set(blocking_ids) == unresolved_or_gated_blocking, (
        "blocking issue IDs must exactly cover unresolved or gated blocking challenges"
    )
```
