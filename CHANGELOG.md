# Changelog

## Unreleased

### Added

- **Data Agent review** (`fabric_rlm.data_agent_review`, notebook
  `examples/notebooks/rlm_data_agent_review.py`). Point it at a Fabric Data
  Agent and it reads what the agent uses and how it is instructed (sources,
  agent and data-source instructions, descriptions, few-shots), profiles the
  same sources through `RLM.learn`, and then: diagnoses the setup against
  the documented guidance (schema names in agent-level instructions that
  belong with the source, references to objects the source does not have,
  definitions that conflict between levels, missing descriptions, length
  past the truncation limit, few-shot counts, schema size, negative
  phrasing, routing rules for multi-source agents); generates questions
  from the schemas and computes each reference by executing a query against
  the source (SQL for lakehouse tables, a bounded aggregate for a semantic
  model), so there is ground truth without anyone writing it; asks the
  agent the same questions through its Responses endpoint (Assistants and
  MCP askers exist too), keeps the run steps, grades the query it executed
  where the steps expose it and its prose otherwise,
  and classifies failures by cause (missing rows, unsorted ranking, a
  narrower scope than asked, values that match nothing, an abstention where
  the source answers, inconsistency across repetitions, misrouting between
  sources); and renders
  suggestions (agent instructions with schema lines moved out, data-source
  instructions with structured sections, descriptions, few-shots built from
  the executed references) into a Markdown report. Applying the suggestions
  is a separate call against the agent's draft stage. Readers, executors,
  askers and writers are small protocols with REST implementations for use
  outside a notebook and SDK implementations for use inside one; the
  notebook is a Fabric Python notebook on the 3.12 runtime.
- **Review context and diagnostics.** `ReviewContext` states the agent's
  scope, priorities, definitions, the reviewer's own questions (a query
  or an answer as ground truth, graded first) and notes; the questions
  are scoped and ordered by it, the definitions reach `RLM.learn`, and the
  RLM second opinion receives it as prompt context. The report explains an
  empty evaluation (year discovery failures, unrecognised fact tables,
  missing date joins), names the source after its item, treats only
  schema-shaped tokens as schema mentions, and picks the order date key
  over due or ship dates whatever the column order.
- **HTML report, learned knowledge, deeper analysis.** `ReviewReport.to_html`
  renders the review as a self-contained page (findings as cards, graded
  questions with the agent's answer, its query, the reference query and
  rows, the run steps). The report summarises what `RLM.learn` recorded
  (profiles, fingerprints, registered operations, lessons, events).
  `deepen` uses the RLM with a language model to propose questions from the
  scope, verifies each reference with two blind solves, asks the agent, and
  explains every question that is not correct with a proposed change.
  Topics the instructions put out of scope are excluded from generation
  and a decline on one is `abstained_by_policy`; a year-over-year answer is
  right by the change or by both totals; few-shot SQL carries the `dbo.`
  prefix when the agent uses it; answers record whether their query ran.
- **Natural questions across a skill matrix.** Generated questions read as
  a business user asks them, with the words the agent's own instructions
  use for tables, measures and attributes (channel synonyms in quotes,
  `Use X for units`, `Revenue = SUM(...)`, `territory means ...`,
  abbreviations such as B2B); the schema phrasing stays on the question for
  the report. The set now covers aggregation, ranking, change, distinct
  counts, a literal filter derived from the ranked reference, a KPI from
  the instructions' own formula, the right measure column for a mapped
  term, an ambiguous total across channels, a channel abbreviation, and an
  out-of-scope topic where declining is the right answer; each question
  carries its skill and the report counts outcomes by skill. Alternates
  name their cause (`wrong_measure`, `row_count_not_distinct`,
  `wrong_channel`, `answered_out_of_scope`). The RLM proposer is told to
  write plain business questions over the same skills.
- **Driver-based questions from discovered names and periods.**
  `discover_drivers` runs a few small queries per fact table (top names
  per role: entity, product, category, place; the month series for the
  largest drop and rise between consecutive months; order counts per
  entity for a threshold), and the generator turns them into the
  questions a user actually asks, each with an executable reference:
  a monthly and a quarterly trend, why the measure dropped between two
  months and which products drove it (a decomposition, largest decreases
  first), how a named entity performs year by year and on a category,
  the share of a category and of the top 10 entities, a count of entities
  above an order threshold, entities that bought one year and not the next,
  a comparison of the two top entities, and the leading category per place
  (a window function). Grouping columns are reached through several
  dimension joins (product to subcategory to category). Answers to named
  questions are graded by the names present; drops by absolute change.
  The RLM proposer receives the discovered names and the drop period.
- **Time expressions and instruction compliance.** Questions now also use
  the periods users write: the same month last year, the month before,
  last month, the week after Thanksgiving, winter, year to date, the last
  30 days of data, a quarter. Each has the reading the reference takes
  and the other readings that are acceptable when the answer states the
  dates it took (`assumption_not_stated` otherwise). `extract_rules`
  reads the checkable rules from the instructions (state the period and
  the channel, rank and trend formats, currency format, the partial-year
  caveat, no personal data, no direct fact-to-fact join, calendar rather
  than fiscal year) and `check_rules` marks the answers that break them;
  the report has an instruction-compliance table and each graded answer
  carries its violations. Two probes trigger rules on purpose: a ranking
  for the partial year and a request for customer contact details. The
  notebook asks 25 questions per source by default, spread round-robin
  across the skills.
- **Any lakehouse shape.** The generator works from the shape of the data
  rather than from one sample's names. A fact table is one with measures
  and a time axis; the time axis is a date dimension with a year column, a
  date or timestamp column on the fact, or one on a joined header table
  (order lines through their order), and every period question renders
  against whichever it found. Joins follow `<name>Key`, `<name>_id` and
  `id_<name>` columns to the table that carries them, with or without a
  schema prefix, and the executor maps `dbo.orders` to a safe alias and
  back for the agent's SQL. When column names say nothing, the column types
  the profile recorded decide: a timestamp or date column is a time axis, a
  numeric column that is not a key is a measure, a text column that is not
  a key, a time or free text is a grouping column, and the nearest grouping
  column stands in for the category role so the driver questions still ask
  what led a change. `SourceSchema.types` carries the types from
  `schema_from_profile`. An empty evaluation now names the missing time
  axis. Tested on an Olist-shaped lakehouse (timestamps on the order
  header, `_id` keys, `dbo.` prefixes) and on one whose names carry no
  English hint at all.
- **The agent's selected tables.** Fabric lists a lakehouse datasource's
  elements as `Schemas` > `dbo` > `Tables` | `Views` > `Table` | `View` >
  columns; the reader walked the tree but took the `Tables` container for a
  table and stopped there, so the selection came back unknown and the whole
  lakehouse was profiled. The walk now passes through the structural
  containers, names a table by its schema (`dbo/factinternetsales`), never
  fetches a table's columns or a view, and the notebook scopes the lakehouse
  to the selection whether it has schemas enabled (`Tables/dbo/<table>`) or
  not (`Tables/<table>`, which the agent still lists under `dbo`), falling
  back to every table only when neither layout resolves.
- **What the RLM learned, made useful.** The report section used to list
  every registered operation (43 identical rows for a 43-table lakehouse)
  and fingerprints. It now says how the review uses the package, then shows
  the agent's selected tables as the RLM sees them (fact or dimension, the
  time axis and measures the questions rely on, personal-data columns, and
  whether a registered aggregate operation covers the table), groups the
  operations by kind, lists the lessons, and, after the deeper analysis,
  what the package learned from the RLM's own verified runs: `deepen`
  captures evidence on every solve, enriches the package through
  `RLM.enrich`, reports the lessons that appeared, and returns the enriched
  package as `report.learned_knowledge` for saving. `summarize_knowledge`
  takes the schemas and the snapshot for this.
- **Hints from the data.** `discover_hints` asks the lakehouse a bounded
  set of small questions nobody would think to put in the instructions and
  turns the answers into findings with basis `data`, each with the
  instruction line it suggests: fact keys with no match in their dimension
  and dimension keys that repeat (an inner join drops or multiplies rows),
  values spelled in several cases or with padding (a lakehouse compares
  case-sensitively, so compare with `lower(trim(...))` or normalise the
  data), the vocabulary of small attributes (users say "bikes" for the
  Bikes category), several date columns on a fact, partial years and
  coverage, negative or missing measures, snowflaked join paths, personal
  data on joined tables, a measure name shared by several facts. The report
  shows them under "Hints from the data", the suggested data-source
  instructions carry them under "From the data (inferred by the review;
  confirm before applying)", the RLM proposer hears them, and a
  `fuzzy_value` question asks for the top category spelled as a user would
  ("bikes"), graded `fuzzy_match_failed` when the agent cannot map it to
  the stored value. `review_agent(hints=False)` turns it off.
- **Tested on other data.** Running the blind pipeline (no agent, no
  instructions) on a SaaS schema (CloudMetrics: companies, industries,
  invoices, payments, subscriptions, usage logs), a bakery chain
  (franchises, customers, transactions), a flat retail file with spaces in
  its column names and an integer date, and ARR tables keyed by a text
  quarter showed where the rules were still shaped by one sample, and the
  rules changed: a key is a key by its form (`customerID`, `customer_id`,
  `Customer ID`, `id_cliente`), never `amount_paid`; joins reach irregular
  plurals (company to companies) and prefixed dimensions (`sales_customers`
  for `customerID`) and skip a table's own primary key; a table with one
  measure, a time axis and a reference to another table is a fact, and so is
  a flat table whose grouping columns sit on its own rows; a fact's own date
  outranks a joined table's; grouping columns on the fact itself are used
  (product, payment method, status), never personal data; identifiers with
  spaces are quoted; an integer date (20240131) and a text date are read
  as dates; a period written as text (2024/Q1, 2024-03) is a time axis of
  its own with year and quarter or month but no day grain; periods and flags
  are never measures; a column called plainly `name` takes its table's word;
  what is counted is named after the table's own key (tickets,
  transactions, invoices); every question the schema supports is generated
  before the limit is applied, so a source with many facts keeps its trend,
  driver and period questions. Tests cover each of those shapes end to end.
- **What moved: a sweep and reports over a source** (`fabric_rlm.sweep`,
  `fabric_rlm.reports`). `what_moved(source)` takes a `LakehouseSource` or
  a `SemanticModel` directly and measures, with the source's own engine
  (DuckDB SQL over OneLake; DAX through the model's relationships), how
  every measure of every fact moved between the periods the time axis
  supports: the latest month against the month before, the same month a
  year earlier, the latest complete year against the previous one. Material
  movements are decomposed by every grouping the joins or relationships
  reach and classified (one group carries it, a few do, groups moved in
  proportion to their size so the grouping explains nothing, groups moved
  both ways), and the leading group is drilled one level further. The
  sweep runs in two phases so the budget goes where it matters (every total
  first, then the material ones largest relative change first), collapses
  measures that move identically, averages scores and rates instead of
  summing them, and flags volume-driven changes, small bases and an
  incomplete last month. Every figure carries the query that produced it
  and an independent per-period query that recomputes it; `verify_sweep`
  runs those, grouped so a decomposition costs two queries to check.
  `report(source, request)` reads a request in plain words against the
  source's vocabulary (its tables, measures, grouping columns and the
  instructions' words for them) into a `ReportSpec` and builds one of four
  reports: trend (by month, and by the largest groups of each grouping
  asked for), root cause (one movement decomposed by the groupings asked
  for and drilled), recap (the full sweep) and top movers (the groups that
  rose and fell most); the page states how the request was read.
  `Sweep.to_html()` and `Report.to_html()` render a self-contained
  dashboard with headline cards, trend lines, a waterfall of drivers, a
  driver scatter (share of base against share of change), the tables and
  the queries behind every figure; `save(path)` writes it as a page. The
  page leads with the answer: up to three takeaways, one sentence each with
  the figure, the comparison, the driver and the caveat, linked to their
  detail, then a short paragraph with the picture, then the supporting
  detail. Every compared period passes a completeness check first: a month
  with fewer than half the rows of a typical month before it, or a year
  with fewer than half the other's months of data, is flagged as coverage
  rather than business change, set aside from the takeaways and shown last
  in the driver analysis, whatever date the data runs to. One verification
  statement, built from the same counts as the header badge, says how many
  figures were recomputed, how many were not, and what those carry.
  Notebook: `examples/notebooks/rlm_what_moved.ipynb`. Measures no longer
  include order numbers, codes or text columns.
- **Monday Morning Brief** (`fabric_rlm.brief`, or `report(source, "monday
  morning brief: revenue by region, orders")`). Name the metrics to track
  and the brief takes the latest complete Monday-to-Sunday week the source
  holds (or the week you name), measures each metric for it with the
  source's engine at day grain, and puts it in context: the week before, the
  same week a year earlier, the averages of the last four and thirteen
  weeks, the seasonal expectation (the recent level scaled by how that week
  of the year ran against its own level in prior years) and the rank against
  the record. It finds level shifts in the weekly history (binary
  segmentation on the means), calls a week unusual when it sits far from
  the expectation in units of the recent residuals, decomposes the
  week-over-week move by the groupings named and drills the leading group,
  splits the move into volume and value per row, states the counterfactual
  for the leading group, reads the day-of-week pattern against the twelve
  weeks before, notes which metrics moved together and which led by a week,
  and keeps a watch list. The page is a newsletter: one look, the watch
  list, one section per metric with the weekly chart (the week, the
  expectation and the level shifts marked), the context table, the drivers,
  the pattern, and the queries. What it says about causes is what the
  history supports and it says so.

### Changed

- `LakehouseSource.query` keeps the Delta reader first and, when the reader
  rejects a table (Spark `void` columns, which no data file carries), reads
  the table's own data files as its transaction log lists them (the last
  checkpoint plus later commits). A table whose features need the reader
  (deletion vectors, column mapping, v2 checkpoints) is never read that way.
  A catalog column no data file carries comes back as NULL; the rejection
  and the file list are remembered per table while the log is unchanged.
  `LakehouseSource.query` accepts a `timeout` (seconds, at most 600) for
  direct callers; a worker's query keeps the 30-second default.

## 0.6.1 — 2026-09-10 — generalized run protocol and learning substrate

### Added

- **`ABSTAIN(reason)` in the worker namespace.** A run that cannot support an
  answer ends on the record instead of inventing one: the result is not
  submitted, `failure_reason` is `"abstained"` and the reason is on the
  trajectory as `abstain_reason`. `from sandbox import ABSTAIN` works like
  `SUBMIT`. The final-turn addendum now asks for values computed by code
  that ran, or an abstention; it no longer says an imperfect answer beats
  nothing.
- **Declared source metadata.** `RLM.learn(..., declared={"orders": {...}})`
  states what a profile cannot infer: the `grain` (columns a row is unique
  by), the `period_column` that defines periods and therefore "current" and
  "latest", `units` per column, `definitions` of metrics and flags, and
  free `notes`. Declared facts become active `semantic_fact` lessons with a
  `declared` basis, reach every task on the source (the agent and the
  operation planner), persist with the package, and are validated against
  the profiled schema: a declaration about a column the source does not
  have is an error. See `fabric_rlm.knowledge_lessons.declared_lessons`.
- **Evidence from host-side operations.** A registered operation the
  runtime executes for a run is recorded on the trajectory
  (`operation_execution`: grain, filter columns, measures, row count,
  seconds, audit status, or the reason it was refused) together with the
  telemetry the bound handles produced while it ran
  (`operation_source_calls`). Evidence harvesting turns both into
  `query_execution` records, so file and Lakehouse sources learn grain
  lessons from their registered operations exactly as semantic models do
  from worker queries, and a refused semantic-model grouping becomes
  expensive-grain evidence. Before this, only handles with worker telemetry
  produced any evidence, so a CSV, Parquet or Lakehouse package could never
  hold a lesson.
- **Lakehouse query grain.** Parent-side Lakehouse SQL telemetry now records
  the columns a query grouped by (`groupby`, parsed from DuckDB's own AST;
  the SQL still never leaves the run), so Lakehouse queries feed grain
  lessons.
- **A current-period flag in a table is a structural lesson.** A boolean
  column named like a current-period marker in a CSV, Parquet, Delta or
  Lakehouse source yields a time-semantics lesson: active at medium
  confidence when the name also carries a period word (`is_current_period`),
  a candidate otherwise (`current` alone has other meanings).

### Changed

- **A dependent code sequence runs in order.** When a response holds
  several Python blocks and the last one reads names an earlier block
  binds, the blocks execute in order as one step, so the model sees real
  output for the whole sequence instead of a `NameError` on its last block.
  A self-contained last block (a revision) still runs alone. Output the
  model wrote itself between blocks is named in the feedback as not
  executed and not evidence, and the prompt now states the one-block
  protocol.
- **Numbers typed into SUBMIT must trace to executed output.** The
  analytical-integrity screen rejects a number typed as a literal inside
  the SUBMIT call (as a value or inside a string) that appears in no REPL
  output of the run, in the registered-operation packet, or in the task
  text. Computed values passed as names are untouched; rounding and
  percent-versus-ratio spellings are accepted. In repair mode this costs a
  turn that prints the figure; the accepted answer then traces. Set
  `FABRIC_RLM_CLAIM_PROVENANCE=0` to switch this one screen off;
  `analytical_integrity=False` still switches off the whole screen.
- **The registered-operation packet is a typed contract and the raw
  sources stay bound.** `knowledge_result` introduces itself in the prompt
  as the host-computed, audited result of a named operation with its
  parameters, the columns and grain of its rows, and the fact that it is
  already aggregated. The raw sources remain bound next to it: learning
  narrows the search, it never removes the cold path. Previously the raw
  handles were removed and the packet was listed as a bare dict.

### Added

- **Evidence and lessons in knowledge packages.** Each turn now records its
  source calls as typed telemetry in `TurnRecord.source_calls` (semantic-model
  `aggregate`, `measure` and `dax` records with grain, counts, estimated
  groups, timings, outcome codes and value-free measure observations such as
  identical or constant measure columns; parent-side Lakehouse SQL timings).
  `RLM(capture_evidence=True)` harvests that telemetry and the run's
  verification and analytical-integrity status into `result.evidence`
  (`EvidenceRecord`); `RLM.enrich(knowledge, results)` promotes evidence into
  `LearnedLesson` records by a per-kind policy (time semantics inferred from
  schema names, active from `learn()` only for a boolean current-period flag
  in a period-like table and a candidate for a name match there; expensive
  grain proved by a cardinality preflight or timeouts in two separate runs;
  a context requirement nominated by a degenerate unfiltered identity and
  activated only when the same measure pair is compared under a period
  filter or grouping and comes out distinct, never by repetition or by a
  non-period filter; valid grain as an execution fact after two separate
  runs, with confidence from verified runs; preferred strategy only when the
  trajectory's lineage shows the coarse result flowing through
  `restrict_to_candidate_tuples` into the fine query, in runs that passed an
  actual verifier and the integrity screen; invalid references after
  repetition; coinciding measure values recorded as `query_behavior` with
  `semantic_equivalence: false`, never as `metric_equivalence`) and
  returns a new package, saving it only when asked. Evidence carries
  execution trust and analytical trust separately; a submission nobody
  verified is never a verified success. Active lessons relevant to a task are
  rendered into a "Learned source guidance" prompt section, each line tagged
  with its source, when the task runs against the live source; the
  registered-operation planner receives only the lessons for the sources its
  operations read; candidates are never shown and the source stays bound.
  Packages without learning records keep format 1 and their fingerprints;
  packages with them use format 2. Lessons carry a dependency scope: schema
  drift stales schema-scoped lessons, data-only drift stales snapshot- and
  operational-scoped ones. Evidence eviction never drops a record a lesson
  cites. `KnowledgeBenchmarkTrial` gains reasoning tokens, source calls,
  failed calls, source seconds, first useful query turn, verifier repairs,
  integrity status and injected lessons, and the report has `cold_parity()`,
  which requires learned correctness at or above cold overall and per task
  and fails closed when a task lacks either arm. The registered-operation
  planner's operation descriptors now carry `required_sources`.

### Fixed

- **`answers_agree` waved a sign flip through as agreement.** The text
  normalization strips punctuation, and the minus sign with it, so `"10"`
  and `"-10"` (also `"10%"` / `"-10%"`, `"$5"` / `"-$5"`, `"(10)"` / `"10"`)
  normalized to the same string and returned early as "agree" without ever
  reaching the numeric comparison, so `verified_task` skipped reconciliation
  on exactly the disagreement it exists to catch. The signed numeric
  comparison now runs first. Numbers keep their sign through a currency
  symbol (`-$5`), a number alone in parentheses is read as an accounting
  negative, and a hyphen glued to a preceding word character is a range
  separator (`2024-2025`), not a minus. Two blank answers no longer agree
  (the reconciler runs and is told neither analyst answered), and the
  short-versus-verbose containment test now matches whole words only, so
  `"Mark"` no longer agrees with `"Denmark"`.
- **Evidence counted configuration as verification.** Loading any skill set
  `verifier_configured`, and evidence harvesting reported the answer as
  `"passed"` from that flag alone, even when the skill had no verifier or
  the verifier was skipped, timed out or crashed under graceful degrade.
  The runtime now records what actually ran against the accepted SUBMIT in
  `trajectory.metadata["verifier_execution"]` (each check with its outcome,
  and `verified`, which is true only when at least one check executed and
  accepted the answer and none degraded). `verifier_status_for` reads that
  record; `verifier_configured` stays as information and now means a
  verifier actually exists for the run.
- **Identical runs collapsed into one observation.** The run fingerprint
  hashed only code, submission flags, knowledge fingerprint and payload, so
  two separate executions with the same code counted as one run for the
  distinct-run thresholds in lesson promotion. Every execution now carries a
  `run_id` in trajectory metadata and the fingerprint includes it; harvesting
  the same result twice keeps the same ids, and a result without an id
  (built by hand, or recorded before ids existed) keeps the content
  fingerprint.
- **`verified_task` reported a reconciled answer when every solve failed,**
  and chose the first of two agreeing answers even when only the second was
  clean of unresolved integrity findings. A solve that did not submit, failed
  or answered blank is never a candidate, not even for a custom `agree`;
  when nothing produced an answer the verdict is `"failed"` and the result
  is the last attempt with its failure reason (`VerifiedResult.ok`); between
  two agreeing answers the one the integrity screen accepted without
  findings wins.
- **`answers_agree` still accepted meaningfully different answers.** A
  Unicode minus (U+2212) or a dash used as a minus was stripped by
  normalization, so `"−10"` agreed with `"10"`; a comma list with a missing
  member (`"A, B"` vs `"A, B, C"`) slipped through whole-word containment.
  Sign variants are normalized before number extraction, and comma-separated
  lists are compared as item sets like semicolon lists (thousands separators
  and an Oxford "and" are not list structure).
- **Evidence was relabelled with the destination package's schema.**
  `RLM.enrich` re-harvested every result with the fingerprints of the
  package it was enriching into and never read `result.evidence`, so
  evidence observed under one schema became "current" evidence for a
  later one, and an active lesson could name a column that no longer
  exists. The runtime now records the schema and snapshot fingerprints a
  run executed against on the trajectory (`knowledge_source_fingerprints`,
  `knowledge_snapshot_fingerprints`, written whenever a package is bound),
  and `enrich` uses that identity, or the stamps already on
  `result.evidence`, never the destination's. Records whose fingerprints
  do not match the package are dropped and noted as an
  `evidence.incompatible` event on the source; a result that recorded no
  identity at all (a run with no package bound) is skipped and noted as
  `evidence.unattributed` on the package. A record with no fingerprint for
  one of its sources no longer counts as current.
- **A stale lesson reactivated itself from the evidence that went stale.**
  `promote_lessons` kept only quarantined and retired lessons sticky, so
  re-deriving a package after preflight marked a lesson stale flipped it
  back to active from the retained records. Stale now lifts only when a
  record from the same call, matching the package's current fingerprints,
  supports the re-derivation.
- **`scope="latest"` on `lakehouse.preaggregate_join` returned one row,
  ordered by whichever key came first.** The SQL ended in
  `ORDER BY <join_key> DESC, <join_key_2> DESC LIMIT 1`, so with
  `join_key=region, join_key_2=month` the latest period lost to the
  lexicographically last region, and with the period first every other
  group in that period was dropped, with no truncation signal because the
  limit sat inside the query. Latest now means every group of the maximum
  period: the period key is the one temporal join key in either position
  (none, or two, is a plan error), rows are filtered to its maximum and
  ordered by the remaining keys, and the row bound is the operation's
  over-fetch so an oversize period fails the audit instead of being
  trimmed. `scope="all"` is unchanged.
- **Numeric filters on registered aggregates depended on spelling.** Both
  the tabular and the Lakehouse aggregate compared
  `CAST(column AS VARCHAR)` with the filter text, so `"1"` matched nothing
  in a DOUBLE column that renders `1.0`, and `"100.0"` nothing in a BIGINT
  column, and the aggregate came back empty without a word. The filter
  value now takes the column's profiled type (integer, number, boolean;
  text columns are compared as before) and a value that does not fit the
  type is refused as a plan error before any query runs. CSV numeric
  filters compare through a tolerant cast, since the profile and DuckDB
  each infer types from their own sample. The packet still echoes the
  parameter as given.

### Changed

- **Registered `semantic_model.measure` operations run through the
  guardrail.** The host executed them with `SemanticModel.measure`, which
  has no cardinality preflight and no deadline, so an oversize grouping was
  fully evaluated and materialized before the row bound rejected it. They
  now run through `aggregate` with `max_groups` set to the operation's row
  bound, so a too-broad grouping is refused before the engine evaluates it
  (a plan error, which falls back like any rejected plan). The packet
  keeps the `measure()` shape: grouping columns as `Table[Column]`, the
  measure column named by the measure. A handle that overrides `measure`
  without overriding `aggregate` keeps its own path.

## 0.6.0 — 2026-09-04 — analytical integrity guardrails and bounded semantic-model aggregation

### Added

- **Source-agnostic analytical integrity guardrails.** `fabric_rlm.analytical_integrity`
  adds `is_material_change` (noise- and tolerance-aware change test with no
  built-in business threshold), `restrict_to_candidate_tuples` (keeps
  multidimensional candidates as compound identities), `validate_ranking`,
  `validate_grain`, `validate_evidence_lineage` (provenance, cross-source
  period/unit/definition/entity reconciliation, contradiction surfacing) and
  the single entry point `validate_analysis_integrity`, which runs whichever
  checks have inputs. The first two and the entry point are predefined in the
  sandbox. New validators `assert_directional_claims_consistent`,
  `assert_ranking_disclosed` and `assert_grain_preserved`; new trajectory
  detectors `cartesian_candidate_filter` and `ranking_drift` and an
  `extract_plan` helper. `RLM(analytical_integrity=True)` (default; env
  `FABRIC_RLM_ANALYTICAL_INTEGRITY=0` disables) rejects, at most twice per
  run, a SUBMIT whose prose contradicts its own numbers, hides a requested
  ranking metric, sorted by something other than the requested concept, or
  followed a cartesian candidate filter. The checks activate by analytical
  operation, not by input type, so `File`, `LakehouseSource` and
  `SemanticModel` evidence is held to the same rules. New `analytical_integrity`
  skill with the cross-source guidance. `RLM(analytical_integrity=...)` takes
  `"repair"` (default), `"strict"` (never accept a submission with findings)
  or `False`; `result.integrity_problems` and `result.integrity_ok` expose
  what a run ended with. The system prompt adds a compact cross-source
  checklist whenever two or more evidence inputs are bound. Cross-source
  reconciliation itself (entities, definitions, periods, units,
  contradictions) is explicit: `validate_evidence_lineage` on declared
  claims, not an automatic runtime check. The tuple-loss detector is tied to
  its provenance (a same-turn or later `restrict_to_candidate_tuples` or
  `merge` on the same dimensions clears it; an unrelated merge does not), the
  ranking-drift detector follows variable lineage from `SUBMIT` so a display
  sort of supporting detail is not mistaken for the ranking and also reads
  polars, pyspark, `sorted` and SQL `ORDER BY`, evidence inputs are counted
  recursively by a `__rlm_evidence_source__` marker, and grain matching
  keeps `Products[Region]` and `Sold To[Region]` distinct. A declared
  ranking metric vouches for a code sort field only when the field names
  the declaration's concept and every token of the field is accounted for
  by the declaration, a tuple repair must operate on the expanded result or
  an alias of it, `assert_grain_preserved` honours table qualification, and
  the `analytical_integrity` skill prefers the governed metric definition
  and recomputes only as a validation check.
- `SemanticModelMetadata` accepts the dictionary-style access generated code
  tends to assume: `meta["columns"]`, `meta.keys()`, `meta.items()`,
  `meta.values()`, `meta.get("measures")`, and `"relationships" in meta`. The
  attribute API and the `metadata()` return type are unchanged; only the four
  public frames are exposed as keys.
- `SemanticModel.aggregate(measures, groupby=..., filters=..., order_by=...,
  top=...)` for grouped measure analysis behind a query-size guardrail. Names
  are validated against the model, a short cardinality preflight estimates the
  result grain, and a request over the limit (10,000 groups by default) or one
  whose estimate does not return within the budget (10 seconds) raises
  `SemanticModelQueryTooBroad` or `SemanticModelQueryRiskUnknown` with
  concrete narrowing guidance instead of consuming the worker timeout. The
  handle stays usable after a rejection. `max_groups=` on the handle or
  `FABRIC_RLM_SEMANTIC_MAX_GROUPS` raise the ceiling;
  `FABRIC_RLM_SEMANTIC_PREFLIGHT_TIMEOUT` sets the preflight budget.
- `SemanticModel.query_telemetry` records the estimate, timing, and outcome of
  each `aggregate` call.

- `RunInspector` shows a one-line plain-language summary next to each turn
  (for example "Aggregated revenue by region" or "Inspected available
  measures - error: KeyError"), derived from the turn's code without an LM
  call. Recovery turns are labelled, and nothing after a failing statement
  is described as having run.

### Fixed

- Trajectory error-kind counts no longer crash on an error string that is
  only whitespace.
- Namespace snapshots of a `SemanticModel` record only the handle's identity
  again. After the first `aggregate()` call the cached name catalog and query
  telemetry were being dumped into every turn's trajectory state.
- The default cardinality preflight budget is 30 seconds instead of 10. A live
  run measured an 8 second engine round trip even for a 44-group count, so the
  old budget would have rejected legitimate queries on any latency spike.
  `FABRIC_RLM_SEMANTIC_PREFLIGHT_TIMEOUT` still overrides it.
- `SemanticModel.columns(table=...)` narrows to one table, matching sempy's
  `list_columns`. Generated code reached for it and lost a turn to a
  `TypeError`.
- The system prompt now says `SUBMIT` is already defined and must not be
  imported; a trace showed `from fabric import SUBMIT` blocked by the sandbox
  on the final turn.

### Changed

- `max_turns` defaults to 20 instead of 10. Semantic-model tasks routinely
  spend the first several turns on metadata discovery before the analysis
  starts, and 10 left too little room to finish.
- The `SemanticModel` prompt listing and the `semantic_model` skill point the
  model at `aggregate()` first and reserve `dax()` for custom DAX. `dax()`
  itself is unchanged.
- Python 3.13 is no longer listed as supported or tested in CI. Supported
  versions are 3.10 to 3.12.

## 0.5.0 — 2026-09-02 — governed knowledge learning and execution

### Added

- **`RLM.learn()` and reusable knowledge packages** — profile supported local
  files, Parquet data, Fabric Lakehouses, Delta tables, and Power BI Semantic
  Models into immutable, fingerprinted packages with bounded source metadata.
- **Registered audited operations** — learned packages can expose typed
  Semantic Model measures, file aggregates, and Lakehouse queries that execute
  in the trusted parent with bounded result packets and deterministic audit
  records.
- **Persistent knowledge stores** — save and reload packages locally or through
  OneLake REST transport with integrity validation, source rebinding, and drift
  preflight checks.
- **Benchmark and demonstration notebooks** — compare cold and learned
  execution, persist diagnostics to OneLake, and verify operation selection,
  audit status, fingerprints, correctness, turns, tokens, and elapsed time.

### Changed

- Knowledge source adapters share generalized profiling, identity, publication,
  and reload-safe registry contracts.
- Semantic measure operations support bounded two-dimensional grouping and up
  to 1,000 result rows while retaining independent column and payload limits.
- Worker timeout recovery separates control-channel deadlines and warms
  replacement workers before resuming execution.

### Fixed

- Parent-only `notebookutils` credential configuration is no longer serialized
  into isolated workers, restoring cold Semantic Model access through SemPy's
  established worker authentication path.
- Fabric Lakehouse schema discovery falls back from Spark to delta-rs and then
  to bounded Delta transaction-log metadata reads when notebook runtimes lack
  an active Spark session or delta-rs cannot read OneLake metadata.
- DSPy cache validation recognizes the public `lm.cache` setting.
- OneLake cleanup failures, malformed knowledge packages, source drift, and
  unauthorized or unbounded operations fail closed instead of producing
  success-shaped results.

### Fabric verification

- A live Semantic Model release gate returned the same nonzero ARR value through
  direct parent execution, a cold worker task, and an audited learned operation.
- The registered-operation benchmark completed in Fabric with persistent
  OneLake trial logs, 100% operation-selection accuracy, 100% audit pass rate,
  and drift rejection for Semantic Model, Lakehouse, and CSV sources.

## 0.4.3 — 2026-08-31 — portable semantic insights and temporal evidence

### Added

- **Deterministic temporal evidence** — source-local coverage, complete
  comparable windows, cross-source freshness reconciliation, cohort-exposure
  sensitivity, and portable persistence/seasonality evidence now govern whether
  an insight may be described as current or action-ready.
- **Audited Pearson correlation** — derived correlation findings can be
  reconstructed from six independently verified sufficient statistics instead
  of trusting an opaque model-produced value.

### Changed

- **`RLMResult.inspect()`** now opens the full run inspector by default while
  keeping individual turns collapsed. The keyboard-scrollable timeline shows
  15 turn rows by default and supports a custom `visible_turns` viewport.
- **SemanticModel metadata and DAX normalization** now return ordinary pandas
  DataFrames with stable snake-case columns, preserving duplicate, empty, and
  colliding result schemas. Pandas is declared in the `analytics` extra for
  equivalent behavior outside the Fabric runtime.

### Fixed

- Nested DAX `ROW(...)` expressions, declared measure lineage, filter
  references, and measures hidden behind `VAR` declarations are validated
  without accepting partial-name or incidental-filter matches.
- One unsupported derived-metric insight can be deterministically rejected
  after targeted repair without discarding independently valid insights.
- Terminal verifier rejection is reported distinctly from exhausting the
  configured turn budget.
- Leap years, ISO weeks, partial periods, stale extracts, and mismatched source
  watermarks now abstain rather than producing inconsistent current-change
  claims.

## 0.4.2 — 2026-08-29 — secure Lakehouse analysis and verified insights

### Added

- **`delta_lakehouse` skill** — read-only discovery and analysis of Fabric
  Lakehouse Delta tables through mounted paths or OneLake `abfss://` paths.
  The `analytics` extra includes delta-rs as a fallback when DuckDB cannot
  download its Delta extension.
- **`LakehouseSource`** — discover whole Lakehouses, Tables, schemas, individual
  Delta tables, and Files in the trusted parent, then expose a compact catalog
  to isolated workers. `list_sources()`, `find_sources()`, and `query()` support
  multiple nested Lakehouse inputs without placing Fabric credentials in
  generated code.
- **Parent-side Lakehouse query broker** — workers select catalog aliases and
  submit SQL while the parent validates and executes bounded reads. Parsed
  relation authorization permits only bound aliases and CTEs derived from them;
  dynamic SQL, external paths, table functions, side-effecting functions, and
  unknown functions fail closed. Execution has a 30-second deadline, 256 MiB
  memory limit, disabled temporary spill, 10,000-row ceiling, incremental
  serialization, and a 5 MiB transfer ceiling.
- **`FileDestination`** — stage artifacts locally in generated code and publish
  them to canonical OneLake `Files` paths through the trusted parent. The
  broker supports legacy and DSPy workers, sealed snapshots, bounded DFS
  uploads, explicit overwrite semantics, remote-size verification, and cleanup
  without exposing storage tokens to the worker.
- **`RLMResult.inspect()` and `RunInspector`** — render a collapsed interactive
  turn timeline in Fabric/Jupyter or save a standalone escaped HTML report with
  code, output, errors, repairs, timing, token usage, and submission payloads.
- **`deep_insight_discovery` and `deep_insight_critic` skills** — produce
  source-agnostic analytical contracts with independently executable numeric
  evidence, typed diagnostic alternatives, conservative action-readiness
  gates, and adversarial decision-quality review.
- **Verified insight audit utilities** — strict source manifests, constrained
  aggregate DuckDB execution, precision-aware host auditing, fingerprinted
  checkpoints, critic-driven evidence closure, and bounded action synthesis.
  CRM Arena Pro and Olist transfer benchmarks exercise zero-turn replay and
  deterministic artifact hashes.

### Fixed

- Lakehouse roots and inferred scopes now reject traversal, encoded paths,
  mixed `Tables`/`Files` segments, duplicate separators, query strings,
  fragments, and malformed OneLake authorities.
- Explicit, discovered, and query-time Lakehouse catalogs reject
  case-insensitive duplicate names instead of silently selecting one source.
- An output named `inspect` remains available as `result.inspect`; call
  `RLMResult.inspect(result)` to inspect that run.
- Secure artifact publication supports Fabric Linux runtimes that omit
  `os.memfd_create` or Python `fcntl` seal constants, and streams sealed
  descriptors directly through the OneLake DFS API when AzCopy rejects
  `/proc/<pid>/fd` paths.
- Delta schema discovery supports both delta-rs `Schema.to_pyarrow()` and
  `Schema.to_arrow()`.

### Fabric verification

- Merged release code resolved and queried real Delta tables and Files through
  OneLake without a SQL endpoint.
- Contract-v3 discovery executed 13 host-audited numeric checks across 10
  CloudMetrics sources and persisted payload, audit, critic, and run artifacts
  to an `abfss://` Files path.
- `FileDestination` published Markdown and Excel files to OneLake; both were
  read back successfully, and the Excel workbook reopened with `openpyxl`.

## 0.4.1 — 2026-08-22 — safer workers and typed outputs

### Added

- **Typed task outputs** — pass concrete Python types as
  `outputs={"result": dict}` to reject wrong-shaped `SUBMIT` values and give
  the model repair feedback. Existing name-only lists remain supported.

### Fixed

- Worker subprocesses scrub provider keys and other secret-bearing environment
  variables by default, including when network blocking is enabled. Explicit
  `SecurityPolicy.disabled()` remains available for trusted workloads.
- Sandbox guidance documents the synchronous `predict_sync(...)` entry point
  and prediction-object field access.
- The PDF analysis skill reads filled form widgets before falling back to
  rendered-page analysis.

## 0.4.0 — 2026-08-07 — a semantic model is now an input

### Added

- **`SemanticModel(dataset, workspace=)`** — bind a Power BI semantic model as
  a task input and it arrives in the sandbox as a live handle: `.schema()` for
  tables, measures with their DAX and descriptions, and relationships in one
  call; `.dax("EVALUATE ...")` returning a DataFrame; `.measure(name,
  groupby=, filters=)` with no DAX to get wrong. Several models bind at once
  and route independently.

  Why a handle rather than instructions, measured on a 19-question eval: a
  task that named a semantic model but gave no way in scored 7/19, with most
  questions spending every turn looking for an entry point. A skill describing
  the entry point scored 18-19/19 at ~2.4k characters resent every turn.
  Binding the handle scored 18/19 three runs in a row with no skill loaded,
  and adding the skill on top changed nothing. Validation runs at
  construction, so a typo'd dataset name fails on that line instead of ten
  turns into a run.

- **`semantic_model` skill** — the fallback for a model passed as a plain
  name. Cut from 8.1k to 2.4k characters after measurement showed the entry
  point carried the entire effect; now points at `SemanticModel` first.

- **`fabric_rlm.grounding`** — `evidence_report()`,
  `submitted_without_evidence()`, `ungrounded_figures()`: read a finished
  trajectory and report answers the run never derived. On an externally
  audited benchmark submission this catches 4/4 of the voided trials plus two
  the audit missed. A diagnostic, never a gate: a pre-registered experiment
  showed the flags carry no accuracy signal beyond dropping runs at random,
  so use them to audit output before it ships, not to pick answers.

- **`examples/semantic_model/`** — a semantic model, a PDF, a CSV and a
  custom context skill combined into a formatted workbook written back to the
  lakehouse, with a scorecard that attributes each wrong cell to the source
  the agent failed to read. 44/44 checks with the handle bound.

### Fixed

- **The security message for `import fabric` no longer talks models out of
  sempy.** The `fabric` SSH package is denied, but the denial read as "network
  egress off" and models concluded the whole semantic-link path was
  unavailable - eleven refusals across three eval runs, each citing a blockage
  that did not apply. The message now names the SSH package and points at
  `import sempy.fabric`. Give-ups fell from 12 to 1 and the mean score rose
  11.7 to 16.5 before any skill was involved.

- **`SemanticModel.schema()` asked sempy for a measure column that does not
  exist.** sempy names it `Measure Description`, not `Description`; the wrong
  name dropped every measure description silently. Found in a trace where the
  model hit the same mistake, got the error our code never saw, and recovered.

- **Worker subprocesses no longer open console windows on Windows.** A
  supervised benchmark run left dozens of empty terminals on the desktop.
  `CREATE_NO_WINDOW` on both spawn sites; Linux (including Fabric notebooks)
  takes the unchanged path.

- **A worker killed by threaded code now says so.** Concurrency around native
  calls (duckdb, drivers, `predict_sync`) can crash the worker fatally with a
  bare protocol error, and the model's retry repeats the pattern. Both death
  paths now name the cause and the fix when the executed code contained
  concurrency markers. The crash itself is tracked as #46.

## 0.3.5 — 2026-08-04 — a 17-second startup cost, removed

First release published to PyPI since 0.3.2, so it also carries everything from
0.3.3 and 0.3.4 (worker timeout recovery, `skills_as_cards`, `block_network`).

### Fixed

- **The worker no longer imports dspy to read a version string.** The JSON-RPC
  startup self-test filled in a `dspy_version` diagnostic field by importing
  dspy, which costs about 18s on a cold interpreter (litellm alone is ~9s), and
  that self-test runs on every `SubprocessPythonInterpreter` start. Every run on
  the dspy engine path paid it.

  | | before | after |
  | --- | --- | --- |
  | `SubprocessPythonInterpreter` startup | 16.8s median, 34.7s max | **0.9s median** |
  | `Interpreter` startup (no self-test) | 0.6s | unchanged |

  `importlib.metadata.version("dspy")` returns the identical string in ~0.5s
  without importing the package. dspy is still imported lazily where it is
  actually used. The gap had been invisible because the legacy `Interpreter`
  does not run this self-test.

  It was also the flakiest thing in the test suite: a 60s startup budget against
  a 16.8s median is only ~2x headroom, so under load startups intermittently
  exceeded it and different tests failed on each run.

- **The behaviour CI gate no longer charges environment failures to the pull
  request.** A retired free-tier model slug 404'd on every question and each 404
  was recorded as a failing question, so the gate reported five per-qid
  regressions on every PR for a week. Nothing had regressed. A question that
  never reached the model cannot be evidence for or against a regression, so the
  gate now aborts on `runner_error` as well as `auth` and names the environment
  as the cause. `wrong_answer` and `infra` still count, deliberately.

### Added

- **`RLMResult.report()`** — a deterministic account of what a run did.
  `report()` prints it, `report(as_dict=True)` returns the facts. No model is
  consulted and nothing is inferred, so it is safe in CI and on a long
  trajectory.

  Covers whether the run submitted and why not, turns against the ceiling,
  tokens split by prompt/completion/cached/reasoning, the LM-versus-worker time
  split, repair turns (submitted, but not on the first attempt), the slowest
  turn, the last turn that errored, and hints — each tied to a fact printed
  above it rather than guessed.

- **`RLMResult.ran_any_code`** — False when no turn executed. Note this does not
  mean the model was unreachable: an LM error raises out of `run()`, so holding
  a result means the LM answered. It means the model replied without emitting a
  code block.

## 0.3.4 — 2026-08-04 — opt-in network isolation for the worker

### Fixed

- **Held dspy below 3.3.** dspy 3.3.0 shipped on 2026-08-03 and broke the engine
  path within hours on a ceiling that allowed it: RLM's `max_iterations` became
  `max_iters`. Three further changes need handling before the ceiling goes back
  up — `Image`/`Audio`/`File` no longer perform implicit I/O, interpreter
  failures became terminal rather than auto-restarting, and LM errors now raise
  DSPy exception classes rather than provider ones. Tracked in #36.

- **Removed a flaky test.** The `deep_recursion` hang shape raced a 2s timeout
  against a C-stack overflow, and which won depended on platform and Python
  version: it failed on 3.10, passed on 3.11, then failed on windows/3.13.
  Worker death is covered deterministically by the tests that kill the process
  outright, so nothing is lost.

### Added

- **`block_network=True` refuses network egress from the worker.** Off by
  default: notebook code legitimately calls APIs, so this is opt-in. Loopback is
  permitted, every other destination is refused.

  `SecurityPolicy` already denies `requests`, `httpx`, `urllib.request.urlopen`
  and `socket.socket`, but it is a **static check on the model's own source**, so
  any library that wraps the request goes straight through it — the model's code
  never names a denied symbol:

  ```python
  from datasets import load_dataset     # nothing denied appears here
  load_dataset("ag_news")               # ...and it fetches anyway
  ```

  Found on DataAgentBench, where trials answered a classification question by
  downloading the public dataset and reading its gold label column. The question
  asks for a category that does not exist in the sanctioned stores, so fetching
  the labels was both the easy route and exactly what the grader compared
  against. Stopping that needs a check at the socket layer at runtime, where it
  does not matter which library made the call.

  The guard is on `connect`, not on socket construction. Denying `socket.socket`
  and `socketpair` outright kills the worker before its first turn: asyncio's
  Windows ProactorEventLoop builds its self-pipe with `socket.socketpair()` and
  `nest_asyncio` creates a loop at import. That pair is two already-connected
  loopback sockets and cannot reach a remote host.

  An explicit `sub_lm=` is rejected at construction, since sub-LM calls are made
  from inside the worker. The implicit `sub_lm_spec` set from `lm` is unaffected.

  Verified 11 of 11 on a Fabric capacity, with both an OpenRouter key and the
  built-in `FabricLM`, including a check that the worker stayed sealed during a
  run whose LM was reachable, and a control confirming egress was available when
  the flag was off — so the blocked cases were falsifiable rather than passing by
  default. See `examples/notebooks/verify_block_network_fabric.ipynb`.

  Three limits, documented and tested rather than implied away. `_socket.socket`
  used directly is not guarded, because a C type's methods cannot be replaced
  from Python. Nothing here constrains a C extension issuing raw syscalls, so
  only an OS-level control can support a claim that a process *cannot* reach the
  network.

  And most importantly: **this is not contamination prevention.** It stops data
  arriving over the network; it does nothing about data already on the machine.
  Redirecting `HF_HOME` does not close that either, because it only changes
  where the *library* looks by default. Measured on a real DataAgentBench trial
  with the guard on and `HF_HOME` redirected: the model was refused at
  `urlopen`, went looking for cache directories, then read
  `os.path.expanduser("~/.cache/huggingface")` by absolute path and loaded all
  127,600 labelled rows without touching the network. If a dataset must be
  unreachable, delete it from disk — and keep auditing what the model actually
  ran, because that is what catches it when prevention fails.

## 0.3.3 — 2026-08-02 — a worker that stops responding no longer ends the run

### Changed

- **A worker timeout is now recoverable instead of fatal.** A timeout kills the
  worker, and the runtime used to return immediately with
  `failure_reason="worker_timeout"`, discarding every turn already completed —
  including tasks that had finished the analysis and were formatting output.
  `recover_worker_timeouts` (default `1`) restarts the worker, re-applies the
  sub-LM configuration, re-binds inputs, and tells the model its namespace is
  gone and the approach was too slow. Recovery consumes a turn, so `max_turns`
  still bounds it, and `recover_worker_timeouts=0` restores the old behaviour.

  The default is one rather than more because each allowed recovery costs up to
  one timeout period on a run that was doomed anyway, and the default timeout is
  300s: a budget of two risks a ten-minute hang in an interactive notebook.

  Timeouts are rare but total — 23 of 983 task runs across four full
  AgenticDataBench runs, each losing the whole run. Tested across six ways a
  worker stops responding (sleep, busy loop, huge allocation, runaway string
  concatenation, blocking stdin read, deep recursion), plus recovery after
  several good turns, `File` inputs, sub-LM reconfiguration, skills loaded, and
  the turn budget. Verified on a Fabric capacity: the worker respawns inside the
  Synapse executor, Lakehouse `File` inputs re-bind, and `FabricLM` keeps working
  across the restart. No measurable effect on runs that never time out.

  The same recovery now also covers a worker that **dies** rather than hangs —
  an out-of-memory kill, a segfault, or code calling `sys.exit()` — which raised
  `WorkerProtocolError` and ended the run untouched. The model is told it
  exhausted memory rather than that it was too slow, since the two need opposite
  advice, and the new `failure_reason="worker_died"` keeps a crash distinct from
  a timeout in the trajectory.

  Worth stating plainly: this branch is defensive, not measured. Across 9,092
  recorded benchmark task runs there are 2 worker timeouts and **zero** worker
  deaths, so it is not expected to move any benchmark number. It was found
  because Python 3.10 segfaults on deep recursion before a timeout can fire
  (3.11 moved frames to a heap-allocated data stack, so it merely runs slowly),
  and the death path was covered by tests that kill the worker outright so it is
  exercised on every supported Python rather than only 3.10.

### Added

- **`skills_as_cards`**, which advertises the chosen skills as one-line cards and
  lets the model call `load_skill(name)` if it wants a body, instead of
  preloading the full playbook. A preloaded body is resent on every turn, so its
  cost is roughly size times turns; for `excel_modify` the prompt drops from
  19,606 to 4,945 characters before any per-turn multiplier. Default `False`;
  under measurement on SpreadsheetBench before any change to that default.

## 0.3.2 — 2026-08-01 — verified execution, and a timeout that fires

### Added

- **`verified_task`: blind double-solve with structural agreement and
  reconciliation.** Solves a task twice in fresh contexts, compares the answers
  in code (exact numbers, identical semicolon-list item sets, normalized prose),
  and on disagreement reconciles in a third fresh context that must re-derive
  from the data. Measured before it was written: +0.076 stratified Pass@1 over
  single solves on a 54-query, three-runs-each benchmark A/B, at about 2.9x
  tokens, with the reconciler choosing correctly on 68-77% of decisive pairs.
  For read-only analytical tasks whose product is a determinate answer; the
  module docstring states where it does not apply (side-effect tasks such as
  workbook modification run the task multiple times; generative output never
  agrees structurally; consistent errors agree and pass). Returns a
  `VerifiedResult` carrying the winning `RLMResult`, the verdict, both
  candidate answers, and every attempt for token accounting.

- **`contrib-skills/financial_documents`, a domain playbook that is not installed
  with the package.** Reporting conventions for 10-K, 10-Q, annual reports and
  earnings releases: parentheses as negative, scale stated in a header rather than
  beside the number, fiscal against calendar year, adjacent three-month and
  twelve-month columns, subtotal and contra rows, restatements and non-GAAP
  measures. Load it by pointing a `SkillLoader` at `contrib-skills/`. It declares
  `pdf_document_analysis` as a dependency, so naming it pulls in the page mechanics
  as well.

  It sits outside the wheel because the evidence is narrow. On a 40-question set
  built from tables in 24 SEC filings, each asking which row holds the largest or
  smallest value in a named column, it scored 35 of 40 against 32 for
  `pdf_document_analysis` alone, fixing 5 and breaking 2. The subset with a negative
  candidate value went from 16 of 20 to 19 of 20, and the traces confirm the
  mechanism: the baseline picked "Boeing Capital $199 million" as the smallest value
  in a column that contained "(231)", reading the bracketed negative as positive.
  The result is not significant (two-sided McNemar, p = 0.453), and because the set
  was built to concentrate sign handling, the gain does not transfer to a general
  accuracy figure. On FinanceBench the expected effect is closer to one question in
  150. See `docs/contrib-skills.md`.

  A generic "enumerate the candidates and compute the extreme" rule was tested on
  the same set and rejected: 32 of 40, three fixed and three broken, because it led
  the model to pull rows in from neighbouring tables. It is not shipped anywhere.

### Fixed

- **A dropped LM connection now errors instead of hanging forever.** Resolved
  LMs carry a default per-request `timeout` (600s; override by passing
  `timeout` in the spec). Without it, a connection the provider silently drops
  blocks the worker indefinitely: the task-level timeout is checked between
  turns, and a blocked HTTP read never returns to let it run. Observed as five
  concurrent workers frozen for 35+ minutes with zero CPU and zero API spend.
  Regression-tested against a server that accepts requests and never responds.

- **Token totals no longer omit LM calls that produced no runnable code.** When a
  response arrives truncated mid-fence, or contains prose instead of a `python`
  block, the runtime retries it rather than executing it. Those attempts never
  became a `TurnRecord`, and since token totals were summed from the trajectory's
  turns, the provider's charge for them vanished from `total_prompt_tokens` and
  `total_completion_tokens`. Any run that hit either guard under-reported its
  spend; a run where *every* response hit one reported `n_turns=0` and no tokens
  at all while having taken a minute and cost real money. Those calls are now
  carried alongside the trajectory and included in the totals. The turn-exhaustion
  warning also reports attempts rather than recorded turns, so it no longer says
  "ran out of turns after 0 of 16".

  No behaviour change: the trajectory itself is unchanged, so the stuck-loop
  circuit breaker sees exactly what it saw before, and nothing about what the
  model is shown or generates is affected. Only the reported numbers move, and
  they move toward what the provider actually billed.

## 0.3.0 — 2026-07-28 — reasoning effort pinned, sharper document guidance

### Changed

- **Reasoning models now default to `reasoning_effort="medium"`.** Previously
  the effort was left unset, which meant the provider decided. Providers
  disagree: the same gpt-5 family model reached through one route arrived with
  reasoning on and through another with it off, using identical library code.
  A reasoning model that is not reasoning does not fail loudly, it fails
  plausibly. On a two-source document task, gpt-5.1 with the provider default
  scored 4.0 of 12 document checks and fabricated four correctly shaped rows of
  economies and dates that appear nowhere in the source; with the effort pinned
  it scored 12.0 of 12 across three consecutive runs. Pinning the value makes a
  run reproducible across routes instead of inheriting endpoint behaviour.

  This affects anyone on the gpt-5 or o1/o3/o4 family who did not set
  `reasoning_effort` explicitly: expect different cost, latency, and quality
  after upgrading. Chat models (`gpt-4o`, `gpt-5.1-chat`) and non-OpenAI
  backends are unchanged. Pass `reasoning_effort` to `FabricLM`, `OpenAILM`, or
  a dict LM spec to override, including back to a provider default.
- **`pdf_document_analysis` now teaches locate-then-read.** The skill covered
  page rendering and vision, but said nothing about pulling one figure out of a
  long report, and models reliably chose the worst method: flatten every page
  into one string and regex for a number near a label. Section labels and stock
  phrases repeat across a document, so the first match usually comes from the
  wrong section, and the answer that came back was confident and wrong. The
  skill now says to locate the passage, print it, read it, and keep the source
  sentence next to the value, and it documents an x-coordinate fallback for the
  borderless tables `find_tables()` cannot parse. Measured on a two-source
  extraction task with gpt-5-mini, three runs per variant: 3.0 of 12
  document-dependent checks before, 8.3 after.
- The free-tier behaviour gate targets `nvidia/nemotron-3-ultra-550b-a55b:free`.
  OpenRouter retired `openai/gpt-oss-120b` from its free tier, so the old slug
  404s on every question and the gate failed for anyone running the suite. The
  new model calibrates at 5/5 on all five questions. Override with
  `BEHAVIOR_SECONDARY_FREE_MODEL`; the old baseline is retained.
- The skill-authoring documents moved out of the installed package. `SkillLoader`
  listed `PLAYBOOK_CONTRACT` and `SKILL_TEMPLATE` as if they were loadable
  skills, so a model calling `list_skills()` saw two entries that are
  documentation about writing skills. They now live at
  `docs/authoring-skills.md` and `docs/skill-template.md`, and `list_skills()`
  returns the seven real skills.

## 0.2.9 — 2026-07-25 — installable on Windows and macOS

### Fixed

- **`pip install fabric-rlm` failed on Windows and macOS.** litellm, which
  arrives through dspy, stopped publishing Windows and macOS wheels at 1.92.0,
  so pip fell back to compiling its sdist and demanded a Rust toolchain. The
  dependency is now pinned to `litellm>=1.64,<1.92`, the last line with a
  universal wheel. Linux and Fabric notebooks were unaffected; every other
  platform could not install the package at all.

## 0.2.8 — 2026-07-23 — lossless SUBMIT payloads and release hardening

### Fixed

- **Lossless final `SUBMIT` payloads.** Final supported strings and collections
  are no longer truncated by the namespace snapshot limits. Iterative state
  remains bounded, while final payloads have a configurable 64 MiB default
  byte cap and fail explicitly when exceeded.
- **`dspy` dependency range.** Pinned to `dspy>=3.2.1,<3.4`. The subprocess
  interpreter mirrors dspy's kwargs-only tool dispatch (3.2.0) and JSON-RPC
  `CodeInterpreter` messaging (3.1.3), so the previous `>=3.1.2` floor sat below
  the protocol the runtime actually tracks. The ceiling guards against churn in
  dspy's experimental `RLM` / `CodeInterpreter` internals.
- **Example notebooks install from PyPI.** Every notebook now installs
  `fabric-rlm` from PyPI (tutorials unpinned, benchmark notebooks pinned to the
  release) instead of a TestPyPI pre-release index.

### Added

- Regression coverage and documentation for 10,000-character strings,
  500-row outputs, UTF-8 byte boundaries, invalid limits, bounded snapshot
  compatibility, and both interpreter surfaces.
- **New example notebooks** — a minimal contract-comparison walkthrough and a
  Spark-log root-cause analysis. The existing PDF notebooks were trimmed to
  focused, ready-to-import recipes.

### Changed

- `py.typed` is now explicitly declared as package data so type checkers pick up
  the inline types regardless of the build toolchain.
- Documentation and packaged skill playbooks were cleaned up for the public
  release (corrected built-in names, install commands, and cross-references),
  and the `excel_modify` playbook gained value-storage and
  verify-by-recomputation guidance.

## 0.2.7 — 2026-06-16 — opt-in Excel workbook structure context

### Added

- **Opt-in Excel workbook context.** Added
  `add_excel_workbook_context(...)`, a convenience wrapper that prepends a
  read-only workbook context block to an RLM task. The default mode is
  `mode="structure"`: sheet names, target ranges, dimensions, merged/formula
  counts, and headers only — no sample row values. `mode="full"` remains
  available when callers explicitly want compact data-only samples.

### Fixed

- **Workbook context for create-output-sheet tasks.** Structural context now
  reports missing target sheets and summarizes the existing workbook instead of
  failing before the model can create the requested sheet. Post-submit artifact
  validation remains strict.

## 0.2.6 — 2026-06-15 — Excel artifact validation and public workbook skill

### Added

- **Context-aware RLM output validation.** `RLM` now accepts
  `output_validator_context`, allowing validators to inspect runtime artifacts
  such as saved workbooks after `SUBMIT`.
- **Excel artifact helpers.** Added `fabric_rlm.excel_artifacts` with reusable
  target-range parsing, target-cell iteration, and artifact sanity validation
  for workbook-writing tasks.

### Changed

- **`excel_modify` is the public workbook-editing skill.** The improved Excel
  modify guidance, including literal-value guardrails and the large-range /
  sheet-level protocol, is now packaged as `excel_modify` instead of the
  experimental `excel_modify_gpt5` name.

## 0.2.5 — 2026-06-11 — task-constructor alias

### Added

- **`RLM.task(...)` ergonomic alias.** `RLM.task(task, inputs=..., outputs=..., **kwargs)`
  now constructs the same inline-task runtime as `RLM.from_task(...)`, preserving
  subclass dispatch, copied caller inputs/outputs, constructor kwargs, and
  legacy-engine deprecation warning behavior.

## 0.2.4 — 2026-06-10 — golden-trajectory replay + loop-robustness

### Added

- **`ReplayLM` golden-trajectory harness (`fabric_rlm.replay_lm`).** Re-run the
  *real* `RLM.run` loop from a recorded `Trajectory` with **zero API calls and
  zero subprocesses**. A recording already stores both sides of every turn — the
  raw LM response (`TurnRecord.response_text`) and the worker outcome
  (`stdout`/`error`/`submitted`/`state`/`submit_payload`) — which is everything
  needed to drive the loop again. Two in-memory fakes, `ReplayLM` (feeds recorded
  responses) and `ReplayInterpreter` (feeds reconstructed `ExecResult`s), plus a
  one-call `replay_trajectory(rlm, trajectory)` helper, give deterministic
  end-to-end regression coverage of feedback formatting, validation, repair
  routing, and stop conditions. Divergence is the signal: if a change makes the
  loop request more turns, execute different code, or stop earlier than the
  recording, `replay_trajectory` raises `DivergenceError`. New public exports:
  `ReplayLM`, `ReplayInterpreter`, `replay_trajectory`, `DivergenceError`.
  Frozen example recordings live in `examples/trajectories/`; see
  `examples/replay_golden_trajectories.py` for the how/why and the Lakehouse
  save/load path. Targets the default `engine="v6-custom"` loop; verifier-free
  recordings are the supported golden path.

### Changed

- **Robust code extraction.** A response with no `python` code fence is no longer
  shipped to the worker as bare prose (which burned a turn on a `SyntaxError`);
  the loop now short-circuits with a clean "resend one complete block" signal,
  the sibling of the existing truncated-fence guard. Extraction also now selects
  the **last** complete fenced block rather than the first, matching how models
  revise — a sketch in block 1, the corrected code in block 2.
- **Repair-turn diversity nudge (default `escalating`).** Validation / verifier
  repair feedback can append a line nudging the model to recompute a repeatedly
  failing field via a different method and cross-check before re-submitting —
  operationalizing the REFLECT principle *within* an attempt to break stuck
  loops. Modes via `FABRIC_RLM_REPAIR_NUDGE`: `off` / `static` / `escalating`
  (default; the line appears only after a field has failed at least twice).

### Fixed

- **PDF-skill parallel gather hardening.** The `pdf_document_analysis` skill's
  map-reduce snippet now bounds concurrency and uses `return_exceptions=True` so
  one bad chunk can't sink the whole gather, and documents the
  split → `predict()` per chunk → gather → synthesize idiom.

### Other

- **Opt-in failure-time truncation hint** (`FABRIC_RLM_TRUNCATION_HINT`, default
  **off**). When a turn's stdout overflows the feedback budget, an opt-in hint
  tells the model it is not seeing all output and should aggregate in Python or
  chunk + `predict()`. Shipped default-off as a tested safety net (A/B showed the
  dominant "sample-and-guess" failure produces small stdout that never triggers
  it, so no benefit is claimed by default).

## 0.2.3 — 2026-06-10 — tail-preserving feedback truncation + docs

### Changed

- **Tail-preserving feedback truncation.** When a turn's stdout, stderr, or
  error traceback overflows the feedback budget, the runtime now keeps both the
  **head and the tail** (with a `... (N chars omitted) ...` marker) instead of
  the head only. Python output is bottom-loaded — the final `print`, the last
  progress line, and a traceback's terminal `SomeError: ...` line all live at
  the end — so head-only truncation routinely dropped exactly the value the
  model needed. Same token budget, strictly more useful signal. Tail ratios are
  tunable via `FABRIC_RLM_STDOUT_TAIL_RATIO` (default 0.4),
  `FABRIC_RLM_STDERR_TAIL_RATIO` (0.5), and `FABRIC_RLM_ERROR_TAIL_RATIO` (0.7);
  set any to `0` to restore the previous head-only behavior. Error/traceback
  feedback now honors a dedicated `ERROR_FEEDBACK_LIMIT` instead of an inline
  2000-char head cut.

  A/B validation on overflowing-output tasks (OpenRouter, identical code, env
  flag flipped) showed combined task success rising from **25% → 87.5%** across
  `gpt-5.1` and `gpt-4.1`, with a clean causal chain: the answer token was
  visible in feedback ⇒ the model passed; truncated away ⇒ it failed. No
  regressions where output fit the budget (e.g. frame-collapsed tracebacks).

### Docs

- **QUICKSTART §3a (Fabric):** added a "Which models can I name?" note pointing
  to the Microsoft Learn Prebuilt AI models list (no hardcoded model names), so
  users pick model strings their workspace actually exposes.

## 0.2.2 — 2026-06-09 — engine consolidation + public-release hardening

### Fixed

- **Skill router: `from_task` blind spot.** Routing scores keywords against
  bound input values; when the user's question lives in `task=` and the
  inputs are just file paths, every run used to elect the same always-on
  bundle. Routing now falls back to the task text **only when the inputs
  carry zero keyword signal**, preserving the original menu-inflation
  protection for benchmark-style signatures. New trajectory metadata:
  `router_used_task_text_fallback`.
- **v7/dspy engine: token accounting.** `RLMResult.total_prompt_tokens` /
  `total_completion_tokens` / `total_cached_tokens` /
  `total_reasoning_tokens` are now populated for `engine="dspy"` runs by
  harvesting the dspy `lm.history` usage entries (previously always `None`,
  including for `engine="auto"` + `tools=` users).
- **Security rejections no longer wipe reported state.** A parent-side
  `SecurityPolicy` rejection fabricates a failed turn without consulting the
  worker; it previously carried `state={}`, erasing `final_state` and the
  turn's state snapshot. Such turns now carry the last real snapshot and a
  new `ExecResult.reached_worker=False` marker.
- **Legacy `Interpreter` stderr drain.** The v6 interpreter now pumps the
  worker's stderr on a background thread (ring-buffered, last 200 lines).
  Previously stderr was only read at exit, so chatty native libraries could
  fill the OS pipe buffer and deadlock the worker into a spurious
  `WorkerTimeout`.
- **CLI:** `--max-turns` / `--timeout` now override the task file even with
  falsy values; added `--version`, `--engine`, `--verbose`; `engine`,
  `verbose`, `enable_router`, `max_active_skills` are honored from the task
  JSON; unknown task-file keys warn instead of being silently dropped.
- README 30-second example used a non-existent `rlm.run(prompt=...)`
  signature; corrected to `RLM.from_task(...)`.
- **`SubprocessPythonInterpreter` startup timeout** raised 15s → 60s
  (override per-instance via `start_timeout=` or globally via
  `FABRIC_RLM_START_TIMEOUT`). Cold CPython spawns on loaded machines/CI
  runners legitimately exceed 15s; genuinely broken installs still fail
  fast because the dead worker closes stdout immediately.
- **Behavior CI gate: credential failures are no longer reported as model
  regressions.** 401/403 and `AuthenticationError`-shaped failures get a
  new `auth` error class and abort the gate immediately with a
  "fix OPENROUTER_API_KEY" message instead of failing every qid.

### Packaging / docs

- Version is now single-sourced from `fabric_rlm.__version__` (pyproject
  reads it via `[tool.setuptools.dynamic]`); README/QUICKSTART no longer
  hardcode wheel versions.
- Added PyPI metadata (`[project.urls]`, classifiers, keywords), and
  `CONTRIBUTING.md` / `SECURITY.md` (threat model + private reporting).
- Example notebooks no longer embed real workspace/lakehouse IDs
  (placeholders instead).
- Removed references to non-shipped design/eval documents from README and
  QUICKSTART; QUICKSTART troubleshooting now documents the real
  `failure_reason` values.
- CI: test matrix expanded to Python 3.10–3.13 on ubuntu + windows; added a
  packaging job (`python -m build`, `twine check`, wheel-content assertions
  for skills markdown and `py.typed`).

### Engine selection (consolidation)

- **`engine="auto"` is the new default** for `RLM(...)`. It picks `"dspy"`
  when a non-empty `tools=[...]` iterable is supplied, otherwise `"default"`.
  Existing code that passes no `tools=` (or an empty `tools=[]`) and didn't
  pass `engine=` keeps identical behavior (resolves to the same canonical
  engine as before).
- **New public aliases**: `engine="default"` (= legacy `"v6-custom"`) and
  `engine="dspy"` (= legacy `"v7-dspy"`).
- **Deprecated**: passing `engine="v6-custom"` or `engine="v7-dspy"`
  directly (or via `RLM.from_task(...)`) now emits a `DeprecationWarning`
  pointing at the user's call site. Behavior is unchanged — both still
  resolve to the same canonical engines. **Removal not before v0.3.**
- Migration: prefer `engine="auto"` (recommended), or explicit
  `engine="default"` / `engine="dspy"`. Adaptive (`engine="adaptive"`)
  is unaffected and remains experimental.
- Internal: `_normalize_engine_name` is pure (no side effects); the
  deprecation warning is emitted at public entry points (`__init__`,
  `from_task`) with correct stacklevel for both call paths. The adaptive
  inner-RLM factory translates canonical inner engines to public aliases
  before constructing inner attempts to avoid library self-warning.

## 0.2.1 — `excel_modify` skill + SpreadsheetBench head-to-head

### New

- **`fabric_rlm/skills/excel_modify.md`** — task-agnostic skill for in-place
  modification of `.xlsx` workbooks via openpyxl. Triggered by keywords
  `xlsx`, `workbook`, `openpyxl`, `sheet`, `cell range`, etc. Bakes in two
  recipes that fixed real benchmark failures:
  1. **Two-load discovery**: load the workbook with `data_only=False` for
     editing and `data_only=True` for reading source values, so cells whose
     source is a formula return numbers rather than the literal `'=D3+F3'`
     string.
  2. **Mandatory verify-by-reload**: after `wb.save()`, reload with
     `data_only=True` and assert no cell in the target range is `None` or
     starts with `=`. Catches the formula-instead-of-value failure class.

### Bench

- **SpreadsheetBench Verified-400 head-to-head** (50Q stratified subset),
  reproducible end-to-end on Fabric:
  - Strategy A (gpt-5 single-shot, dspy.Predict + subprocess exec):
    23/50 = 46.0%, $2.21
  - Strategy F (gpt-4.1-mini + RLM + Python interpreter + `excel_modify`):
    21/50 = 42.0%, $0.51 (4.3× cheaper, 2.3× faster wall-clock)
  - Union pass rate: 29/50 = 58.0%
  - Reproduce with the benchmark notebooks in `examples/notebooks/`
    (`spreadsheetbench_400_openrouter_minimax_mlflow.ipynb` and
    `ssb400_minimax_m3_fabric_repro.ipynb`).

## 0.1.11 — PLAN / VERIFY / REFLECT (PVR) contract

**Bug fix (dev6):** `Trajectory.__bool__` now explicitly returns `True`. Previously a `Trajectory` with zero turns evaluated as falsy because `__len__` was defined and Python falls back to it for truthiness, causing downstream `if traj: ...` guards (in benchmarks and result-collection helpers) to silently discard the trajectory's metadata — including the entire `adaptive` payload. Found while diagnosing a 5-way comparison where `EffortLadderPolicy` appeared to record 0 attempts on every question.

The default `core` skill now ships with an explicit **PLAN / VERIFY /
REFLECT** contract, and the adaptive engine injects synthesized REFLECT
context on every failed attempt (not only validator rejections).

- **PLAN** — model decomposes the task before writing worker code.
- **VERIFY** — model self-checks the answer against task constraints
  before calling `SUBMIT(...)`.
- **REFLECT** — when an attempt fails (validator rejection, worker
  error, timeout), the next attempt receives a structured
  `PRIOR_ATTEMPT_FEEDBACK` block containing the failure reason and the
  prior answer to consider.

**Generalization ablation** (4 cases × 2 conditions, fresh bandit state):

| case | OFF pass | ON pass | OFF→ON attempts | OFF→ON tokens |
|---|---|---|---|---|
| easy-math, easy-csv | pass | pass | 1→1 | small overhead, no regression |
| Backprop_hard (solvable, multi-step) | fail ladder exhausted | pass rung 3 | 7→3 | 1.28M→413K (-68%) |
| VLIW_hard (capability ceiling) | fail | fail | 6→6 | 294K→242K (-18%) |

**OOD ablation** (structured extraction outside training distribution):

| case | OFF pass | ON pass | OFF→ON attempts | OFF→ON tokens | OFF→ON elapsed |
|---|---|---|---|---|---|
| rfp-extract (4 fields from RFP PDF text 100KB) | pass | pass | 2→1 | 18K→4.9K (-74%) | 60.6s→7.9s |
| spark-extract (5 fields from Spark log JSON 200KB) | pass | pass | 1→**3** | 6K→40K (+528%) | 10.1s→176s |

The Spark-log case revealed a new failure mode: PVR's VERIFY clause can spuriously
self-reject a correct first answer, amplifying retries on tasks the model would otherwise
nail cold. Correctness is preserved (always passes eventually), but the cost can be 10×+.

**Refined heuristic — when to enable PVR**:

| profile | PVR? |
|---|---|
| Easy single-step the model nails cold (Spark log triage, simple lookups) | optional/off — VERIFY can spuriously self-reject |
| Multi-field extraction with strict format (RFP) | **on** — PLAN/VERIFY enforces completeness |
| Multi-step reasoning, derivations, code synthesis | **on** — REFLECT prevents brute-force-and-fail |
| Capability ceiling | optional — fails marginally cheaper but doesn't rescue |

PVR is **on by default**. Disable with `FABRIC_RLM_PVR=0` for token-
sensitive batch workloads on known-trivial tasks.

### 0.1.11.dev4 — Trajectory capture + diagnostic finding on PVR mechanism

Added opt-in turn capture (`FABRIC_RLM_CAPTURE_TURNS=1`) on
`AttemptRecord.to_summary()` so notebooks can persist per-turn
`response_text` / `code` / `stdout` for offline analysis. The OOD
ablation notebook now writes per-condition trajectory JSONL files into
the run directory.

**Diagnostic finding (important; informs how PVR is described):**
captured trajectories show the model emits **zero** `## PLAN` /
`## VERIFY` markers across every turn of every attempt at
`reasoning_effort='minimal'`, even when the skill prompt is delivered
verbatim. Strengthening the skill rules with explicit "MUST",
"contract violation" language, and worked code examples did **not**
make the model comply (verified on `pvr_ood_ablation` run
`20260502-150436-6b9f67`).

The measurable wins from PVR (Backprop -50% attempts / -16% tokens; RFP
-74% tokens) therefore come from the **inter-attempt REFLECT injection**
in `AdaptiveRunner._with_feedback` (the `[ADAPTIVE: prior attempt
rejected]` block fed into retries) and the runtime's post-SUBMIT
reflection turn — *not* from PLAN/VERIFY scaffolding in the skill
prompt. The PLAN/VERIFY rules in `core.md` are kept as-is for higher
reasoning-effort runs (where the model may still honor them) but the
operative contract today is REFLECT. A future change should either (a)
add a runtime check that re-prompts when `## PLAN` / `## VERIFY`
markers are missing, or (b) rename the documentation to "REFLECT
contract" and drop the PLAN/VERIFY claims.

Also fixed a `TypeError: unhashable type: 'slice'` crash in the OOD
notebook's `attempts_summary` cell when an attempt's `answer` payload
is a dict instead of a string.

## 0.1.10 — Experimental `engine="adaptive"`

**Adaptive escalation, opt-in.** When a validator rejects an attempt, an
outer meta-controller climbs a fixed ladder until either the validator
passes or a budget is exhausted:

```
rung 0 → baseline (cheap LM, default turns)
rung 1 → more_turns
rung 2 → more_effort         (medium reasoning_effort)
rung 3 → best_of_N           (parallel rollouts, same cheap LM)
rung 4 → strong_lm           (escalate to e.g. gpt-5, reasoning_effort=high)
```

Two surfaces:

- `RLM(engine="adaptive", adaptive={...})` — thin wrapper, headline ergonomics.
- `from fabric_rlm.experimental import AdaptiveRunner, LadderPolicy, Budget` —
  power user; gives you the per-attempt `AttemptRecord` log.

Notes:

- Inner engine defaults to `v6-custom`; pass `inner_engine="v7-dspy"` to switch.
- Emits a `UserWarning("experimental")` once at construction.
- Per-run summary attached to `result.trajectory.metadata["adaptive"]` with
  `winner_rung`, `attempts: [{rung, ...}]`, `stop_reason`, `elapsed_seconds`.
- Bench harness lives at `bench/adaptive/` (4 modes × 3 buckets, including a
  Spark RCA case). Baseline mechanics verified end-to-end on real LMs:
  `MFMC_hard_1` failed at gpt-4.1-mini, the ladder escalated through
  `[0,2,3,3,3,4,4,4]`, and gpt-5 at rung 4 solved it.

**Legacy / deprecated**: nothing removed; the only API surface added is
`engine="adaptive"` plus the `experimental` submodule. Nothing else is
behaviourally affected.

**Tests**: 53 passing — 32 policy + 8 runner + 7 runtime + 2 eval + the
existing 3 legacy + 1 spot-check.

## 0.1.9 — Slim core release

**Repository slimming.** This release ships `fabric-rlm-core`, a clean
distribution containing only the production runtime and the proven skills:

- **Kept:** runtime, subprocess interpreter (with the v0.1.8 asyncio fix),
  LM backends (OpenAI / Anthropic / FabricLM), skill loader & router,
  trajectory + replay, validators, and the skills `core`,
  `validation`, `error_handling`, `data_exploration`,
  `pdf_document_analysis`.
- **Removed:** `fabric_rlm.adaptive` (deprecation shim),
  `fabric_rlm.experimental.*` (AdaptiveOrchestrator),
  `fabric_rlm.skill_distiller`, the `benchmarks/` package,
  longcot signatures/schemas/skills, and all `_*` repo-level scratch.
- **API:** no breaking change for code that uses only the documented public
  API. `from fabric_rlm import AdaptiveOrchestrator` no longer works (it
  has been deprecated since 0.1.7 and only re-exported via a shim).
- **Docs:** new `README.md`, scrubbed `QUICKSTART.md` (no §9b Adaptive
  escalation, no longcot examples), `LICENSE` (MIT), `.gitignore`,
  `.gitattributes`.
- **Tests:** dropped longcot/adaptive/v6-skill-verifier suites; the kept
  ~33 tests cover runtime, interpreter, validators, serializers, replay,
  LM, skill loader/router, and the playbook contract.

## 0.1.8 — Asyncio fix in the subprocess worker

Fixed `_worker.py` calling `asyncio.run()` from inside an already-running
event loop in async-host environments (Fabric notebooks, Jupyter). Worker
now detects an existing loop and awaits in-place. Validated on the cc
(93%) and inv (97% RLM, 100% direct) Fabric runs.

## 0.1.7 — Universal validator + self-report contract

(Removed in 0.1.9 along with the rest of `experimental.adaptive`.)

## 0.1.6 — `data_exploration` skill hardening

- Skill cookbook annexed with chained-bracket gotcha, STRING-EQUALITY
  gotcha, Step 7 zero-result sanity check, universal placeholders.

## 0.1.5 — `data_exploration` skill: parsing fixes

Bug fixes around heterogeneous JSONL ingestion and downstream chained
bracket access.

## 0.1.4 — `data_exploration` skill: first iteration

Initial DuckDB + ripgrep + Python-streaming skill for files larger than
the LM context window.

## 0.1.3 — Reasoning-model handling

`FabricLM` / `OpenAILM` auto-handle reasoning models (e.g. `gpt-5`,
`o1`, `o3`).

## 0.1.2 — Skill text mentions pre-installed deps

`data_exploration` skill text now explicitly tells the LM that `duckdb`
and `polars` are pre-installed in the Fabric Python runtime.

## 0.1.1 — Large-file / log analysis (opt-in)

Added the opt-in `data_exploration` skill family for analyzing files
larger than the LM context window.

## 0.1.0 — Initial release

Public API for fabric-rlm: `RLM`, `RLMResult`, `FabricLM`, skills,
trajectory + replay, validators.
