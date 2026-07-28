---
applies_when:
  keywords:
    - log
    - logs
    - csv
    - parquet
    - jsonl
    - json
    - dataset
    - rows
    - lines
    - file
    - filesystem
    - lakehouse
    - aggregate
    - count
    - top
    - slow
    - error
    - failure
    - root cause
    - oom
    - exception
    - trace
    - schema
    - explore
    - inspect
    - unknown
    - nested
  output_fields: []
excludes: []
depends_on: []
specificity: domain
---
# data_exploration

Summary: Discover the schema FIRST, then load once into DuckDB and query many times. Use this whenever a task points at a file path and you do not already know its exact column structure.

You never read the raw bytes back into the LM. You write Python in the subprocess that filters/aggregates and prints only small summaries.

## Gate: is this path a Delta table? (check BEFORE the discovery protocol)

If the path contains `/Tables/`, starts with `abfss://`, or the directory holds a
`_delta_log` folder, it is a **Delta table, not a file**. The discovery protocol
below and every `read_parquet` / `read_csv` / glob pattern in this skill are the
WRONG tools for it.

A Delta directory keeps parquet files that are no longer part of the table:
`UPDATE` / `DELETE` / `MERGE` write new files and tombstone the old ones in
`_delta_log`. Globbing reads deleted rows back in. Measured on a 10-row table
after one delete and one append (ground truth 6 rows, sum 1149.0):
`read_parquet('<table>/**/*.parquet')` returned **16 rows, sum 1699.0**. It did
not raise and it did not warn.

Use `delta_scan`, which honors the log:

```python
import duckdb
con = duckdb.connect()
con.sql("INSTALL delta; LOAD delta;")
print(con.sql(f"SELECT count(*) FROM delta_scan('{path}')").fetchone())
```

For anything beyond a simple scan (schema and partition discovery from metadata,
`abfss://` tokens, time travel, joining a table to a file in `Files/`), request
the `delta_lakehouse` skill, which covers Delta end to end and is read-only.

## Mandatory first-turn protocol (do this BEFORE writing any aggregation query)

If you have not already inspected this exact file in a previous turn, your first code action MUST run the four-step discovery below. It typically takes one turn and prevents 3+ wasted turns of column-name and JSON-function errors.

```python
# Step 1 — Raw sample (portable; works on Windows + Linux + Fabric)
from itertools import islice
path = "/path/to/your/file"
with open(path, "r", errors="replace") as f:
    sample = list(islice(f, 5))
print("RAW SAMPLE (first 5 lines):")
for ln in sample:
    print(ln[:1000])

# Step 2 — Load once into DuckDB (in-memory)
import duckdb, time
con = duckdb.connect()
t0 = time.time()
con.execute("CREATE TABLE t AS SELECT * FROM read_json_auto(?)", [path])
print(f"LOAD: {time.time()-t0:.2f}s")

# Step 3 — DESCRIBE the actual schema (do NOT assume column names)
schema = con.execute("DESCRIBE t").fetchall()
print("SCHEMA:")
for col_name, col_type, *_ in schema:
    print(f"  {col_name}: {col_type}")
print("ROWS:", con.execute("SELECT count(*) FROM t").fetchone()[0])

# Step 4 — If there is a discriminator column (Event, type, level, kind),
#          group by it to learn the sub-schemas.
print("SAMPLE ROW:")
print(con.execute("SELECT * FROM t LIMIT 1").fetchall())
```

After this prints, you know exactly what columns exist and which case applies (see "DuckDB JSONL: two cases you MUST handle" below). Only then do you write the real aggregation queries.

## Anti-patterns (these caused real production failures — do NOT repeat them)

- `json_extract_scalar(...)` — that's BigQuery / Postgres syntax. **DuckDB uses `json_extract_string(...)`** for scalar extraction, or the `->>` operator.
- `duckdb.quote_identifier(...)` — not a Python DuckDB API. Use SQL double-quotes for identifiers, or avoid building dynamic identifiers entirely (use `?` placeholders for VALUES, double-quoted literals for identifiers).
- Assuming `read_json_auto` will produce flat columns. **Run `DESCRIBE t` first.** Heterogeneous JSONL (where rows have different schemas — e.g. application event streams, framework logs, audit feeds) collapses to a single column called `json` of type `MAP(VARCHAR, JSON)`.
- Chained MAP brackets >1 level deep in Case B: `json['group']['subgroup']['leaf']`. The first `[]` returns a `JSON` value, NOT another MAP — the second/third `[]` silently produce SQL NULL, and `WHERE x IS NOT NULL` filters every row out. **Use the JSON arrow operators `->` and `->>` past the first level** (see Case B cookbook below).
- Trusting a query that returned 0 rows / `SUM = 0` without re-checking your field path. If you expected matches and got none, the path is wrong. Re-inspect a sample row with `print(con.execute("SELECT json FROM t WHERE json_extract_string(json, '$.<discriminator>')='<expected_kind>' LIMIT 1").fetchone())` and re-extract.
- `json['key']::VARCHAR` for string equality / display. Casting a `JSON` value to `VARCHAR` keeps the JSON quotes — you get the literal 7-char string `"error"` (with the quote marks), not `error`. So `WHERE json['kind']::VARCHAR = 'error'` matches **zero rows**, and your printed values look ugly. **For unquoted strings use `json_extract_string(json, '$.key')` or `(json ->> 'key')`** — both return raw VARCHAR. Reserve `::VARCHAR` cast for when you actually want the JSON-quoted form.
- `read_parquet('<path>/**/*.parquet')` against a `/Tables/` path. That is a Delta table; the glob resurrects tombstoned rows and returns an inflated answer with no error. See the Delta gate at the top of this skill.
- `data = open(path).read()` on a multi-MB file, then process in memory.
- Re-streaming the SAME file in turn 1, turn 2, turn 3 to answer different sub-questions. Load into DuckDB once on turn 1.
- Pasting raw lines into the prompt or pulling tens of thousands of matched lines into a Python list. Aggregate in code first.
- Using pandas `read_csv` for files larger than RAM. Prefer DuckDB or polars `scan_csv` (lazy).

## DuckDB JSONL: two cases you MUST handle

After `CREATE TABLE t AS SELECT * FROM read_json_auto(path)`, run `DESCRIBE t`. There are two possible outcomes:

### Case A — Flat columns (homogeneous JSONL)

`DESCRIBE t` shows multiple typed columns (`Event VARCHAR`, `Time BIGINT`, ...). Query directly:

```sql
SELECT Event, count(*) AS n FROM t GROUP BY Event ORDER BY n DESC LIMIT 10;
```

### Case B — Single `json MAP(VARCHAR, JSON)` column (heterogeneous JSONL)

`DESCRIBE t` shows ONE column: `json MAP(VARCHAR, JSON)`. This is the common shape for any heterogeneous JSON event stream — application logs, audit events, telemetry, framework event logs (Spark/Kafka/Airflow/etc.), webhook archives. Query via MAP indexing on the `json` column. Substitute the field names below for whatever your DESCRIBE step revealed:

```sql
-- Distribution of the discriminator field (often called "Event", "type", "kind", "level", "action")
-- IMPORTANT: use json_extract_string (or `->>`) for string equality and display — see note below.
SELECT (json ->> '<discriminator>') AS kind, count(*) AS n
FROM t
GROUP BY kind
ORDER BY n DESC
LIMIT 10;

-- One level deep — MAP indexing is fine, just cast the leaf
SELECT
  (json ->> '<discriminator>')                  AS kind,
  json['<group>']['<id_field>']::INT            AS id,
  (json['<group>'] ->> '<status_field>')        AS status
FROM t
WHERE (json ->> '<discriminator>') = '<some_kind>'
  AND json['<group>']['<status_field>'] IS NOT NULL;
```

**STRING-EQUALITY GOTCHA — `::VARCHAR` keeps JSON quotes.** Casting a `JSON` value with `::VARCHAR` gives you the JSON-encoded form including the surrounding `"`. So `'error'::VARCHAR(JSON value) = '"error"'` — the literal 7-character string. A filter like `WHERE json['kind']::VARCHAR = 'error'` therefore matches **zero rows**, and printed values look ugly. For string comparisons or display, ALWAYS extract through `->>` or `json_extract_string`:

```sql
-- WRONG — matches zero rows because LHS is `"error"` (with quote marks)
WHERE json['kind']::VARCHAR = 'error'

-- RIGHT — both return the unquoted VARCHAR `error`
WHERE (json ->> 'kind')                       = 'error'
WHERE json_extract_string(json, '$.kind')     = 'error'
```

Reserve `::VARCHAR` cast for when you actually want the raw JSON text (rare).

**CRITICAL — going more than ONE level deep:** `json['<group>']` returns a `JSON` value, NOT another MAP. Chaining a second `['…']` against a JSON value silently returns SQL NULL, and `WHERE x IS NOT NULL` / `SUM(x)` will then drop or zero every row — you'll get an empty result that looks like a real answer. You MUST switch to the JSON arrow operators (`->` returns JSON, `->>` returns VARCHAR) or `json_extract_string` past the first hop:

```sql
-- WRONG — silently returns NULL, every WHERE / SUM is empty:
json['<group>']['<subgroup>']['<leaf>']::UBIGINT

-- RIGHT — `->` returns JSON, `->>` returns VARCHAR; cast the leaf:
(json['<group>'] -> '<subgroup>' ->> '<leaf>')::UBIGINT

-- Equivalent — explicit JSON path string:
json_extract_string(json['<group>'], '$."<subgroup>"."<leaf>"')::UBIGINT
```

Generic worked example — top 3 longest events by some nested duration, and a sum of nested bytes:

```sql
-- Top 3 by a nested duration field
SELECT
  (json['<id_group>']  ->> '<id_field>')::UBIGINT       AS id,
  (json['<id_group>']  ->> '<owner_field>')             AS owner_id,
  (json['<metrics>']   ->> '<duration_field>')::BIGINT  AS duration_ms
FROM t
WHERE (json ->> '<discriminator>') = '<event_kind>'
ORDER BY duration_ms DESC NULLS LAST
LIMIT 3;

-- Sum of a deeply-nested numeric field (3 levels)
SELECT COALESCE(SUM(
    (json['<metrics>'] -> '<subgroup>' ->> '<bytes_field>')::UBIGINT
), 0) AS total_bytes
FROM t
WHERE (json ->> '<discriminator>') = '<event_kind>';
```

Sanity check: after running a query against deeply-nested fields, **always print `len(rows)` and any SUM result**. If you got 0 rows / `SUM = 0` when the discovery sample clearly showed the field is populated, your path expression is wrong — re-print one matching row's `json` value and fix the path before SUBMIT.

### Optional — `union_by_name=true` (use only when you understand the trade-off)

`read_json_auto(path, union_by_name=true)` forces DuckDB to unify all per-row schemas into one flat table — every distinct top-level key becomes its own column. Pros: lets you query `Event` directly. Cons: on highly heterogeneous logs the schema can explode to hundreds of mostly-null columns and slow inference dramatically.

```python
con.execute("CREATE TABLE t AS SELECT * FROM read_json_auto(?, union_by_name=true)", [path])
```

Decision rule:
- Moderate heterogeneity (a dozen event types, modest top-level keys) → `union_by_name=true` is convenient.
- High heterogeneity (heterogeneous framework or audit event logs with many event-specific nested payloads) → accept the MAP column and use Case B indexing.

## Tool ranking (after discovery is done)

1. **DuckDB** — best for any structured-or-semi-structured file you will query more than once. Pre-installed in the Fabric Python runtime; just `import duckdb`.
2. **polars** — also pre-installed in Fabric; great for lazy scans (`pl.scan_csv`, `pl.scan_parquet`) when you want a DataFrame API.
3. **Python streaming (`for line in open(...)`)** — universal fallback when DuckDB / polars are unavailable or the pattern is too irregular. The default `SecurityPolicy` blocks `subprocess` / `os.system`, so external shell tools like `rg` or `grep` are not available — stay inside Python.

> Do not waste a turn on `%pip install duckdb` or `polars` — both ship with the Fabric Python runtime. Just import.

## Pattern A — DuckDB load-once, query-many (PREFERRED)

When the same file answers multiple sub-questions, pay the parsing cost ONCE then query in milliseconds. Print load time and per-query time separately so the load-once-query-many advantage is visible:

```python
import duckdb, time
con = duckdb.connect()
t0 = time.time()
con.execute("CREATE TABLE log AS SELECT * FROM read_json_auto(?)", [path])
print(f"LOAD: {time.time()-t0:.2f}s")

# Now ALL subsequent questions are sub-second:
for label, sql in [
    ("Q1", "SELECT (json ->> 'kind') AS k, count(*) FROM log GROUP BY k ORDER BY 2 DESC LIMIT 10"),
    ("Q2", "SELECT count(*) FROM log WHERE (json ->> 'kind')='error' AND (json['error_payload'] ->> 'severity') = 'FATAL'"),
]:
    t = time.time(); rows = con.execute(sql).fetchall(); print(f"{label}: {time.time()-t:.3f}s  rows={len(rows)}")
```

For CSV/Parquet (always flat schemas):

```python
con.execute("CREATE TABLE t AS SELECT * FROM read_csv_auto(?)", [path])
con.execute("CREATE TABLE t AS SELECT * FROM read_parquet(?)", [path])
```

## Pattern B — Python streaming (always works)

```python
import re
oom = 0
slow = []  # (dur_ms, task_id)
RX_FIN = re.compile(r"Finished task (\d+).*in (\d+) ms")
with open(path, "r", errors="replace") as f:
    for ln in f:
        if "OutOfMemoryError" in ln:
            oom += 1
        m = RX_FIN.search(ln)
        if m:
            slow.append((int(m.group(2)), m.group(1)))
slow.sort(reverse=True)
print({"oom": oom, "top3": slow[:3]})
```

> External shell tools (`subprocess`, `os.system`, ripgrep CLI, …) are
> blocked by the default `SecurityPolicy`. Stay inside Python — the
> streaming pattern above is comparable in performance for the line
> counts we typically see and works on every platform.

## Result-size preflight

Before fetching unbounded results into stdout, bound the output. If your query has no `LIMIT`, no `count()`, and no `GROUP BY`, prepend a `count(*)` query first to see the cardinality. Anything over ~1000 rows should be aggregated, sampled, or summarised — never printed in full.

## Pre-flight checklist (before SUBMIT)

- [ ] Step 1: I ran the discovery protocol (RAW SAMPLE + LOAD + DESCRIBE + sample row) before writing aggregation queries.
- [ ] Step 2: I checked which JSONL case applies (flat columns vs `MAP(VARCHAR, JSON)`) and used the matching syntax.
- [ ] Step 3: I am using DuckDB load-once if multiple sub-questions hit the same file.
- [ ] Step 4: My final output to SUBMIT contains only aggregated values (counts, top-K, IDs) — never raw log lines.
- [ ] Step 5: I printed intermediate counts/shapes between exploration steps.
- [ ] Step 6 (anti-drift): I am answering the EXACT sub-questions the user asked, in the order they were asked, with the EXACT field names they specified. I am NOT substituting nearby or more interesting questions discovered during exploration. If a requested field is genuinely unanswerable from the data, I say so explicitly rather than silently swapping it for something else.
- [ ] Step 7 (zero-result sanity): For any query whose result is `0 rows`, `SUM = 0`, or `count = 0`, I confirmed the field path against an actual sample row before SUBMITting. Empty results from chained MAP brackets in Case B are almost always a path bug, not a data fact. **Especially** check that string equality uses `(json ->> 'k') = 'v'`, NOT `json['k']::VARCHAR = 'v'` (the cast keeps JSON quotes and matches nothing).

## Graceful degradation

DuckDB is OPTIONAL. Always wrap imports defensively:

```python
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
```

If DuckDB is missing, fall back to Pattern B (pure-Python streaming). The skill is about strategy, not specific binaries.

---

> Pattern 0 / discovery-triple convention inspired by `duckdb/duckdb-skills` (Apache 2.0). DuckDB JSON cookbook (`MAP(VARCHAR, JSON)` indexing, chained-bracket trap past one level, sample-row sanity check) derived from real failure trajectories on heterogeneous event-log workloads.
