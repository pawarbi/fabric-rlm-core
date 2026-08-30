---
applies_when:
  keywords:
    - delta
    - delta table
    - delta_scan
    - deltatable
    - delta log
    - _delta_log
    - onelake
    - abfss
    - /tables/
    - .lakehouse
    - lakehouse table
    - attached lakehouse
    - time travel
    - table version
  output_fields: []
excludes: []
depends_on: []
specificity: domain
---
# delta_lakehouse

Summary: READ-ONLY analysis of Delta tables in a Fabric Lakehouse using DuckDB with a delta-rs fallback. A Delta table is a transaction log plus data files, NOT a folder of parquet. Resolve the path, discover schema and partitions from metadata, then query through a Delta-aware reader. Works against an attached lakehouse mount and an `abfss://` OneLake path.

You never read raw table bytes back into the LM. You print small summaries only.

## When a LakehouseSource is already bound

A bound `LakehouseSource` is resolved in the parent Fabric process before the
isolated worker starts. Its catalog is the discovery boundary and already
contains source names, direct paths, kinds, and columns.

```python
all_sources = lakehouse.list_sources()
delta_sources = lakehouse.list_sources(kind="delta")
usage_candidates = lakehouse.find_sources("usage", kind="delta")

result = lakehouse.query(
    """
    SELECT c.region,
           SUM(s.mrr) AS active_mrr,
           COUNT(DISTINCT c.company_id) AS company_count
    FROM companies AS c
    JOIN subscriptions AS s USING (company_id)
    WHERE s.status = 'active'
    GROUP BY c.region
    ORDER BY active_mrr DESC
    LIMIT 1
    """,
    sources={
        "companies": "dbo.companies",
        "subscriptions": "dbo.subscriptions",
    },
)
print(result)
```

Use `list_sources()` or `find_sources()` first, select only the entries relevant
to the question, and pass their catalog names to `lakehouse.query(...)`. Use the
catalog's `columns` metadata to plan joins before querying. The trusted parent
opens those sources and returns bounded rows; storage credentials never enter
worker code.

Do not call `open_delta` for a bound `LakehouseSource`. Do not call
`_storage_token`. Do not call `notebookutils` or `DefaultAzureCredential`
either. Those direct-path helpers are for raw path inputs, not resolved
Lakehouse inputs.
Fabric workload credentials are commonly unavailable in the isolated worker,
so retrying them wastes turns and cannot repair the boundary. Never widen
beyond the supplied catalog, and never submit placeholder or hypothetical
metrics after a query failure.

## READ THIS FIRST - the failure that returns a wrong answer silently

A Delta table directory contains parquet files that are no longer part of the
table. Every `UPDATE`, `DELETE`, and `MERGE` writes new files and *tombstones* the
old ones in `_delta_log`. The stale files stay on disk until someone vacuums.

Globbing them reads deleted rows back in. Measured on a 10-row table after one
delete and one append (ground truth 6 rows, sum 1149.0):

```
read_parquet('<table>/**/*.parquet')   ->  16 rows, sum 1699.0   WRONG
delta_scan('<table>')                  ->   6 rows, sum 1149.0   correct
DeltaTable(...).to_pyarrow_dataset()   ->   6 rows, sum 1149.0   correct
```

The glob does not raise, does not warn, and returns numbers that look plausible.
Nothing downstream can catch it.

**Rule: if the path contains `/Tables/`, or the directory contains `_delta_log`,
you MUST NOT use `read_parquet` / `read_csv` / `glob` on it. There is no
exception, no clever variant, and no fallback that makes it acceptable.**

That includes the version that looks correct. `dt.file_uris()` returns the exact
file list from the transaction log, so feeding it to `read_parquet` seems like
the rigorous way to do it. It is not. When the table has **deletion vectors**
(on by default for many Fabric lakehouse tables), those files still physically
contain rows that have been deleted; the log records the deletions separately and
the reader is responsible for applying them. `read_parquet` does not. `file_uris()`
is a metadata API for inspecting layout, never a read path.

If you cannot read a Delta table through a Delta reader, the correct outcome is to
say so, not to approximate it with parquet.

## Do / Don't

| Do | Don't | Why |
| --- | --- | --- |
| `delta_scan(path)`, or `DeltaTable(...)` as fallback | `read_parquet('<Tables path>/**/*.parquet')` | The glob reads tombstoned rows and inflates the answer with no error |
| Call `open_delta` once, reuse the returned relation | Re-open the table for each sub-question | Re-resolving costs a turn and re-downloads the extension |
| `attach_lakehouse(root)` when a question spans tables | Open each table and join the results in pandas | Attached views join in SQL with nothing copied into memory |
| Plan with catalog columns, then open only selected sources | Open every source just to inspect schemas | Catalog metadata is already available and remote table setup can dominate runtime |
| Print the schema before writing a query | Assume a column is called `revenue` | A guessed name errors, or silently matches the wrong column |
| Read `num_records` from metadata first | `dt.to_pandas()` before you know the size | A fact table will exhaust memory in one call |
| `USING SAMPLE n ROWS` to look at data | `LIMIT n` as a sample | `LIMIT` reads the first file only, so you see one partition |
| `date - INTERVAL 11 MONTH` | `date_add('month', -11, date)` | DuckDB's `date_add` accepts `(date, interval)`, not the Spark/SQL Server three-argument form |
| `count(DISTINCT col)` when it matters | Trust `approx_unique` as exact | It is a HyperLogLog estimate and can exceed the row count |
| Print `fetchall()` rows yourself | `relation.show()` | Box-drawing characters crash on cp1252 stdout |
| `con.register("name", dataset)` | Put a bare Python variable name in the SQL | Replacement scan reads the *calling* frame, so it breaks inside a helper |
| Bind the token as a parameter: `ACCESS_TOKEN ?` | Format the token into the SQL string | Tokens end up in logs, tracebacks, and the trajectory |
| Omit `ENDPOINT`, or give it an `https://` scheme | `ENDPOINT 'onelake.dfs.fabric.microsoft.com'` | A bare host fails with `relative URL without a base` |
| `storage_options=None` for a mounted path | Hand a mount a token | A mount is plain POSIX IO and needs no credential |
| Put partition columns in `WHERE` | Filter in pandas after loading everything | Pruning skips whole files at the scan |
| Write results under `Files/` | Write anything under `Tables/` | An untracked parquet inside a table dir is invisible to correct readers |
| Shape data with `CREATE TABLE ... AS` in DuckDB | `write_deltalake`, `merge`, `vacuum`, `optimize` | This skill is read-only; those are engineering ops |
| Let a failed `INSTALL delta` fall through to delta-rs | Treat a failed `INSTALL` as a reason to glob | delta-rs honors `_delta_log`; the glob does not |
| Use `file_uris()` to inspect layout only | `read_parquet(dt.file_uris())` | With deletion vectors those files still hold deleted rows |
| Report that a table is unreadable in this runtime | Approximate it with a parquet read | A wrong number is worse than no number |

## Step 1 - resolve the path (attached lakehouse OR abfss)

Two path shapes reach the same table. Handle both.

**Attached (mounted) lakehouse** - the default in a Fabric notebook with a
lakehouse attached. A plain POSIX path, no credentials needed:

```
/lakehouse/default/Tables/<table>              # non-schema-enabled lakehouse
/lakehouse/default/Tables/<schema>/<table>     # schema-enabled lakehouse
```

**OneLake `abfss://`** - any workspace or lakehouse, attached or not. Needs a
storage token:

```
abfss://<workspace>@onelake.dfs.fabric.microsoft.com/<lakehouse>.Lakehouse/Tables/<schema>/<table>
```

**Friendly names do not work on every tenant.** Many tenants have
`FriendlyNameSupportDisabled`, where that URL is rejected with:

```
Request Failed with WorkspaceId and ArtifactId should be either valid Guids or valid Names
ErrorCode:FriendlyNameSupportDisabled
```

The message reads like a validation quirk but it is a hard tenant setting. Use
GUIDs, and note the `.Lakehouse` suffix belongs on a *name* and must NOT be
appended to a GUID:

```
abfss://<workspaceId>@onelake.dfs.fabric.microsoft.com/<lakehouseId>/Tables/<schema>/<table>
```

`onelake_root()` below resolves both ids from `notebookutils.runtime.context`
and builds this for you. Prefer it over hand-assembling a path. When abfss fails
and a mount is available, the mount needs no ids at all and is the faster route.

Copy this resolver verbatim. It picks the right engine for either shape:

```python
import duckdb

def delta_opts(path):
    """storage_options for delta-rs: None for a mount, a token dict for abfss."""
    if "://" not in path:
        return None
    return {"bearer_token": _storage_token(), "use_fabric_endpoint": "true"}


def _prepare(con, path):
    """Load the delta extension, plus azure and a OneLake token for abfss paths."""
    is_remote = "://" in path
    con.sql("INSTALL delta; LOAD delta;")
    if is_remote:
        con.sql("INSTALL azure; LOAD azure;")
        # ACCOUNT_NAME is 'onelake'. Do NOT set ENDPOINT to a bare hostname
        # (see anti-patterns). Omitting ENDPOINT works: the abfss:// URL
        # already carries the host.
        # DuckDB cannot replace a secret while an active view is using it.
        # Reuse the session secret when several tables share one connection.
        has_secret = con.sql(
            "SELECT count(*) FROM duckdb_secrets() "
            "WHERE name = 'onelake_tok'"
        ).fetchone()[0]
        if not has_secret:
            token = _storage_token().replace("'", "''")
            con.execute(
                "CREATE SECRET onelake_tok "
                "(TYPE azure, PROVIDER access_token, "
                f"ACCESS_TOKEN '{token}', ACCOUNT_NAME 'onelake')"
            )
    return con


def open_delta(path, con=None):
    """Return (con, relation_sql) for a read-only DuckDB view of ONE Delta table.

    Handles both an attached-lakehouse mount and an abfss:// OneLake URL.
    To query many tables at once, use attach_lakehouse instead.
    """
    con = con or duckdb.connect()
    escaped_path = path.replace("'", "''")

    try:
        _prepare(con, path)
        con.sql(f"SELECT 1 FROM delta_scan('{escaped_path}') LIMIT 1")
        return con, f"delta_scan('{escaped_path}')"
    except Exception as e:
        print(f"[delta_scan unavailable: {type(e).__name__}: {str(e)[:120]}] falling back to delta-rs")

    # Fallback: delta-rs reader registered into the same DuckDB session.
    from deltalake import DeltaTable
    from deltalake.exceptions import DeltaProtocolError
    dt = DeltaTable(path, storage_options=delta_opts(path))
    try:
        # register() binds the name on the connection itself. Do NOT rely on
        # DuckDB's replacement scan picking a local variable out of the calling
        # frame; that breaks as soon as the query runs inside a helper function.
        con.register("delta_tbl", dt.to_pyarrow_dataset())
    except DeltaProtocolError as e:
        # Deletion vectors: delta-rs cannot read this table at all. There is no
        # correct parquet-level workaround, so stop here rather than improvise.
        raise RuntimeError(
            f"This table needs DuckDB's delta extension and it is unavailable: {e} "
            "Do NOT fall back to read_parquet: with deletion vectors the data "
            "files still contain logically deleted rows. Report that the table "
            "cannot be read in this runtime instead of returning a number."
        ) from e
    return con, "delta_tbl"


def _storage_token():
    """Storage bearer token. notebookutils inside Fabric, azure-identity outside."""
    try:
        import notebookutils
        return notebookutils.credentials.getToken("storage")
    except Exception:
        from azure.identity import DefaultAzureCredential
        return DefaultAzureCredential().get_token("https://storage.azure.com/.default").token


def _ls(path):
    """(name, full_path) per child. notebookutils for abfss, os for a filesystem."""
    if "://" in path:
        import notebookutils
        return [(f.name.rstrip("/"), f.path.rstrip("/")) for f in notebookutils.fs.ls(path)]
    import os
    return [(n, f"{path}/{n}") for n in sorted(os.listdir(path))]


def _is_table(path):
    """A directory is a Delta table when it holds a _delta_log."""
    try:
        if "://" in path:
            return any(n == "_delta_log" for n, _ in _ls(path))
        import os
        return os.path.isdir(os.path.join(path, "_delta_log"))
    except Exception:
        return False


def find_delta_tables(root):
    """{qualified_name: path} for every Delta table under a lakehouse or schema.

    Handles both Fabric layouts, side by side: Tables/<table> becomes `name`,
    Tables/<schema>/<table> becomes `schema.name`. Descends exactly one level,
    so partition directories are never walked.
    """
    found = {}
    for name, path in _ls(root):
        if name.startswith("_"):
            continue
        if _is_table(path):
            found[name] = path
            continue
        try:
            for sub, subpath in _ls(path):
                if not sub.startswith("_") and _is_table(subpath):
                    found[f"{name}.{sub}"] = subpath
        except Exception:
            pass
    return found


def onelake_root(workspace=None, lakehouse=None, schema=None):
    """Build a OneLake Tables/ root, preferring GUIDs over friendly names.

    Many tenants set FriendlyNameSupportDisabled, where names in a OneLake URL
    are rejected outright. GUIDs always work, so default to the ones the
    notebook already knows. Note the `.Lakehouse` suffix belongs on a NAME and
    must NOT be appended to a GUID.
    """
    ws, lh = workspace, lakehouse
    if not (ws and lh):
        # Only reach for the notebook context when something is actually missing,
        # so explicit ids work outside Fabric too.
        import notebookutils
        ctx = notebookutils.runtime.context
        ws = ws or ctx.get("currentWorkspaceId") or ctx.get("workspaceId")
        lh = lh or ctx.get("defaultLakehouseId") or ctx.get("lakehouseId")
        if not ws or not lh:
            raise RuntimeError(
                "Could not resolve workspace/lakehouse ids from "
                f"notebookutils.runtime.context. Available keys: {sorted(ctx)}. "
                "Pass them explicitly."
            )
    item = lh if _looks_like_guid(lh) else f"{lh}.Lakehouse"
    root = f"abfss://{ws}@onelake.dfs.fabric.microsoft.com/{item}/Tables"
    return f"{root}/{schema}" if schema else root


def _looks_like_guid(s):
    parts = str(s).split("-")
    return len(parts) == 5 and len(parts[0]) == 8 and all(
        c in "0123456789abcdefABCDEF-" for c in str(s))


def attach_lakehouse(root, con=None):
    """Register every Delta table under `root` as a read-only DuckDB view.

    `root` is a lakehouse Tables/ directory or a single schema inside one,
    as a mount or an abfss:// URL. Returns (con, attached, skipped).
    """
    con = con or duckdb.connect()
    prepare_failure = None
    try:
        _prepare(con, root)
    except Exception as prepare_error:
        # open_delta has the same boundary: DuckDB setup may fail while
        # delta-rs can still read the table correctly.
        prepare_failure = (
            type(prepare_error).__name__,
            str(prepare_error)[:70],
        )

    try:
        tables = find_delta_tables(root)
    except Exception as e:
        msg = str(e)
        hint = ""
        if "FriendlyNameSupportDisabled" in msg or "valid Guids or valid Names" in msg:
            hint = (
                " This tenant does not accept workspace or lakehouse NAMES in a OneLake "
                "URL. Use GUIDs: abfss://<workspaceId>@onelake.dfs.fabric.microsoft.com/"
                "<lakehouseId>/Tables, with no .Lakehouse suffix after a GUID. "
                "onelake_root() builds this for you."
            )
        raise RuntimeError(f"Cannot list {root}: {type(e).__name__}: {msg[:250]}{hint}") from e

    def quote_ident(name):
        return '"' + name.replace('"', '""') + '"'

    attached, skipped = [], []
    for index, (qname, path) in enumerate(sorted(tables.items())):
        esc = path.replace("'", "''")
        if "." in qname:
            sch, tbl = qname.split(".", 1)
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {quote_ident(sch)}")
            target = f"{quote_ident(sch)}.{quote_ident(tbl)}"
        else:
            target = quote_ident(qname)
        if prepare_failure is None:
            try:
                con.execute(
                    f"CREATE OR REPLACE VIEW {target} "
                    f"AS SELECT * FROM delta_scan('{esc}')"
                )
                attached.append(qname)
                continue
            except Exception as delta_scan_error:
                delta_scan_failure = (
                    type(delta_scan_error).__name__,
                    str(delta_scan_error)[:70],
                )
        else:
            delta_scan_failure = prepare_failure

        try:
            from deltalake import DeltaTable

            dt = DeltaTable(path, storage_options=delta_opts(path))
            source = f"__fabric_rlm_delta_{index}"
            con.register(source, dt.to_pyarrow_dataset())
            con.execute(
                f"CREATE OR REPLACE VIEW {target} "
                f"AS SELECT * FROM {quote_ident(source)}"
            )
            attached.append(qname)
        except Exception as delta_rs_error:
            # One unreadable table must not abort the whole lakehouse.
            skipped.append((
                qname,
                f"delta_scan {delta_scan_failure[0]}: "
                f"{delta_scan_failure[1]}; delta-rs "
                f"{type(delta_rs_error).__name__}: {str(delta_rs_error)[:70]}",
            ))
    return con, attached, skipped
```

## Step 1b - attach a whole lakehouse, or one schema

`open_delta` is right when the question names one table. When it spans tables
("revenue per customer", "returns by product"), do not open them one at a time.
Point at the lakehouse and let SQL do the joins:

```python
con, attached, skipped = attach_lakehouse(
    "abfss://<workspace>@onelake.dfs.fabric.microsoft.com/<lakehouse>.Lakehouse/Tables"
)
print(f"attached {len(attached)}: {attached}")
for name, why in skipped:
    print(f"  skipped {name}: {why}")

rows = con.sql("""
    SELECT c.name, count(*) AS orders, round(sum(o.amt), 2) AS total
    FROM orders o
    JOIN customers c USING (cust_id)
    GROUP BY 1 ORDER BY total DESC LIMIT 10
""").fetchall()
for r in rows:
    print(r)
```

The same call takes a mount (`/lakehouse/default/Tables`) or one schema
(`.../Tables/dbo`) when you only want part of it.

**Naming.** `Tables/orders` attaches as `orders`. `Tables/dbo/products` attaches
as `"dbo"."products"` in a DuckDB schema of the same name, so both Fabric layouts
coexist and cross-schema joins are ordinary SQL. Names are quoted, so spaces and
dashes survive.

**What it costs.** DuckDB binds a view when you create it, so attaching reads each
table's `_delta_log` once. Locally that is a few milliseconds per table; over
`abfss` it is one round trip per table, so a large lakehouse takes a moment. It
does **not** read any data. Nothing is scanned until you query.

**Eager binding is a feature here.** A table that cannot be read fails at attach
time and lands in `skipped` with the reason, instead of ambushing you mid-query.
One unreadable table never aborts the rest. Deletion-vector tables show up here
when the `delta` extension is missing.

**Still read-only.** These are views over `delta_scan`. Nothing is copied, nothing
is written. Run `SELECT table_schema, table_name FROM information_schema.tables`
to see what you have, then `DESCRIBE <table>` per table before querying, exactly
as in Step 2.

Note on tokens outside Fabric: the default `SecurityPolicy` strips `AZURE_CLIENT_*`
and `AZURE_TENANT_*` from the worker environment, so `DefaultAzureCredential`
cannot pick up a service principal from env vars there. Inside a Fabric notebook
`notebookutils` is used and this does not apply.

Then every query below uses the returned relation:

```python
con, T = open_delta(path)
print(con.sql(f"SELECT count(*) FROM {T}").fetchone())
```

## Step 2 - mandatory first-turn discovery (schema BEFORE any aggregation)

Run this once, before you write a single aggregation. It is cheap, a few hundred
printed characters, and it prevents the guessed-column-name and wrong-grain
failures that account for most analysis errors on real tables.

This is the same discipline as inspecting a workbook's sheets and header rows
before extracting from it: **you do not know what the columns are called until you
have printed them.** A query written against a guessed column name either errors
and costs a turn, or silently matches something you did not mean.

```python
from deltalake import DeltaTable
import pyarrow as pa

con, T = open_delta(path)
dt = DeltaTable(path, storage_options=delta_opts(path))

# 1) Identity and layout, straight from the log. No data file is opened.
adds = pa.table(dt.get_add_actions()).to_pydict()
print("=== table ===")
print(f"  version {dt.version()}   files {len(adds.get('path', []))}   "
      f"bytes {sum(adds.get('size_bytes', [])):,}   rows<= {sum(adds.get('num_records', [])):,}")
print(f"  partitions      : {dt.partitions() or 'none'}")
print(f"  reader_features : {dt.protocol().reader_features or []}")

# 2) Schema. Print it; never assume a name or a type.
print("=== schema ===")
for name, typ, *_ in con.sql(f"DESCRIBE SELECT * FROM {T}").fetchall():
    print(f"  {name}: {typ}")

# 3) Actual rows. USING SAMPLE, not LIMIT (see the sampling trap below).
print("=== sample ===")
rel = con.sql(f"SELECT * FROM {T} USING SAMPLE 5 ROWS")
print("  " + " | ".join(d[0] for d in rel.description))
for row in rel.fetchall():
    print("  " + " | ".join(str(v)[:24] for v in row))

# 4) Per-column profile: nulls, spread, cardinality, in one statement.
print("=== profile ===")
rel = con.sql(f"SUMMARIZE SELECT * FROM {T}")
cols = [d[0] for d in rel.description]
for row in rel.fetchall():
    d = dict(zip(cols, row))
    print(f"  {d['column_name']:18} {d['column_type']:9} nulls={d['null_percentage']}%"
          f"  ~uniq={d['approx_unique']}  min={str(d['min'])[:14]}  max={str(d['max'])[:14]}")
```

### What to read off it, before writing any query

- **Column names and types.** Use them verbatim. If the question says "revenue"
  and the schema says `amount`, resolve that now, not inside an aggregate.
- **Partitions.** These are your free filters. A `WHERE` on a partition column
  skips whole files; a `WHERE` on anything else does not.
- **`rows<=`.** The size preflight. Decide here whether the table can be pulled
  into pandas at all, or whether everything has to stay in SQL.
- **`nulls=`.** A column that is 66% null changes what an average means and
  whether a join on it is safe.
- **`~uniq=`.** Tells you the grain. If `order_id` is unique and equal to the row
  count, one row is one order; if not, the table is already aggregated or
  duplicated, and a naive `sum` double counts.
- **`reader_features`.** If it lists `deletionVectors`, only the `delta` extension
  can read this table. See Graceful degradation.

### Three traps in this step specifically

**`LIMIT` is not a sample.** On a partitioned table it reads the first file and
stops, so every row you see comes from one partition. Measured on a table
partitioned by `region`: `SELECT region ... LIMIT 5` returned `south` five times,
while `USING SAMPLE 5 ROWS` returned a mix of `east`, `west`, and `north`. If you
eyeball a `LIMIT` sample you will form a wrong idea of the data's spread. Use
`USING SAMPLE` to look around; use `LIMIT` only when you genuinely want "any few
rows" and do not care which.

**`approx_unique` is approximate.** It is a HyperLogLog estimate and it can exceed
the row count, which makes the error obvious once you look: on a 380-row table it
reported 405 distinct `order_id` where the exact count is 380. Read it as an
order of magnitude for judging grain. When the exact number carries the answer,
use `count(DISTINCT col)`.

**`num_records` is a physical row count.** Deletion vectors sit beside the data
files and are applied by the reader, so on a table that has them this sum is an
**upper bound**, not the answer. Use it for the size preflight; take true counts
from `SELECT count(*) FROM {T}`. Metadata keeps working on deletion-vector tables
even when the delta-rs reader refuses them.

## Step 3 - query patterns (analysis, not engineering)

**Partition-pruned aggregate.** Put partition columns in `WHERE` so whole files
are skipped:

```python
rows = con.sql(f"SELECT region, count(*) n, sum(amt) total FROM {T} WHERE region='west' GROUP BY 1").fetchall()
for r in rows:
    print(r)
```

**Transform in memory, never write back.** Materialize a shaped copy inside the
DuckDB session. The source table is untouched:

```python
con.sql(f"CREATE TABLE staged AS SELECT region, amt, amt*1.08 AS amt_gross FROM {T}")
```

**Join a Delta table to a file in `Files/`.** Same session, one query:

```python
rows = con.sql("""
    SELECT l.label, round(sum(s.amt_gross), 2) AS gross
    FROM staged s
    JOIN read_csv('/lakehouse/default/Files/lookup.csv') l USING (region)
    GROUP BY 1 ORDER BY 1
""").fetchall()
for r in rows:
    print(r)
```

**Time travel, for point-in-time comparison.** This is analysis, not rollback:

```python
con.register("v3", DeltaTable(path, version=3, storage_options=delta_opts(path)).to_pyarrow_dataset())
print(con.sql(f"SELECT count(*) AS then_rows FROM v3").fetchone())
print(con.sql(f"SELECT count(*) AS now_rows FROM {T}").fetchone())
for h in dt.history(5):
    print(h.get("version"), h.get("timestamp"), h.get("operation"))
```

**Writing results out.** Outputs go to `Files/`, never into a `Tables/` path. A
stray parquet written inside a table directory is invisible to correct readers
and poisons anyone who globs it:

```python
con.sql(f"COPY (SELECT * FROM staged) TO '/lakehouse/default/Files/out/summary.parquet'")
```

## Read-only boundary

This skill analyzes tables. It never modifies them. Do NOT call, for any reason:

`dt.delete(...)`, `dt.update(...)`, `dt.merge(...)`, `dt.vacuum(...)`,
`dt.optimize.compact()`, `dt.optimize.z_order(...)`, `dt.restore(...)`,
`dt.repair(...)`, `dt.create_checkpoint()`, `dt.cleanup_metadata()`,
`dt.compact_logs()`, `write_deltalake(...)` in any mode, or Spark
`.write.format("delta")`.

`vacuum` is the dangerous one: it physically deletes tombstoned files, which
destroys time travel and makes any earlier delete permanent. There is no undo.

If a task seems to require one of these, it is a data-engineering task and out of
scope. Compute the answer in DuckDB and report it instead.

## Anti-patterns

- `read_parquet('<Tables path>/**/*.parquet')`. See the top of this skill. This is
  the single most likely thing to go wrong.
- Putting workspace and lakehouse **names** in a OneLake URL without checking.
  On a tenant with `FriendlyNameSupportDisabled` every call fails with
  "WorkspaceId and ArtifactId should be either valid Guids or valid Names",
  which reads like the names are malformed rather than unsupported. Use
  `onelake_root()`, or GUIDs with no `.Lakehouse` suffix.
- Appending `.Lakehouse` to a lakehouse **GUID**. The suffix goes on a name only.
- Setting `ENDPOINT` to a bare hostname in the azure secret. `ENDPOINT
  'onelake.dfs.fabric.microsoft.com'` fails with `Unable parse source url ...
  relative URL without a base`, which reads like a bug rather than a config
  error. Either omit `ENDPOINT` (recommended, the abfss URL carries the host) or
  give it a scheme: `'https://onelake.dfs.fabric.microsoft.com'`.
- `read_parquet(dt.file_uris())`. The most convincing wrong answer available: the
  file list is authoritative, so this looks rigorous. It ignores deletion vectors,
  which are exactly the case where the file list alone is not the table.
- `DeltaTable.files()`. Removed in deltalake 1.x. Use `file_uris()`, and only for
  inspecting layout.
- `dt.get_add_actions().to_pydict()`. In deltalake 1.x this returns an
  `arro3.core.Table`, not a pyarrow one, and raises `AttributeError`. Wrap it:
  `pa.table(dt.get_add_actions())`.
- Naming a Python variable directly in SQL and expecting DuckDB to find it.
  DuckDB's replacement scan inspects the *calling* frame, so
  `ds = dt.to_pyarrow_dataset()` then `con.sql("SELECT * FROM ds")` works at
  notebook top level and then breaks the moment the same query moves inside a
  function. Use `con.register("name", ds)`, which binds on the connection.
- `dt.to_pandas()` before checking `num_records`. Do the metadata count in Step 2
  first; a fact table will exhaust memory.
- `relation.show()` for previewing rows. DuckDB draws the table with Unicode box
  characters, which raises `UnicodeEncodeError: 'charmap' codec can't encode` on a
  cp1252 stdout (the Windows default). Print `fetchall()` rows yourself, which is
  also how you keep the output small.
- Writing an aggregation before running Step 2. Guessed column names are the most
  common way to burn a turn on this task.
- Re-opening the table once per sub-question. Call `open_delta` once, then run
  many queries against the returned relation.
- Passing `storage_options` to a mounted path. Use `None`; a mount needs no token.

## Graceful degradation

`INSTALL delta` downloads the extension binary. If the runtime has no egress it
will fail, and `open_delta` falls back to delta-rs, which is a pure Python wheel
and needs no download. Both respect `_delta_log`, so on an ordinary table the
answer is identical either way.

**The fallback is not universal.** On a table with the `deletionVectors` reader
feature, delta-rs raises `DeltaProtocolError` from both `to_pyarrow_dataset()`
and `to_pandas()`; only DuckDB's `delta` extension can read it. Metadata still
works there, so `get_add_actions()`, `history()`, `schema()`, and `file_uris()`
keep answering even when no reader will.

That combination is the moment of maximum temptation: the table is right there,
you can list its files, and `read_parquet` would return *something*. Do not. The
resolver raises a `RuntimeError` in this case on purpose. Report that the table
needs the delta extension in this runtime.

## Where these rules come from

Measured against duckdb 1.5.0 / deltalake 1.5.0 on a table built as 10 rows,
delete ids 6-10, append one row. `tests/test_delta_lakehouse_skill.py` pins each
of these, so if a future version changes the behavior the test fails and the
corresponding rule above should be re-checked rather than left standing.

- Naive glob returned **16 rows / sum 1699.0**. `delta_scan` and the delta-rs
  reader both returned **6 rows / sum 1149.0**. Ground truth is 6 / 1149.0.
- Row count from `num_records` matched a full scan exactly, with no data file
  opened.
- On a 400-row table partitioned by `region`, `SELECT region ... LIMIT 5` returned
  `south` five times while `USING SAMPLE 5 ROWS` returned `east`, `west`, and
  `north`. `LIMIT` is a first-file read, not a sample.
- `SUMMARIZE` reported `approx_unique = 405` for a column whose exact
  `count(DISTINCT ...)` is 380, on a 380-row table. The estimate exceeded the row
  count, which is the clearest possible sign not to quote it as a fact.
- `relation.show()` raised `UnicodeEncodeError` on cp1252 stdout. Printing
  `fetchall()` rows worked.
- `CREATE TABLE ... AS SELECT` off a `delta_scan`, joined to a CSV, left the
  source table at the same version. In-session transformation does not write.
- `ENDPOINT 'onelake.dfs.fabric.microsoft.com'` failed at URL construction with
  `Unable parse source url ... relative URL without a base`. Omitting `ENDPOINT`,
  or using `'https://onelake.dfs.fabric.microsoft.com'`, parsed and reached the
  auth layer.
- `con.register(...)` answered correctly from inside a helper function; a bare
  variable name relies on frame inspection and does not.
- The delta-rs fallback returned numbers identical to `delta_scan` on an ordinary
  table, so a runtime with no egress loses speed, never correctness.
- On a table with `delta.enableDeletionVectors`, delta-rs raised
  `DeltaProtocolError` from **both** `to_pyarrow_dataset()` and `to_pandas()`
  ("reader features: {'deletionVectors'} ... not yet supported"), while
  `delta_scan` read it correctly. `get_add_actions()`, `history()`, `schema()`,
  and `file_uris()` all still worked on that same table. Metadata surviving while
  every reader fails is precisely what makes a parquet workaround tempting.
- That `num_records` is a physical row count, so deletion vectors make it an upper
  bound, follows from the Delta protocol rather than from a local measurement:
  delta-rs rewrote files instead of emitting a deletion vector here, so no local
  table exercised the over-count. Treat the caveat as reasoning, not evidence.
- DuckDB's `delta` extension exposes no write path at all, which is why
  `delta_scan` is the recommended default: the fast path cannot mutate anything.

The verification notebook completed against a live OneLake `abfss://` path in
Microsoft Fabric using a schema-enabled Lakehouse. It exercised GUID path
construction, token retrieval, table and schema discovery, Delta metadata,
single-table SQL, whole-schema attachment through the delta-rs fallback,
bounded profiling, and the read-only version check.

That live run did not exercise the attached `/lakehouse/default` mount or a
table carrying deletion vectors. Those paths remain covered by local Delta
tables and protocol-focused tests rather than the live Fabric run.

## Pre-flight checklist (before SUBMIT)

- [ ] I never called `read_parquet` / glob on a `/Tables/` path, and never fed
      `file_uris()` to a parquet reader.
- [ ] I ran Step 2 discovery and used the printed column names verbatim.
- [ ] Any cardinality I report came from `count(DISTINCT ...)`, not `approx_unique`.
- [ ] Row counts came from `num_records`, not from loading the table.
- [ ] I called no mutating API from the read-only boundary list.
- [ ] Any output file I wrote landed under `Files/`, not `Tables/`.
