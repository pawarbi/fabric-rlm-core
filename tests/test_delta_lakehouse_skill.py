"""Tests for the ``delta_lakehouse`` skill and the Delta gate in ``data_exploration``.

Three layers:

1. Metadata and routing, which need no optional dependencies.
2. The gate text inside ``data_exploration`` that keeps a Delta path away from
   ``read_parquet`` even when ``delta_lakehouse`` loses the router's cap.
3. Executable checks that the skill's central claims are *true* against the
   installed duckdb / deltalake, including running the ``open_delta`` resolver
   extracted straight out of the Markdown.

Anything that would touch the network is marked ``network`` and deselected by
default; everything else runs offline.
"""

from __future__ import annotations

import re

import pytest

from fabric_rlm.skill_loader import SkillLoader
from fabric_rlm.skill_router import SkillRouter

SKILL = "delta_lakehouse"


# --------------------------------------------------------------------------
# Layer 1: metadata + routing
# --------------------------------------------------------------------------


def test_skill_is_packaged_and_loads() -> None:
    loader = SkillLoader()
    assert SKILL in loader.list_skills()

    skill = loader.load(SKILL)
    assert skill.title == SKILL
    assert skill.specificity == "domain"
    assert skill.summary.startswith("READ-ONLY")
    # No dependency edge: the gate in data_exploration is deliberately
    # standalone so neither skill force-loads the other.
    assert skill.dependencies == ()
    assert skill.excludes == ()


@pytest.mark.parametrize(
    "keyword",
    ["delta", "onelake", "abfss", "/tables/", ".lakehouse", "time travel"],
)
def test_routing_keywords_declared(keyword: str) -> None:
    skill = SkillLoader().load(SKILL)
    assert keyword in skill.applies_when_keywords


@pytest.mark.parametrize(
    "question",
    [
        "What are the top 5 products by revenue in the sales delta table?",
        "Explore /lakehouse/default/Tables/dbo/orders and give me the monthly trend",
        "Profile abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/fact_sales",
        "Compare the table version from last week using time travel",
    ],
)
def test_delta_questions_activate_the_skill(question: str) -> None:
    router = SkillRouter.from_loader(SkillLoader())
    decision = router.route(question)
    assert SKILL in decision.active, decision.scores


def test_mounted_path_alone_is_enough_to_activate() -> None:
    """The word 'delta' never appears; only the mounted /Tables/ path shape does.

    This is the case that regressed before the `/tables/` keyword existed: the
    question looks like an ordinary file-exploration ask.
    """
    router = SkillRouter.from_loader(SkillLoader())
    decision = router.route("summarize /lakehouse/default/Tables/dbo/orders")
    assert SKILL in decision.active
    assert decision.scores[SKILL] > 0


@pytest.mark.parametrize(
    "question",
    [
        "Summarize the error rates in app_logs.jsonl",
        "Read the csv from my lakehouse Files folder and count the rows",
        "Extract the totals from the Q3 workbook.xlsx",
        "Parse the invoice pdf and pull out the line items",
    ],
)
def test_non_delta_questions_do_not_activate_the_skill(question: str) -> None:
    """A separate skill only pays for itself if it stays out of unrelated work."""
    router = SkillRouter.from_loader(SkillLoader())
    decision = router.route(question)
    assert SKILL not in decision.active
    assert SKILL not in decision.scores


def test_delta_and_data_exploration_coexist_under_the_cap() -> None:
    """Both may load together; neither excludes the other."""
    router = SkillRouter.from_loader(SkillLoader())
    decision = router.route(
        "explore the delta table at /lakehouse/default/Tables/dbo/orders"
    )
    assert SKILL in decision.active
    assert "data_exploration" in decision.active


# --------------------------------------------------------------------------
# Layer 2: the gate inside data_exploration
# --------------------------------------------------------------------------


def test_data_exploration_has_the_delta_gate() -> None:
    content = SkillLoader().load("data_exploration").content
    gate = content.split("## Mandatory first-turn protocol")[0]

    # The gate must come BEFORE the discovery protocol, or the model will have
    # already run open()/read_json_auto against a table directory.
    assert "Gate: is this path a Delta table?" in gate
    assert "_delta_log" in gate
    assert "delta_scan" in gate
    assert "/Tables/" in gate
    assert "abfss://" in gate
    assert SKILL in gate  # points at the deep skill
    assert "path.replace" in gate


def test_data_exploration_flags_the_glob_anti_pattern() -> None:
    content = SkillLoader().load("data_exploration").content
    anti = content.split("## Anti-patterns")[1]
    assert "read_parquet" in anti and "/Tables/" in anti
    assert "tombstoned" in anti


# --------------------------------------------------------------------------
# Layer 3: the claims, executed
# --------------------------------------------------------------------------


def test_do_dont_table_covers_the_load_bearing_rules() -> None:
    content = SkillLoader().load(SKILL).content
    table = content.split("## Do / Don't")[1].split("## Step 1")[0]

    rows = [ln for ln in table.splitlines() if ln.startswith("|")]
    assert len(rows) >= 10, "Do/Don't table looks truncated"
    # Every row must carry a reason, not just a rule.
    for row in rows[2:]:
        assert row.count("|") >= 4, f"row missing the why column: {row}"

    for rule in ("read_parquet", "con.register", "ACCESS_TOKEN ?", "ENDPOINT", "num_records"):
        assert rule in table, f"Do/Don't table lost the {rule} rule"


def test_provenance_section_states_what_was_and_was_not_verified() -> None:
    """The skill claims measured numbers and must name its live-test boundary."""
    content = SkillLoader().load(SKILL).content
    prov = content.split("## Where these rules come from")[1]

    assert "16 rows / sum 1699.0" in prov
    assert "6 rows / sum 1149.0" in prov
    assert "relative URL without a base" in prov
    assert "live OneLake" in prov
    assert "schema-enabled Lakehouse" in prov
    assert "did not exercise" in prov
    assert "deletion vectors" in prov


def _skill_code_blocks(name: str = SKILL) -> list[str]:
    content = SkillLoader().load(name).content
    return re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)


def _open_delta_block() -> str:
    blocks = [b for b in _skill_code_blocks() if "def open_delta" in b]
    assert len(blocks) == 1, f"expected exactly one open_delta block, got {len(blocks)}"
    return blocks[0]


@pytest.fixture
def duck():
    """A DuckDB connection that is always closed.

    Left open, the connection keeps handles on parquet files under tmp_path,
    which on Windows collides with pytest's end-of-session tmpdir cleanup.
    """
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    try:
        yield con
    finally:
        con.close()


@pytest.fixture
def delta_table(tmp_path):
    """A table whose history guarantees tombstoned files on disk.

    10 rows, delete ids 6-10, append one row of 999.0 => ground truth is 6 rows
    summing to 1149.0, while the raw parquet on disk still holds the pre-delete
    file. These are the exact numbers quoted in the skill.
    """
    pa = pytest.importorskip("pyarrow")
    dl = pytest.importorskip("deltalake")

    path = tmp_path / "sales"
    dl.write_deltalake(
        str(path),
        pa.table({"id": list(range(1, 11)), "amt": [float(i * 10) for i in range(1, 11)]}),
    )
    dl.DeltaTable(str(path)).delete("id > 5")
    dl.write_deltalake(str(path), pa.table({"id": [99], "amt": [999.0]}), mode="append")
    return path


def test_the_glob_really_is_wrong_and_delta_scan_really_is_right(delta_table, duck) -> None:
    """The number in the skill is a measurement, so pin it as an invariant.

    If a future duckdb/deltalake makes the naive glob accidentally correct, this
    fails and the skill's headline warning should be re-checked rather than
    quietly left in place.
    """
    dl = pytest.importorskip("deltalake")
    con = duck
    p = delta_table.as_posix()

    truth = (6, 1149.0)

    wrong = con.sql(f"SELECT count(*), sum(amt) FROM read_parquet('{p}/**/*.parquet')").fetchone()
    assert wrong != truth, "the glob is supposed to be wrong; recheck the skill's premise"
    assert wrong[0] > truth[0], f"expected inflated row count from tombstones, got {wrong}"

    ds = dl.DeltaTable(str(delta_table)).to_pyarrow_dataset()  # noqa: F841 - duckdb scan
    assert con.sql("SELECT count(*), sum(amt) FROM ds").fetchone() == truth

    con.sql("INSTALL delta; LOAD delta;")
    assert con.sql(f"SELECT count(*), sum(amt) FROM delta_scan('{p}')").fetchone() == truth


def test_open_delta_block_runs_against_a_mounted_style_path(delta_table, duck) -> None:
    """Execute the copy-paste resolver from the Markdown verbatim.

    A mount is an ordinary POSIX path, so a local directory exercises exactly
    the attached-lakehouse branch.
    """
    pytest.importorskip("deltalake")

    ns: dict = {}
    exec(compile(_open_delta_block(), "<delta_lakehouse skill>", "exec"), ns)

    con, relation = ns["open_delta"](delta_table.as_posix(), con=duck)
    assert "delta_scan" in relation, "a local path must not take the remote branch"
    assert con.sql(f"SELECT count(*), sum(amt) FROM {relation}").fetchone() == (6, 1149.0)


def test_open_delta_escapes_apostrophes_in_paths() -> None:
    class RecordingConnection:
        def __init__(self):
            self.queries = []

        def sql(self, query):
            self.queries.append(query)

    ns = _attach_ns()
    ns["_prepare"] = lambda con, _path: con
    con = RecordingConnection()

    _, relation = ns["open_delta"](
        "/lakehouse/default/Tables/o'hare", con=con
    )

    assert relation == "delta_scan('/lakehouse/default/Tables/o''hare')"
    assert con.queries == [
        "SELECT 1 FROM delta_scan("
        "'/lakehouse/default/Tables/o''hare') LIMIT 1"
    ]


class _NoDeltaExtension:
    """A connection proxy where `INSTALL delta` fails, as on a no-egress runtime."""

    def __init__(self, con):
        self._con = con
        self.install_attempted = False

    def sql(self, query, *a, **kw):
        if "INSTALL delta" in query:
            self.install_attempted = True
            raise RuntimeError("IO Error: Failed to download extension 'delta'")
        return self._con.sql(query, *a, **kw)

    def __getattr__(self, name):
        return getattr(self._con, name)


def test_open_delta_falls_back_to_delta_rs_when_the_extension_is_unavailable(
    delta_table, duck, capsys
) -> None:
    """No-egress runtimes must still get the correct answer, never the glob."""
    pytest.importorskip("deltalake")

    ns: dict = {}
    exec(compile(_open_delta_block(), "<delta_lakehouse skill>", "exec"), ns)

    proxy = _NoDeltaExtension(duck)
    con, relation = ns["open_delta"](delta_table.as_posix(), con=proxy)

    assert proxy.install_attempted
    assert relation == "delta_tbl", "fallback must register a named relation"
    assert "falling back to delta-rs" in capsys.readouterr().out

    # Same answer as delta_scan: both engines honor _delta_log.
    assert con.sql(f"SELECT count(*), sum(amt) FROM {relation}").fetchone() == (6, 1149.0)

    # And it must survive being queried from another scope, which is what the
    # replacement-scan approach could not do.
    def query_elsewhere(c):
        return c.sql(f"SELECT sum(amt) FROM {relation}").fetchone()[0]

    assert query_elsewhere(con) == 1149.0


@pytest.fixture
def lakehouse(tmp_path):
    """A Tables/ root mixing both Fabric layouts, plus decoys.

    Tables/customers          flat
    Tables/orders             flat, partitioned (must not be walked into)
    Tables/dbo/products       schema-nested
    Tables/dbo/returns        schema-nested
    Tables/not_a_table/       a directory with no _delta_log
    """
    pa = pytest.importorskip("pyarrow")
    dl = pytest.importorskip("deltalake")

    root = tmp_path / "lh" / "Tables"
    dl.write_deltalake(str(root / "customers"), pa.table(
        {"cust_id": [1, 2, 3], "name": ["a", "b", "c"]}))
    dl.write_deltalake(str(root / "orders"), pa.table(
        {"order_id": [10, 11, 12, 13], "cust_id": [1, 1, 2, 3],
         "amt": [5.0, 7.0, 9.0, 11.0], "region": ["e", "w", "e", "w"]}),
        partition_by=["region"])
    dl.write_deltalake(str(root / "dbo" / "products"), pa.table(
        {"sku": ["x", "y"], "price": [1.5, 2.5]}))
    dl.write_deltalake(str(root / "dbo" / "returns"), pa.table(
        {"order_id": [11], "reason": ["damaged"]}))
    (root / "not_a_table").mkdir(parents=True, exist_ok=True)
    (root / "not_a_table" / "README.txt").write_text("junk", encoding="utf-8")
    return root


def _attach_ns():
    ns: dict = {}
    exec(compile(_open_delta_block(), "<delta_lakehouse skill>", "exec"), ns)
    return ns


def test_find_delta_tables_handles_both_layouts(lakehouse) -> None:
    pytest.importorskip("deltalake")
    found = _attach_ns()["find_delta_tables"](lakehouse.as_posix())

    assert set(found) == {"customers", "orders", "dbo.products", "dbo.returns"}
    assert "not_a_table" not in found, "a directory without _delta_log is not a table"


def test_find_delta_tables_does_not_walk_into_partitions(lakehouse) -> None:
    """`orders` is partitioned; region=... dirs must never register as tables."""
    pytest.importorskip("deltalake")
    found = _attach_ns()["find_delta_tables"](lakehouse.as_posix())
    assert not [k for k in found if "region=" in k], found


def test_attach_lakehouse_registers_everything_queryable(lakehouse, duck) -> None:
    pytest.importorskip("deltalake")
    con, attached, skipped = _attach_ns()["attach_lakehouse"](lakehouse.as_posix(), con=duck)

    assert set(attached) == {"customers", "orders", "dbo.products", "dbo.returns"}
    assert skipped == []

    seen = {(s, t) for s, t in con.sql(
        "SELECT table_schema, table_name FROM information_schema.tables").fetchall()}
    assert ("main", "customers") in seen
    assert ("main", "orders") in seen
    assert ("dbo", "products") in seen


def test_attach_lakehouse_falls_back_when_delta_scan_is_unavailable(
    lakehouse, duck
) -> None:
    pytest.importorskip("deltalake")
    duck.execute("SET autoinstall_known_extensions = false")
    duck.execute("SET autoload_known_extensions = false")
    ns = _attach_ns()
    ns["_prepare"] = lambda con, _path: con

    con, attached, skipped = ns["attach_lakehouse"](
        lakehouse.as_posix(), con=duck
    )

    assert set(attached) == {
        "customers",
        "orders",
        "dbo.products",
        "dbo.returns",
    }
    assert skipped == []
    assert con.sql("SELECT count(*) FROM customers").fetchone() == (3,)
    assert con.sql(
        'SELECT count(*) FROM "dbo"."returns"'
    ).fetchone() == (1,)


def test_attach_lakehouse_falls_back_when_prepare_fails(
    lakehouse, duck
) -> None:
    pytest.importorskip("deltalake")
    ns = _attach_ns()

    def fail_prepare(_con, _path):
        raise RuntimeError("DuckDB OneLake secret unavailable")

    ns["_prepare"] = fail_prepare

    con, attached, skipped = ns["attach_lakehouse"](
        lakehouse.as_posix(), con=duck
    )

    assert set(attached) == {
        "customers",
        "orders",
        "dbo.products",
        "dbo.returns",
    }
    assert skipped == []
    assert con.sql("SELECT count(*) FROM customers").fetchone() == (3,)


def test_attached_tables_join_without_per_table_setup(lakehouse, duck) -> None:
    """The point of attaching: joins become ordinary SQL."""
    pytest.importorskip("deltalake")
    con, _, _ = _attach_ns()["attach_lakehouse"](lakehouse.as_posix(), con=duck)

    rows = con.sql("""
        SELECT c.name, count(*) AS n, sum(o.amt) AS total
        FROM orders o JOIN customers c USING (cust_id)
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    assert rows == [("a", 2, 12.0), ("b", 1, 9.0), ("c", 1, 11.0)]

    # And across a schema boundary.
    assert con.sql(
        'SELECT count(*) FROM "dbo"."returns" r JOIN orders o USING (order_id)'
    ).fetchone() == (1,)


def test_attach_accepts_a_single_schema_root(lakehouse, duck) -> None:
    """Point at Tables/dbo to get just that schema's tables, unqualified."""
    pytest.importorskip("deltalake")
    con, attached, _ = _attach_ns()["attach_lakehouse"](
        (lakehouse / "dbo").as_posix(), con=duck)

    assert set(attached) == {"products", "returns"}
    assert con.sql("SELECT count(*) FROM products").fetchone() == (2,)


def test_one_unreadable_table_does_not_abort_the_attach(lakehouse, duck) -> None:
    """A broken table lands in `skipped` with a reason; the rest still attach."""
    pytest.importorskip("deltalake")
    broken = lakehouse / "broken"
    (broken / "_delta_log").mkdir(parents=True)
    (broken / "_delta_log" / "00000000000000000000.json").write_text("not json{", encoding="utf-8")

    ns = _attach_ns()
    assert "broken" in ns["find_delta_tables"](lakehouse.as_posix()), "should be detected"

    con, attached, skipped = ns["attach_lakehouse"](lakehouse.as_posix(), con=duck)
    assert "broken" in dict(skipped), f"expected broken to be skipped, got {skipped}"
    assert {"customers", "orders", "dbo.products", "dbo.returns"} <= set(attached)
    assert con.sql("SELECT count(*) FROM customers").fetchone() == (3,)


def test_attach_is_read_only(lakehouse, duck) -> None:
    dl = pytest.importorskip("deltalake")
    before = {p: dl.DeltaTable(str(lakehouse / p.replace(".", "/"))).version()
              for p in ("customers", "orders", "dbo.products")}
    _attach_ns()["attach_lakehouse"](lakehouse.as_posix(), con=duck)
    after = {p: dl.DeltaTable(str(lakehouse / p.replace(".", "/"))).version()
             for p in before}
    assert before == after


def test_attach_creates_views_not_copies(lakehouse, duck) -> None:
    """Views keep it lazy and read-only; a CTAS would copy the whole lakehouse."""
    pytest.importorskip("deltalake")
    con, _, _ = _attach_ns()["attach_lakehouse"](lakehouse.as_posix(), con=duck)
    kinds = dict(con.sql(
        "SELECT table_name, table_type FROM information_schema.tables").fetchall())
    assert set(kinds.values()) == {"VIEW"}, kinds


def test_attach_reports_an_unlistable_root_instead_of_crashing(duck) -> None:
    """Reported from a real Fabric session: a bad root raised a raw Azure traceback.

    find_delta_tables guarded its inner loop but not the top-level listing, so
    attach_lakehouse died mid-cell instead of saying what was wrong.
    """
    pytest.importorskip("deltalake")
    ns = _attach_ns()

    boom = "HttpResponseError: Request Failed with WorkspaceId and ArtifactId " \
           "should be either valid Guids or valid Names ErrorCode:FriendlyNameSupportDisabled"

    def _explode(_path):
        raise RuntimeError(boom)

    ns["_ls"] = _explode
    with pytest.raises(RuntimeError) as exc_info:
        ns["attach_lakehouse"]("abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables",
                               con=duck)

    msg = str(exc_info.value)
    assert "Cannot list" in msg
    # The whole point: the error has to say what to do about it.
    assert "GUID" in msg
    assert "onelake_root" in msg
    assert ".Lakehouse suffix after a GUID" in msg


def test_onelake_root_does_not_suffix_a_guid() -> None:
    """`.Lakehouse` belongs on a name; appending it to a GUID breaks the path."""
    ns = _attach_ns()
    guid = "0333b170-84c4-493f-9a49-515071a0040a"

    assert ns["_looks_like_guid"](guid) is True
    assert ns["_looks_like_guid"]("my_lakehouse") is False

    built = ns["onelake_root"](workspace="ws-guid", lakehouse=guid)
    assert built.endswith(f"/{guid}/Tables"), built
    assert ".Lakehouse" not in built

    named = ns["onelake_root"](workspace="ws-guid", lakehouse="my_lakehouse")
    assert named.endswith("/my_lakehouse.Lakehouse/Tables"), named

    scoped = ns["onelake_root"](workspace="ws", lakehouse=guid, schema="dbo")
    assert scoped.endswith("/Tables/dbo")


def test_skill_documents_the_friendly_name_trap() -> None:
    content = SkillLoader().load(SKILL).content
    assert "FriendlyNameSupportDisabled" in content
    assert "valid Guids or valid Names" in content
    body = " ".join(content.split("**OneLake `abfss://`**")[1].split("## Step 1b")[0].split())
    assert "must NOT be appended to a GUID" in body


def test_skill_documents_attaching_a_lakehouse() -> None:
    content = SkillLoader().load(SKILL).content
    assert "## Step 1b" in content
    body = " ".join(content.split("## Step 1b")[1].split("## Step 2")[0].split())
    assert "attach_lakehouse" in body
    for point in ("Naming", "costs", "Eager binding", "read-only"):
        assert point in body, f"Step 1b lost the {point} note"
    # The cost note must be honest about the per-table round trip.
    assert "one round trip per table" in body


@pytest.fixture
def wide_table(tmp_path):
    """A partitioned table with nulls and repeated keys, for exploration checks."""
    pa = pytest.importorskip("pyarrow")
    dl = pytest.importorskip("deltalake")

    n = 400
    path = tmp_path / "orders"
    dl.write_deltalake(
        str(path),
        pa.table(
            {
                "order_id": list(range(1, n + 1)),
                "region": ["east", "west", "north", "south"] * (n // 4),
                "customer": [f"cust{i % 37:03d}" for i in range(n)],
                "amount": [float((i * 7) % 500) for i in range(n)],
                "note": [None if i % 3 else f"n{i}" for i in range(n)],
            }
        ),
        partition_by=["region"],
    )
    dl.DeltaTable(str(path)).delete("order_id > 380")
    return path


def _discovery_block() -> str:
    blocks = [b for b in _skill_code_blocks() if "=== profile ===" in b]
    assert len(blocks) == 1, f"expected one discovery block, got {len(blocks)}"
    return blocks[0]


def test_discovery_block_runs_and_prints_every_section(wide_table, duck, capsys) -> None:
    """Execute Step 2 verbatim from the Markdown against a real table."""
    pytest.importorskip("deltalake")

    ns: dict = {"path": wide_table.as_posix()}
    exec(compile(_open_delta_block(), "<open_delta>", "exec"), ns)
    # Reuse the closing fixture connection rather than opening an untracked one.
    original = ns["open_delta"]
    ns["open_delta"] = lambda p, con=None: original(p, con=duck)

    exec(compile(_discovery_block(), "<discovery>", "exec"), ns)

    out = capsys.readouterr().out
    for section in ("=== table ===", "=== schema ===", "=== sample ===", "=== profile ==="):
        assert section in out, f"discovery block did not print {section}"

    # Schema must surface real column names, which is the whole point.
    for col in ("order_id", "region", "customer", "amount", "note"):
        assert col in out, f"schema output missing {col}"

    assert "partitions" in out and "region" in out
    assert "nulls=" in out and "~uniq=" in out


def test_discovery_output_stays_small(wide_table, duck, capsys) -> None:
    """Discovery is only mandatory because it is cheap. Keep it that way."""
    pytest.importorskip("deltalake")

    ns: dict = {"path": wide_table.as_posix()}
    exec(compile(_open_delta_block(), "<open_delta>", "exec"), ns)
    original = ns["open_delta"]
    ns["open_delta"] = lambda p, con=None: original(p, con=duck)
    exec(compile(_discovery_block(), "<discovery>", "exec"), ns)

    out = capsys.readouterr().out
    assert len(out) < 3000, f"discovery printed {len(out)} chars; too big for turn 1"
    assert out.count("\n") < 40


def test_limit_is_not_a_sample_but_using_sample_is(wide_table, duck) -> None:
    """Pins the trap the skill warns about, on a partitioned table."""
    try:
        duck.sql("INSTALL delta; LOAD delta;")
    except Exception as exc:  # pragma: no cover - offline machine
        pytest.skip(f"delta extension unavailable: {exc}")
    T = f"delta_scan('{wide_table.as_posix()}')"

    limited = {r[0] for r in duck.sql(f"SELECT region FROM {T} LIMIT 5").fetchall()}
    assert len(limited) == 1, "LIMIT was expected to read a single partition file"

    sampled = {r[0] for r in duck.sql(f"SELECT region FROM {T} USING SAMPLE 40 ROWS").fetchall()}
    assert len(sampled) > 1, "USING SAMPLE should cross partitions"


def test_approx_unique_is_not_exact(wide_table, duck) -> None:
    """The skill tells the model not to quote approx_unique. Show why."""
    try:
        duck.sql("INSTALL delta; LOAD delta;")
    except Exception as exc:  # pragma: no cover - offline machine
        pytest.skip(f"delta extension unavailable: {exc}")
    T = f"delta_scan('{wide_table.as_posix()}')"

    rel = duck.sql(f"SUMMARIZE SELECT * FROM {T}")
    cols = [d[0] for d in rel.description]
    prof = {dict(zip(cols, r))["column_name"]: dict(zip(cols, r)) for r in rel.fetchall()}

    exact, total = duck.sql(f"SELECT count(DISTINCT order_id), count(*) FROM {T}").fetchone()
    assert exact == total == 380
    approx = int(prof["order_id"]["approx_unique"])
    assert approx != exact, "approx_unique happened to be exact; the warning still stands"

    # And the profile must expose the fields the skill reads off it.
    for field in ("column_type", "null_percentage", "approx_unique", "min", "max"):
        assert field in prof["note"]
    assert float(prof["note"]["null_percentage"]) > 50  # the mostly-null column


def test_skill_prefers_fetchall_over_show() -> None:
    """`.show()` crashes on cp1252 stdout, so no example may use it."""
    for block in _skill_code_blocks():
        assert ".show()" not in block, "example uses .show(); it breaks on Windows stdout"

    content = SkillLoader().load(SKILL).content
    assert "UnicodeEncodeError" in content


def test_discovery_is_declared_mandatory_and_schema_first() -> None:
    content = SkillLoader().load(SKILL).content
    heading = [ln for ln in content.splitlines() if ln.startswith("## Step 2")]
    assert heading and "mandatory" in heading[0].lower()
    assert "BEFORE any aggregation" in heading[0]

    body = " ".join(content.split("## Step 2")[1].split("## Step 3")[0].split())
    # The Excel-style rule, stated explicitly.
    assert "you do not know what the columns are called until you have printed them" in body
    for signal in ("USING SAMPLE", "approx_unique", "null", "partitions", "reader_features"):
        assert signal in body


@pytest.fixture
def dv_table(tmp_path):
    """A table with the deletionVectors reader feature, as Fabric commonly has."""
    pa = pytest.importorskip("pyarrow")
    dl = pytest.importorskip("deltalake")

    path = tmp_path / "dv_sales"
    dl.write_deltalake(
        str(path),
        pa.table({"id": list(range(1, 11)), "amt": [float(i * 10) for i in range(1, 11)]}),
        configuration={"delta.enableDeletionVectors": "true"},
    )
    dl.DeltaTable(str(path)).delete("id > 5")
    return path


def test_delta_rs_reader_cannot_read_deletion_vector_tables(dv_table) -> None:
    """Pins the limitation the skill's hard-fail branch exists for.

    If a future deltalake supports deletion vectors, this fails and the resolver
    can be simplified back to a plain fallback.
    """
    dl = pytest.importorskip("deltalake")
    from deltalake.exceptions import DeltaProtocolError

    dt = dl.DeltaTable(str(dv_table))
    assert "deletionVectors" in (dt.protocol().reader_features or [])

    with pytest.raises(DeltaProtocolError):
        dt.to_pyarrow_dataset()
    with pytest.raises(DeltaProtocolError):
        dt.to_pandas()


def test_metadata_still_works_on_deletion_vector_tables(dv_table) -> None:
    """Metadata surviving while readers fail is what tempts a parquet workaround."""
    pa = pytest.importorskip("pyarrow")
    dl = pytest.importorskip("deltalake")

    dt = dl.DeltaTable(str(dv_table))
    assert pa.table(dt.get_add_actions()).num_rows >= 1
    assert dt.history(3)
    assert dt.schema() is not None
    assert dt.file_uris()


def test_delta_scan_reads_deletion_vector_tables(dv_table, duck) -> None:
    try:
        duck.sql("INSTALL delta; LOAD delta;")
    except Exception as exc:  # pragma: no cover - offline machine
        pytest.skip(f"delta extension unavailable: {exc}")
    assert duck.sql(f"SELECT count(*) FROM delta_scan('{dv_table.as_posix()}')").fetchone() == (5,)


def test_open_delta_refuses_rather_than_falling_back_to_parquet(dv_table, duck) -> None:
    """The whole point: when no Delta reader works, stop. Never approximate."""
    pytest.importorskip("deltalake")

    ns: dict = {}
    exec(compile(_open_delta_block(), "<delta_lakehouse skill>", "exec"), ns)

    proxy = _NoDeltaExtension(duck)  # delta extension unavailable AND DV table
    with pytest.raises(RuntimeError) as exc_info:
        ns["open_delta"](dv_table.as_posix(), con=proxy)

    message = str(exc_info.value)
    assert "read_parquet" in message, "the error must name the thing not to do"
    assert "delta extension" in message


def test_no_skill_example_ever_reads_parquet_from_a_table() -> None:
    """Absolute rule: nothing in this skill may model a parquet read of a table."""
    for block in _skill_code_blocks():
        for call in ("read_parquet(", "read_csv(", "parquet_scan("):
            if call not in block:
                continue
            # The only permitted parquet/csv reads are of Files/ inputs and the
            # COPY ... TO output, never of a Tables/ path or a file list.
            assert "/Tables/" not in block, f"{call} appears alongside a Tables path"
            assert "file_uris" not in block, f"{call} fed from file_uris()"


def test_file_uris_is_documented_as_metadata_not_a_read_path() -> None:
    content = SkillLoader().load(SKILL).content
    assert "read_parquet(dt.file_uris())" in content
    assert "deletion vectors" in content.lower()
    # The rule must be stated without an escape hatch. Normalize wrapping so the
    # assertion tracks the wording, not the line breaks.
    headline = " ".join(content.split("## Do / Don't")[0].lower().split())
    assert "no exception" in headline
    assert "never a read path" in headline or "never a read path" in " ".join(
        content.lower().split()
    )


@pytest.mark.parametrize(
    "path",
    [
        "/lakehouse/default/Tables/orders",
        "/lakehouse/default/Tables/dbo/orders",
        "C:/mnt/lakehouse/Tables/orders",
    ],
)
def test_delta_opts_returns_none_for_a_mounted_path(path: str) -> None:
    """A mount needs no token; handing it storage_options is an anti-pattern."""
    ns: dict = {}
    exec(compile(_open_delta_block(), "<delta_lakehouse skill>", "exec"), ns)
    assert ns["delta_opts"](path) is None


def test_delta_opts_builds_fabric_token_options_for_abfss() -> None:
    """The abfss branch must produce the option names delta-rs actually accepts."""
    ns: dict = {}
    exec(compile(_open_delta_block(), "<delta_lakehouse skill>", "exec"), ns)
    ns["_storage_token"] = lambda: "fake-token"

    opts = ns["delta_opts"](
        "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/fact"
    )
    assert opts == {"bearer_token": "fake-token", "use_fabric_endpoint": "true"}


def test_open_delta_branches_on_url_shape() -> None:
    """The remote branch must wire up azure + a token secret, not fall through."""
    src = _open_delta_block()
    assert 'is_remote = "://" in path' in src
    assert "INSTALL azure" in src
    assert "PROVIDER access_token" in src
    assert "ACCOUNT_NAME 'onelake'" in src
    # The token is bound as a parameter, never interpolated into the SQL text.
    assert "ACCESS_TOKEN ?" in src


def test_metadata_only_discovery_matches_a_full_scan(delta_table, duck) -> None:
    """Step 2 claims row counts come back without scanning data files."""
    pa = pytest.importorskip("pyarrow")
    dl = pytest.importorskip("deltalake")
    dt = dl.DeltaTable(str(delta_table))
    adds = pa.table(dt.get_add_actions()).to_pydict()
    from_metadata = sum(adds["num_records"])

    con = duck
    ds = dt.to_pyarrow_dataset()  # noqa: F841 - duckdb scan
    from_scan = con.sql("SELECT count(*) FROM ds").fetchone()[0]

    assert from_metadata == from_scan == 6


def test_deltalake_api_drift_traps_are_still_real() -> None:
    """Both traps called out in the anti-patterns must still bite.

    When a future deltalake restores `.files()` or returns pyarrow again, these
    fail and the corresponding bullets should come out of the skill.
    """
    pa = pytest.importorskip("pyarrow")
    dl = pytest.importorskip("deltalake")

    assert not hasattr(dl.DeltaTable, "files"), "`.files()` is back; drop that bullet"
    assert hasattr(dl.DeltaTable, "file_uris")

    content = SkillLoader().load(SKILL).content
    assert "file_uris" in content
    assert "arro3" in content
    assert "pa.table(dt.get_add_actions())" in content
    assert pa is not None


def test_read_only_boundary_names_real_apis() -> None:
    """Every API the skill forbids must actually exist, or the ban is theatre."""
    dl = pytest.importorskip("deltalake")

    content = SkillLoader().load(SKILL).content
    boundary = content.split("## Read-only boundary")[1].split("## Anti-patterns")[0]

    forbidden = set(re.findall(r"`dt\.([a-z_]+)\(", boundary))
    assert forbidden, "boundary section parsed empty"
    missing = sorted(n for n in forbidden if not hasattr(dl.DeltaTable, n))
    assert not missing, f"skill forbids non-existent DeltaTable methods: {missing}"

    # The two that cause irreversible loss must be named explicitly.
    assert "vacuum" in forbidden
    assert "delete" in forbidden
    assert "write_deltalake" in boundary

    # optimize.* lives on an accessor rather than DeltaTable itself.
    assert hasattr(dl.DeltaTable, "optimize")
    assert "z_order" in boundary and "compact()" in boundary


def test_skill_prescribes_no_mutating_call_anywhere_in_its_code() -> None:
    """Guard the examples themselves, not just the prose."""
    mutators = (
        ".vacuum(",
        ".merge(",
        ".restore(",
        ".repair(",
        ".create_checkpoint(",
        ".cleanup_metadata(",
        ".compact_logs(",
        "write_deltalake(",
        ".optimize.compact(",
        ".optimize.z_order(",
    )
    for block in _skill_code_blocks():
        for m in mutators:
            assert m not in block, f"skill example calls a mutating API: {m}"


def test_every_code_block_survives_the_default_security_policy() -> None:
    """A skill whose own examples get rejected is worse than no skill.

    The worker runs LM-emitted code through SecurityPolicy first, so every block
    the model is told to copy must pass the default denylist.
    """
    from fabric_rlm.security import SecurityPolicy

    policy = SecurityPolicy.default()
    for i, block in enumerate(_skill_code_blocks()):
        violation = policy.validate_code(block)
        assert violation is None, f"block {i} rejected by default policy: {violation}"


def test_gate_snippet_in_data_exploration_also_survives_the_policy() -> None:
    from fabric_rlm.security import SecurityPolicy

    content = SkillLoader().load("data_exploration").content
    gate = content.split("## Mandatory first-turn protocol")[0]
    blocks = re.findall(r"```python\n(.*?)\n```", gate, re.DOTALL)
    assert blocks, "the gate lost its delta_scan example"

    policy = SecurityPolicy.default()
    for block in blocks:
        assert policy.validate_code(block) is None


def test_examples_write_outputs_to_files_not_tables() -> None:
    for block in _skill_code_blocks():
        if "COPY (" in block:
            assert "/Files/" in block
            assert "/Tables/" not in block.split("COPY (")[1]


# --------------------------------------------------------------------------
# abfss specifics
# --------------------------------------------------------------------------


def test_prepare_reuses_an_existing_onelake_secret() -> None:
    class ExistingSecretResult:
        @staticmethod
        def fetchone():
            return (1,)

    class ExistingSecretConnection:
        @staticmethod
        def sql(query):
            if "duckdb_secrets()" in query:
                return ExistingSecretResult()
            return None

        @staticmethod
        def execute(*_args, **_kwargs):
            raise AssertionError("an active OneLake secret must not be replaced")

    ns = _attach_ns()
    ns["_storage_token"] = lambda: pytest.fail(
        "an existing secret must not request a replacement token"
    )

    assert ns["_prepare"](
        ExistingSecretConnection(),
        "abfss://ws@onelake.dfs.fabric.microsoft.com/lh/Tables",
    )


def test_azure_secret_form_in_the_skill_is_accepted_by_duckdb(duck) -> None:
    """The CREATE SECRET the skill prescribes must parse. No network needed."""
    con = duck
    try:
        con.sql("INSTALL azure; LOAD azure;")
    except Exception as exc:  # pragma: no cover - offline machine
        pytest.skip(f"azure extension unavailable: {exc}")

    con.execute(
        "CREATE OR REPLACE SECRET onelake_tok "
        "(TYPE azure, PROVIDER access_token, ACCESS_TOKEN ?, ACCOUNT_NAME 'onelake')",
        ["dummy-token"],
    )
    assert con.sql("SELECT count(*) FROM duckdb_secrets()").fetchone()[0] >= 1


def test_bare_hostname_endpoint_is_the_documented_trap(duck) -> None:
    """A bare ENDPOINT host fails at URL construction, before any socket.

    The failure mode is obscure ("relative URL without a base"), which is why the
    skill tells the model to omit ENDPOINT or give it a scheme.
    """
    con = duck
    try:
        con.sql("INSTALL azure; LOAD azure; INSTALL delta; LOAD delta;")
    except Exception as exc:  # pragma: no cover - offline machine
        pytest.skip(f"extensions unavailable: {exc}")

    con.execute(
        "CREATE OR REPLACE SECRET s (TYPE azure, PROVIDER access_token, "
        "ACCESS_TOKEN 'dummy', ACCOUNT_NAME 'onelake', "
        "ENDPOINT 'onelake.dfs.fabric.microsoft.com')"
    )
    with pytest.raises(Exception) as exc_info:
        con.sql(
            "SELECT * FROM delta_scan("
            "'abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/t') LIMIT 1"
        ).fetchall()
    assert "parse source url" in str(exc_info.value).lower()

    content = SkillLoader().load(SKILL).content
    assert "relative URL without a base" in content
    assert "https://onelake.dfs.fabric.microsoft.com" in content
