import ast
from pathlib import Path


NOTEBOOK = (
    Path(__file__).parents[1]
    / "examples"
    / "notebooks"
    / "development"
    / "rlm_knowledge_benchmark_matrix.py"
)


def test_benchmark_scopes_lakehouse_discovery_to_orders_table() -> None:
    source = NOTEBOOK.read_text(encoding="utf-8")
    setup = source.split(
        "# ## Prepare exact Delta, CSV, and Parquet benchmark sources",
        maxsplit=1,
    )[1].split("def pick_column", maxsplit=1)[0]
    tree = ast.parse(setup)

    constructors = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "LakehouseSource"
    ]
    assert len(constructors) == 1
    assert ast.unparse(constructors[0].args[0]) == (
        "f'{LAKEHOUSE_ROOT}/Tables/dbo/knowledge_validation_orders'"
    )

    assignments = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    resolved = assignments["orders_lakehouse"]
    assert isinstance(resolved, ast.Call)
    assert isinstance(resolved.func, ast.Attribute)
    assert resolved.func.attr == "resolve"
    assert resolved.func.value is constructors[0]

    query = assignments["orders_packet"]
    assert isinstance(query, ast.Call)
    assert isinstance(query.func, ast.Attribute)
    assert isinstance(query.func.value, ast.Name)
    assert query.func.value.id == "orders_lakehouse"

    entry = assignments["orders_entry"]
    assert isinstance(entry, ast.Subscript)
    assert isinstance(entry.value, ast.Attribute)
    assert isinstance(entry.value.value, ast.Name)
    assert entry.value.value.id == "orders_lakehouse"


def test_benchmark_persists_failures_before_results_are_written() -> None:
    source = NOTEBOOK.read_text(encoding="utf-8")

    bootstrap_index = source.index("RUN_LOG_PATH =")
    install_index = source.index("%pip install")
    assert bootstrap_index < install_index

    assert "def persist_run_log(" in source
    assert "def persist_uncaught_exception(" in source
    assert "get_ipython().set_custom_exc(" in source
    assert source.count("install_uncaught_exception_logger()") >= 2
    assert source.count('        "failed",') >= 2
    assert "traceback.format_exception(" in source
    assert 'phase="trial"' in source
    assert "task_id=task.task_id" in source
    assert "arm=arm" in source
