from collections import Counter
from datetime import date, datetime
import json

import duckdb
import pytest

from evaluation.generalization.fixtures import generate_fixtures
from evaluation.generalization import representations, runner


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    root = tmp_path_factory.mktemp("representations")
    generate_fixtures(root, large_rows=20)
    representations.prepare_representations(root)
    return root


def _rows(values):
    def encode(value):
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        raise TypeError(type(value).__name__)

    return Counter(json.dumps(row, default=encode) for row in values)


@pytest.mark.parametrize("domain", ["inventory", "manufacturing", "service"])
@pytest.mark.parametrize("variant", ["descriptive", "abbreviated", "camel"])
@pytest.mark.parametrize("representation", ["parquet", "lakehouse"])
def test_representations_preserve_every_row_and_column(fixtures, domain, variant, representation):
    sources = runner._domain_sources(fixtures, domain, variant, representation=representation)
    csv_sources = representations.domain_sources(fixtures, domain, variant, "csv")
    assert sources.keys() == csv_sources.keys()
    assert all(not name.endswith("_large") for name in sources)
    with duckdb.connect() as connection:
        for alias, source in sources.items():
            cursor = connection.execute("SELECT * FROM read_csv_auto(?)", [csv_sources[alias]])
            columns = [column[0] for column in cursor.description]
            expected = cursor.fetchall()
            if representation == "parquet":
                cursor = connection.execute("SELECT * FROM read_parquet(?)", [source])
                actual_columns = [column[0] for column in cursor.description]
                actual_rows = cursor.fetchall()
            else:
                result = source.query(
                    f'SELECT * FROM "{alias}"', sources={alias: alias}, max_rows=1000,
                )
                actual_columns, actual_rows = result["columns"], result["rows"]
                assert not result["truncated"]
            assert actual_columns == columns
            assert _rows(actual_rows) == _rows(expected)


def test_preparation_refuses_to_replace_existing_representations(fixtures):
    with pytest.raises(FileExistsError):
        representations.prepare_representations(fixtures)


def test_unknown_or_unprepared_representation_is_explicit(tmp_path):
    generate_fixtures(tmp_path, large_rows=20)
    with pytest.raises(ValueError, match="representation"):
        representations.domain_sources(tmp_path, "inventory", "descriptive", "oracle")
    with pytest.raises(FileNotFoundError):
        representations.domain_sources(tmp_path, "inventory", "descriptive", "parquet")


def test_live_cli_selects_representation_without_changing_arm_settings(tmp_path):
    args = runner._parser().parse_args([
        "live", "--fixtures", str(tmp_path), "--output", str(tmp_path / "result.json"),
        "--representation", "lakehouse",
    ])
    assert args.representation == "lakehouse"
    assert args.repetitions == 3
    assert args.max_turns == 6
