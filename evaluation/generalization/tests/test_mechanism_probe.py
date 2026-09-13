from evaluation.generalization.fixtures import generate_fixtures
from evaluation.generalization.mechanism_probe import probe


def test_real_file_operation_restricts_sources_but_fallback_keeps_them(tmp_path):
    generate_fixtures(tmp_path, large_rows=20)
    result = probe(tmp_path)
    assert result["observed_restriction"]
    assert result["synthesis_aliases"] == ["knowledge_result"]
    assert result["packet"]["rows"] == [{"value": 310.0}]
    assert result["independent_reference"]["shipped"] == 262
    assert result["fallback_aliases"] == result["bound_aliases"]
