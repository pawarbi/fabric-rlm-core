from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    not Path("/lakehouse/default/Files").exists(),
    reason="Fabric default Lakehouse is not attached in this environment",
)


def test_fabric_lakehouse_mount_is_available() -> None:
    files_root = Path("/lakehouse/default/Files")

    assert files_root.exists()
    assert files_root.is_dir()


def test_fabric_validation_output_root_can_be_created() -> None:
    output_root = Path("/lakehouse/default/Files/fabric_rlm_validation/pytest_smoke")
    output_root.mkdir(parents=True, exist_ok=True)
    marker = output_root / "marker.txt"
    marker.write_text("ok", encoding="utf-8")

    assert marker.read_text(encoding="utf-8") == "ok"

