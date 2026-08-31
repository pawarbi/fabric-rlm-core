"""Execute one source-aware evidence-closure cycle from critic challenges."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from fabric_rlm._benchmark_manifest import load_source_manifest


def _load_example(name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(f"_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_STAGED = _load_example("olist_staged_deep_insight_benchmark.py")
_CRITIC = _load_example("olist_deep_insight_critic.py")


def resolve_closure_audit(
    record: dict[str, Any], original_audit: dict[str, Any]
) -> dict[str, Any]:
    """Preserve the attested input audit when critic closure has no targets."""

    closure_audit = record.get("audit")
    if closure_audit is None:
        if record.get("summary", {}).get("skipped") is not True:
            raise ValueError("critic closure returned no audit without skipping")
        return original_audit
    return _STAGED.audit_to_dict(closure_audit)


def run_critic_closure(
    manifest_path: str | Path,
    payload_path: str | Path,
    audit_path: str | Path,
    critic_path: str | Path,
    *,
    output_dir: str | Path,
    model: str = _STAGED.DEFAULT_MODEL,
    api_base: str = _STAGED.DEFAULT_API_BASE,
    max_turns: int = 14,
    timeout: float = 3600,
) -> dict[str, Any]:
    """Bind critic inputs, execute closure, and persist verified artifacts."""

    manifest = load_source_manifest(manifest_path)
    payload = _CRITIC.load_json(payload_path, "discovery payload")
    audit = _CRITIC.load_json(audit_path, "audit artifact")
    critic = _CRITIC.load_json(critic_path, "critic artifact")
    _CRITIC.validate_discovery(payload)
    _CRITIC.validate_audit(audit)
    expected_fingerprint = _CRITIC.source_fingerprint(payload, audit)
    if critic.get("source_fingerprint") != expected_fingerprint:
        raise ValueError("critic artifact does not match discovery and audit inputs")
    _CRITIC.verify_critic(critic)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required to run critic closure")
    (
        dspy,
        RLM,
        DuckDBAuditExecutor,
        audit_deep_insight,
        summarize_trajectory,
        _,
    ) = _STAGED._load_runtime_dependencies()
    lm = dspy.LM(
        model,
        api_key=api_key,
        api_base=api_base,
        max_tokens=20000,
        temperature=0,
        cache=False,
        reasoning={"max_tokens": 4096, "exclude": True},
    )
    dspy.configure(lm=lm)

    output = Path(output_dir)
    record = _STAGED.run_critic_evidence_closure(
        payload,
        critic,
        dict(manifest.sources),
        lm=lm,
        rlm_type=RLM,
        executor_type=DuckDBAuditExecutor,
        audit_function=audit_deep_insight,
        summarize_trajectory=summarize_trajectory,
        verify_function=_STAGED.verify_portable_contract,
        max_turns=max_turns,
        timeout=timeout,
        checkpoint_path=output / "critic_closure.checkpoint.json",
    )
    paths = {
        "payload": output / "payload.json",
        "audit": output / "audit.json",
        "run": output / "run.json",
    }
    _STAGED._atomic_json(paths["payload"], record["payload"])
    persisted_audit = resolve_closure_audit(record, audit)
    _STAGED._atomic_json(paths["audit"], persisted_audit)
    _STAGED._atomic_json(
        paths["run"],
        {
            "status": "success",
            "model": model,
            "summary": record["summary"],
            "counts": {
                "insights": len(record["payload"]["insights"]),
                "audit_checks": len(persisted_audit["checks"]),
            },
        },
    )
    return {"record": record, "paths": paths}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Close measurable gated critic challenges with host audit."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--critic", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default=_STAGED.DEFAULT_MODEL)
    parser.add_argument("--api-base", default=_STAGED.DEFAULT_API_BASE)
    parser.add_argument("--turns", default=14, type=int)
    parser.add_argument("--timeout", default=3600, type=float)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_critic_closure(
        args.manifest,
        args.payload,
        args.audit,
        args.critic,
        output_dir=args.output_dir,
        model=args.model,
        api_base=args.api_base,
        max_turns=args.turns,
        timeout=args.timeout,
    )
    print(
        json.dumps(
            {
                "status": "success",
                "audit_checks": len(
                    json.loads(
                        result["paths"]["audit"].read_text(encoding="utf-8")
                    )["checks"]
                ),
                "artifacts": {
                    name: str(path) for name, path in result["paths"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
