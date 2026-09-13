"""Deterministic source-restriction reproduction; only planner text is mocked."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from unittest.mock import patch

from fabric_rlm import RLM

from .runner import _domain_sources, _write_json


def probe(fixtures: Path) -> dict:
    sources = _domain_sources(fixtures, "inventory", "descriptive")
    knowledge = RLM.learn(sources=sources)
    plan = {
        "operation_id": "order_lines.tabular.aggregate.v1",
        "parameters": {"aggregate": "sum", "measure": "ordered_units"},
    }
    rlm = RLM.from_task(
        "What share of ordered units has shipped?",
        outputs={"answer": dict}, knowledge=knowledge, lm=lambda **kwargs: [],
        skills=[], enable_skill_autoloading=False,
    )
    bound, metadata = rlm._bind_knowledge_inputs({})
    with patch("fabric_rlm.runtime._call_lm_with_meta",
               return_value=(json.dumps(plan), None, 0.0)):
        synthesis, execution = rlm._prepare_registered_operation(bound, metadata)
    with patch("fabric_rlm.runtime._call_lm_with_meta",
               return_value=('{"fallback":true,"reason":"ratio needs two sources"}', None, 0.0)):
        fallback, fallback_meta = rlm._prepare_registered_operation(bound, metadata)
    with open(sources["order_lines"], newline="") as stream:
        ordered = sum(int(row["ordered_units"]) for row in csv.DictReader(stream))
    with open(sources["shipment_events"], newline="") as stream:
        shipped = sum(int(row["shipped_units"]) for row in csv.DictReader(stream))
    return {
        "integration_scope": "real local CSV learn/preflight/host execution; scripted planner",
        "plan": plan, "bound_aliases": sorted(bound),
        "synthesis_aliases": sorted(synthesis),
        "packet": synthesis["knowledge_result"],
        "execution_mode": execution["knowledge_mode"],
        "fallback_aliases": sorted(fallback),
        "fallback_mode": fallback_meta["knowledge_mode"],
        "independent_reference": {"shipped": shipped, "ordered": ordered, "ratio": shipped / ordered},
        "lessons": len(knowledge.package.lessons),
        "observed_restriction": "shipment_events" not in synthesis and "shipment_events" in fallback,
        "proposed_fix": {
            "universal": "Require complete task coverage before one-operation synthesis; otherwise fall back or permit bounded additional operations.",
            "source_metadata": "Declare join keys, grain, and metric numerator/denominator.",
            "optional_skill": "Domain metric definitions only; no inventory-specific core branch.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.fixtures)
    _write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("observed_restriction", "lessons", "independent_reference")}))


if __name__ == "__main__":
    main()
