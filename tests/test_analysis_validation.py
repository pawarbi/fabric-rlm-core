from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from fabric_rlm.experimental import (
    SplitFold,
    SplitPlan,
    validate_split_plan,
)


def test_split_plan_is_immutable_serializable_and_fingerprinted() -> None:
    plan = SplitPlan(
        plan_id="grouped-cv",
        strategy="grouped",
        seed=42,
        row_ids=("r1", "r2", "r3", "r4"),
        folds=(
            SplitFold(
                fold_id="fold-1",
                train_ids=("r1", "r2"),
                validation_ids=("r3", "r4"),
            ),
        ),
    )

    assert plan.preprocessing_fit_scope == "training_fold_only"
    assert plan.to_dict()["folds"][0]["validation_ids"] == ["r3", "r4"]
    assert plan.fingerprint == SplitPlan(
        plan_id="grouped-cv",
        strategy="grouped",
        seed=42,
        row_ids=("r1", "r2", "r3", "r4"),
        folds=plan.folds,
    ).fingerprint
    with pytest.raises(FrozenInstanceError):
        plan.seed = 7


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "fold_id": "fold-1",
                "train_ids": ("r1", "r2"),
                "validation_ids": ("r2", "r3"),
            },
            "overlap",
        ),
        (
            {
                "fold_id": "fold-1",
                "train_ids": ("r1", "r1"),
                "validation_ids": ("r2",),
            },
            "duplicates",
        ),
        (
            {
                "fold_id": "fold-1",
                "train_ids": (),
                "validation_ids": ("r2",),
            },
            "train_ids",
        ),
    ],
)
def test_split_fold_rejects_invalid_partitions(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        SplitFold(**kwargs)


def test_split_plan_rejects_unknown_rows_and_holdout_reuse() -> None:
    with pytest.raises(ValueError, match="unknown row"):
        SplitPlan(
            plan_id="bad-row",
            strategy="random",
            seed=1,
            row_ids=("r1", "r2"),
            folds=(
                SplitFold(
                    fold_id="fold-1",
                    train_ids=("r1",),
                    validation_ids=("r3",),
                ),
            ),
        )

    with pytest.raises(ValueError, match="final_holdout_ids"):
        SplitPlan(
            plan_id="reused-holdout",
            strategy="random",
            seed=1,
            row_ids=("r1", "r2", "r3"),
            final_holdout_ids=("r3",),
            folds=(
                SplitFold(
                    fold_id="fold-1",
                    train_ids=("r1", "r3"),
                    validation_ids=("r2",),
                ),
            ),
        )


def test_nested_folds_must_remain_inside_parent_training_rows() -> None:
    outer = SplitFold(
        fold_id="outer-1",
        train_ids=("r1", "r2", "r3"),
        validation_ids=("r4",),
    )
    valid = SplitPlan(
        plan_id="nested-valid",
        strategy="nested",
        seed=1,
        row_ids=("r1", "r2", "r3", "r4"),
        folds=(
            outer,
            SplitFold(
                fold_id="outer-1-inner-1",
                parent_fold_id="outer-1",
                train_ids=("r1", "r2"),
                validation_ids=("r3",),
            ),
        ),
    )

    assert valid.folds[1].parent_fold_id == "outer-1"

    with pytest.raises(ValueError, match="escapes parent training"):
        SplitPlan(
            plan_id="nested-escape",
            strategy="nested",
            seed=1,
            row_ids=("r1", "r2", "r3", "r4"),
            folds=(
                outer,
                SplitFold(
                    fold_id="outer-1-inner-1",
                    parent_fold_id="outer-1",
                    train_ids=("r1", "r2"),
                    validation_ids=("r4",),
                ),
            ),
        )


def test_grouped_validation_rejects_entity_leakage() -> None:
    plan = SplitPlan(
        plan_id="grouped-cv",
        strategy="grouped",
        seed=42,
        row_ids=("r1", "r2", "r3", "r4"),
        folds=(
            SplitFold(
                fold_id="fold-1",
                train_ids=("r1", "r3"),
                validation_ids=("r2", "r4"),
            ),
        ),
    )

    with pytest.raises(ValueError, match="entity leakage"):
        validate_split_plan(
            plan,
            group_by_row={"r1": "g1", "r2": "g1", "r3": "g2", "r4": "g3"},
        )


def test_temporal_validation_rejects_future_leakage() -> None:
    plan = SplitPlan(
        plan_id="rolling",
        strategy="temporal",
        seed=0,
        row_ids=("r1", "r2", "r3", "r4"),
        folds=(
            SplitFold(
                fold_id="fold-1",
                train_ids=("r1", "r4"),
                validation_ids=("r2", "r3"),
            ),
        ),
    )

    with pytest.raises(ValueError, match="future leakage"):
        validate_split_plan(
            plan,
            time_by_row={"r1": 1, "r2": 2, "r3": 3, "r4": 4},
        )


def test_validation_rejects_duplicate_content_across_partitions() -> None:
    plan = SplitPlan(
        plan_id="duplicates",
        strategy="random",
        seed=5,
        row_ids=("r1", "r2", "r3"),
        folds=(
            SplitFold(
                fold_id="fold-1",
                train_ids=("r1", "r2"),
                validation_ids=("r3",),
            ),
        ),
    )

    with pytest.raises(ValueError, match="duplicate-row leakage"):
        validate_split_plan(
            plan,
            row_fingerprints={"r1": "a", "r2": "b", "r3": "a"},
        )


def test_validation_rejects_target_derived_features() -> None:
    plan = SplitPlan(
        plan_id="features",
        strategy="stratified",
        seed=5,
        row_ids=("r1", "r2"),
        folds=(
            SplitFold(
                fold_id="fold-1",
                train_ids=("r1",),
                validation_ids=("r2",),
            ),
        ),
        feature_columns=("amount", "future_converted_label"),
        prohibited_feature_columns=("future_converted_label",),
    )

    with pytest.raises(ValueError, match="target-derived"):
        validate_split_plan(plan)


def test_validated_split_plan_preserves_training_only_preprocessing_boundary() -> None:
    plan = SplitPlan(
        plan_id="safe-grouped",
        strategy="grouped",
        seed=42,
        row_ids=("r1", "r2", "r3", "r4"),
        folds=(
            SplitFold(
                fold_id="fold-1",
                train_ids=("r1", "r2"),
                validation_ids=("r3", "r4"),
            ),
        ),
        feature_columns=("amount",),
    )

    validated = validate_split_plan(
        plan,
        group_by_row={"r1": "g1", "r2": "g1", "r3": "g2", "r4": "g2"},
    )

    assert validated.plan is plan
    assert validated.fingerprint == plan.fingerprint
    assert validated.preprocessing_fit_ids("fold-1") == ("r1", "r2")
