from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from fabric_rlm.experimental import (
    SplitFold,
    SplitPlan,
    build_grouped_split_plan,
    build_nested_grouped_split_plan,
    build_random_split_plan,
    build_stratified_split_plan,
    build_temporal_split_plan,
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


def test_validation_rejects_duplicate_content_in_final_holdout() -> None:
    with pytest.raises(ValueError, match="duplicate-row leakage"):
        build_random_split_plan(
            ("r1", "r2", "r3", "r4", "r5"),
            plan_id="holdout-duplicates",
            seed=5,
            fold_count=2,
            final_holdout_ids=("r5",),
            row_fingerprints={
                "r1": "duplicate",
                "r2": "b",
                "r3": "c",
                "r4": "d",
                "r5": "duplicate",
            },
        )


def test_grouped_builder_rejects_entity_split_into_final_holdout() -> None:
    with pytest.raises(ValueError, match="entity leakage"):
        build_grouped_split_plan(
            ("r1", "r2", "r3", "r4", "r5", "r6"),
            group_by_row={
                "r1": "g1",
                "r2": "g1",
                "r3": "g2",
                "r4": "g2",
                "r5": "g3",
                "r6": "g3",
            },
            plan_id="holdout-group-leakage",
            seed=5,
            fold_count=2,
            final_holdout_ids=("r6",),
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


def test_random_split_plan_is_deterministic_and_preserves_final_holdout() -> None:
    row_ids = tuple(f"r{index:02d}" for index in range(1, 26))
    first = build_random_split_plan(
        row_ids,
        plan_id="random-cv",
        seed=42,
        fold_count=5,
        final_holdout_ids=("r24", "r25"),
    )
    second = build_random_split_plan(
        tuple(reversed(row_ids)),
        plan_id="random-cv",
        seed=42,
        fold_count=5,
        final_holdout_ids=("r25", "r24"),
    )

    assert first.fingerprint == second.fingerprint
    assert first.plan.final_holdout_ids == ("r24", "r25")
    validation_ids = [
        row_id
        for fold in first.plan.folds
        for row_id in fold.validation_ids
    ]
    assert sorted(validation_ids) == list(row_ids[:-2])
    assert len(validation_ids) == len(set(validation_ids))
    assert all(
        {"r24", "r25"}.isdisjoint(fold.train_ids + fold.validation_ids)
        for fold in first.plan.folds
    )
    different = build_random_split_plan(
        row_ids,
        plan_id="random-cv",
        seed=43,
        fold_count=5,
        final_holdout_ids=("r24", "r25"),
    )
    assert first.fingerprint != different.fingerprint


def test_stratified_split_plan_preserves_each_class_in_every_fold() -> None:
    row_ids = tuple(f"r{index:02d}" for index in range(1, 21))
    strata = {
        row_id: ("positive" if index <= 8 else "negative")
        for index, row_id in enumerate(row_ids, start=1)
    }

    validated = build_stratified_split_plan(
        row_ids,
        strata_by_row=strata,
        plan_id="stratified-cv",
        seed=11,
        fold_count=4,
    )

    for fold in validated.plan.folds:
        validation_strata = [strata[row_id] for row_id in fold.validation_ids]
        assert validation_strata.count("positive") == 2
        assert validation_strata.count("negative") == 3


def test_grouped_split_plan_keeps_entities_in_one_validation_fold() -> None:
    row_ids = tuple(f"r{index:02d}" for index in range(1, 25))
    groups = {
        row_id: f"group-{(index - 1) // 3 + 1}"
        for index, row_id in enumerate(row_ids, start=1)
    }

    validated = build_grouped_split_plan(
        row_ids,
        group_by_row=groups,
        plan_id="grouped-cv",
        seed=19,
        fold_count=4,
    )

    validation_fold_by_group: dict[str, str] = {}
    for fold in validated.plan.folds:
        for row_id in fold.validation_ids:
            group = groups[row_id]
            prior_fold = validation_fold_by_group.setdefault(group, fold.fold_id)
            assert prior_fold == fold.fold_id
        assert {groups[row_id] for row_id in fold.train_ids}.isdisjoint(
            groups[row_id] for row_id in fold.validation_ids
        )
    assert set(validation_fold_by_group) == set(groups.values())


@pytest.mark.parametrize(
    ("builder", "kwargs", "match"),
    [
        (
            build_random_split_plan,
            {"row_ids": ("r1", "r2"), "fold_count": 3},
            "fold_count",
        ),
        (
            build_stratified_split_plan,
            {
                "row_ids": ("r1", "r2", "r3"),
                "strata_by_row": {"r1": "a", "r2": "a", "r3": "b"},
                "fold_count": 2,
            },
            "stratum",
        ),
        (
            build_grouped_split_plan,
            {
                "row_ids": ("r1", "r2"),
                "group_by_row": {"r1": "g1"},
                "fold_count": 2,
            },
            "group_by_row",
        ),
    ],
)
def test_split_plan_builders_reject_unsafe_inputs(
    builder: object,
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        builder(plan_id="invalid", seed=1, **kwargs)


def test_temporal_split_plan_builds_expanding_windows_without_future_leakage() -> None:
    row_ids = tuple(f"r{index:02d}" for index in range(1, 13))
    time_by_row = {
        row_id: index for index, row_id in enumerate(row_ids, start=1)
    }

    validated = build_temporal_split_plan(
        tuple(reversed(row_ids)),
        time_by_row=time_by_row,
        plan_id="expanding",
        seed=0,
        fold_count=3,
        initial_training_count=6,
    )

    assert [
        (fold.train_ids, fold.validation_ids)
        for fold in validated.plan.folds
    ] == [
        (row_ids[:6], row_ids[6:8]),
        (row_ids[:8], row_ids[8:10]),
        (row_ids[:10], row_ids[10:12]),
    ]
    assert validated.plan.strategy == "temporal"


def test_temporal_split_plan_preserves_strictly_later_final_holdout() -> None:
    row_ids = tuple(f"r{index:02d}" for index in range(1, 11))
    time_by_row = {
        row_id: index for index, row_id in enumerate(row_ids, start=1)
    }

    validated = build_temporal_split_plan(
        row_ids,
        time_by_row=time_by_row,
        plan_id="temporal-holdout",
        seed=0,
        fold_count=2,
        initial_training_count=4,
        final_holdout_ids=("r09", "r10"),
    )

    assert validated.plan.final_holdout_ids == ("r09", "r10")
    assert all(
        {"r09", "r10"}.isdisjoint(fold.train_ids + fold.validation_ids)
        for fold in validated.plan.folds
    )


def test_temporal_split_plan_rejects_tied_boundary() -> None:
    with pytest.raises(ValueError, match="time boundary"):
        build_temporal_split_plan(
            ("r1", "r2", "r3", "r4", "r5", "r6"),
            time_by_row={
                "r1": 1,
                "r2": 2,
                "r3": 2,
                "r4": 3,
                "r5": 4,
                "r6": 5,
            },
            plan_id="tied-boundary",
            seed=0,
            fold_count=2,
            initial_training_count=2,
        )


def test_temporal_split_plan_rejects_holdout_future_leakage() -> None:
    with pytest.raises(ValueError, match="final holdout"):
        build_temporal_split_plan(
            ("r1", "r2", "r3", "r4", "r5", "r6"),
            time_by_row={
                "r1": 1,
                "r2": 2,
                "r3": 3,
                "r4": 4,
                "r5": 5,
                "r6": 6,
            },
            plan_id="early-holdout",
            seed=0,
            fold_count=2,
            initial_training_count=1,
            final_holdout_ids=("r3",),
        )


def test_nested_grouped_split_plan_keeps_inner_folds_inside_outer_training() -> None:
    row_ids = tuple(f"r{index:02d}" for index in range(1, 37))
    groups = {
        row_id: f"group-{(index - 1) // 3 + 1:02d}"
        for index, row_id in enumerate(row_ids, start=1)
    }

    validated = build_nested_grouped_split_plan(
        row_ids,
        group_by_row=groups,
        plan_id="nested-grouped",
        seed=29,
        outer_fold_count=3,
        inner_fold_count=2,
    )

    outer_folds = [
        fold for fold in validated.plan.folds if fold.parent_fold_id is None
    ]
    inner_folds = [
        fold for fold in validated.plan.folds if fold.parent_fold_id is not None
    ]
    assert len(outer_folds) == 3
    assert len(inner_folds) == 6
    outer_by_id = {fold.fold_id: fold for fold in outer_folds}
    for inner in inner_folds:
        assert set(inner.train_ids + inner.validation_ids) == set(
            outer_by_id[inner.parent_fold_id].train_ids
        )
        assert {
            groups[row_id] for row_id in inner.train_ids
        }.isdisjoint(groups[row_id] for row_id in inner.validation_ids)


def test_nested_grouped_split_plan_is_deterministic_and_uses_distinct_seeds() -> None:
    row_ids = tuple(f"r{index:02d}" for index in range(1, 25))
    groups = {
        row_id: f"group-{(index - 1) // 2 + 1:02d}"
        for index, row_id in enumerate(row_ids, start=1)
    }
    first = build_nested_grouped_split_plan(
        row_ids,
        group_by_row=groups,
        plan_id="nested-grouped",
        seed=31,
        outer_fold_count=3,
        inner_fold_count=2,
    )
    second = build_nested_grouped_split_plan(
        tuple(reversed(row_ids)),
        group_by_row=groups,
        plan_id="nested-grouped",
        seed=31,
        outer_fold_count=3,
        inner_fold_count=2,
    )

    assert first.fingerprint == second.fingerprint
    assert first.plan.strategy == "nested"
    assert len({fold.fold_id for fold in first.plan.folds}) == 9
