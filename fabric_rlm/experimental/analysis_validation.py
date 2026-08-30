"""Immutable validation plans and leakage checks for analysis benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import random
from typing import Literal

from fabric_rlm.experimental.analysis_reproducibility import fingerprint


SplitStrategy = Literal["random", "stratified", "grouped", "temporal", "nested"]
_SPLIT_STRATEGIES = {"random", "stratified", "grouped", "temporal", "nested"}


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _unique_text_tuple(
    values: object,
    field_name: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field_name} must be a sequence of strings")
    normalized = tuple(
        _required_text(value, f"{field_name}[{index}]")
        for index, value in enumerate(values)
    )
    if not allow_empty and not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates")
    return normalized


@dataclass(frozen=True)
class SplitFold:
    """One train/validation partition in a persisted validation plan."""

    fold_id: str
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    parent_fold_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fold_id", _required_text(self.fold_id, "fold_id"))
        object.__setattr__(
            self,
            "train_ids",
            _unique_text_tuple(self.train_ids, "train_ids", allow_empty=False),
        )
        object.__setattr__(
            self,
            "validation_ids",
            _unique_text_tuple(
                self.validation_ids,
                "validation_ids",
                allow_empty=False,
            ),
        )
        overlap = sorted(set(self.train_ids) & set(self.validation_ids))
        if overlap:
            raise ValueError(
                f"train_ids and validation_ids overlap: {overlap[0]}"
            )
        if self.parent_fold_id is not None:
            object.__setattr__(
                self,
                "parent_fold_id",
                _required_text(self.parent_fold_id, "parent_fold_id"),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "train_ids": list(self.train_ids),
            "validation_ids": list(self.validation_ids),
            "parent_fold_id": self.parent_fold_id,
        }


@dataclass(frozen=True)
class SplitPlan:
    """Persisted split assignments and preprocessing boundaries."""

    plan_id: str
    strategy: SplitStrategy
    seed: int
    row_ids: tuple[str, ...]
    folds: tuple[SplitFold, ...]
    final_holdout_ids: tuple[str, ...] = ()
    feature_columns: tuple[str, ...] = ()
    prohibited_feature_columns: tuple[str, ...] = ()
    preprocessing_fit_scope: str = "training_fold_only"

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _required_text(self.plan_id, "plan_id"))
        if self.strategy not in _SPLIT_STRATEGIES:
            raise ValueError(
                "strategy must be random, stratified, grouped, temporal, or nested"
            )
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        object.__setattr__(
            self,
            "row_ids",
            _unique_text_tuple(self.row_ids, "row_ids", allow_empty=False),
        )
        if not isinstance(self.folds, (list, tuple)) or not self.folds:
            raise ValueError("folds must be a non-empty sequence")
        normalized_folds = tuple(self.folds)
        if any(not isinstance(fold, SplitFold) for fold in normalized_folds):
            raise ValueError("folds must contain SplitFold values")
        fold_ids = tuple(fold.fold_id for fold in normalized_folds)
        if len(set(fold_ids)) != len(fold_ids):
            raise ValueError("folds must not contain duplicate fold_id values")
        object.__setattr__(self, "folds", normalized_folds)
        for field_name in (
            "final_holdout_ids",
            "feature_columns",
            "prohibited_feature_columns",
        ):
            object.__setattr__(
                self,
                field_name,
                _unique_text_tuple(getattr(self, field_name), field_name),
            )
        if self.preprocessing_fit_scope != "training_fold_only":
            raise ValueError(
                "preprocessing_fit_scope must be training_fold_only"
            )

        known_rows = set(self.row_ids)
        holdout = set(self.final_holdout_ids)
        unknown_holdout = sorted(holdout - known_rows)
        if unknown_holdout:
            raise ValueError(
                f"final_holdout_ids contains unknown row: {unknown_holdout[0]}"
            )
        for fold in self.folds:
            assigned = set(fold.train_ids) | set(fold.validation_ids)
            unknown_rows = sorted(assigned - known_rows)
            if unknown_rows:
                raise ValueError(
                    f"fold {fold.fold_id} contains unknown row: {unknown_rows[0]}"
                )
            reused_holdout = sorted(assigned & holdout)
            if reused_holdout:
                raise ValueError(
                    "final_holdout_ids must not be reused in folds: "
                    f"{reused_holdout[0]}"
                )
        fold_by_id = {fold.fold_id: fold for fold in self.folds}
        for fold in self.folds:
            if fold.parent_fold_id is None:
                continue
            parent = fold_by_id.get(fold.parent_fold_id)
            if parent is None:
                raise ValueError(
                    f"unknown parent_fold_id: {fold.parent_fold_id}"
                )
            parent_training = set(parent.train_ids)
            child_rows = set(fold.train_ids) | set(fold.validation_ids)
            if not child_rows <= parent_training:
                raise ValueError(
                    f"nested fold {fold.fold_id} escapes parent training rows"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "strategy": self.strategy,
            "seed": self.seed,
            "row_ids": list(self.row_ids),
            "folds": [fold.to_dict() for fold in self.folds],
            "final_holdout_ids": list(self.final_holdout_ids),
            "feature_columns": list(self.feature_columns),
            "prohibited_feature_columns": list(
                self.prohibited_feature_columns
            ),
            "preprocessing_fit_scope": self.preprocessing_fit_scope,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())


@dataclass(frozen=True)
class ValidatedSplitPlan:
    """A split plan that passed structural and leakage checks."""

    plan: SplitPlan
    fingerprint: str

    def preprocessing_fit_ids(self, fold_id: str) -> tuple[str, ...]:
        normalized = _required_text(fold_id, "fold_id")
        for fold in self.plan.folds:
            if fold.fold_id == normalized:
                return fold.train_ids
        raise KeyError(f"unknown fold_id: {normalized}")


def _require_complete_mapping(
    values: Mapping[str, object],
    row_ids: tuple[str, ...],
    field_name: str,
) -> None:
    missing = sorted(set(row_ids) - set(values))
    if missing:
        raise ValueError(f"{field_name} is missing row: {missing[0]}")


def _builder_inputs(
    *,
    row_ids: tuple[str, ...] | list[str],
    final_holdout_ids: tuple[str, ...] | list[str],
    fold_count: int,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    normalized_rows = tuple(
        sorted(_unique_text_tuple(row_ids, "row_ids", allow_empty=False))
    )
    normalized_holdout = tuple(
        sorted(_unique_text_tuple(final_holdout_ids, "final_holdout_ids"))
    )
    unknown_holdout = sorted(set(normalized_holdout) - set(normalized_rows))
    if unknown_holdout:
        raise ValueError(
            f"final_holdout_ids contains unknown row: {unknown_holdout[0]}"
        )
    if type(fold_count) is not int or fold_count < 2:
        raise ValueError("fold_count must be an integer of at least 2")
    holdout = set(normalized_holdout)
    eligible_rows = tuple(
        row_id for row_id in normalized_rows if row_id not in holdout
    )
    if fold_count > len(eligible_rows):
        raise ValueError("fold_count must not exceed eligible row count")
    return normalized_rows, normalized_holdout, eligible_rows


def _folds_from_validation_buckets(
    eligible_rows: tuple[str, ...],
    validation_buckets: list[list[str]],
) -> tuple[SplitFold, ...]:
    eligible = set(eligible_rows)
    return tuple(
        SplitFold(
            fold_id=f"fold-{index + 1}",
            train_ids=tuple(sorted(eligible - set(validation_ids))),
            validation_ids=tuple(sorted(validation_ids)),
        )
        for index, validation_ids in enumerate(validation_buckets)
    )


def build_random_split_plan(
    row_ids: tuple[str, ...] | list[str],
    *,
    plan_id: str,
    seed: int,
    fold_count: int = 5,
    final_holdout_ids: tuple[str, ...] | list[str] = (),
    feature_columns: tuple[str, ...] | list[str] = (),
    prohibited_feature_columns: tuple[str, ...] | list[str] = (),
    row_fingerprints: Mapping[str, str] | None = None,
) -> ValidatedSplitPlan:
    """Build deterministic random K-fold assignments."""

    normalized_rows, normalized_holdout, eligible_rows = _builder_inputs(
        row_ids=row_ids,
        final_holdout_ids=final_holdout_ids,
        fold_count=fold_count,
    )
    shuffled = list(eligible_rows)
    random.Random(seed).shuffle(shuffled)
    buckets = [[] for _ in range(fold_count)]
    for index, row_id in enumerate(shuffled):
        buckets[index % fold_count].append(row_id)
    plan = SplitPlan(
        plan_id=plan_id,
        strategy="random",
        seed=seed,
        row_ids=normalized_rows,
        folds=_folds_from_validation_buckets(eligible_rows, buckets),
        final_holdout_ids=normalized_holdout,
        feature_columns=tuple(feature_columns),
        prohibited_feature_columns=tuple(prohibited_feature_columns),
    )
    return validate_split_plan(plan, row_fingerprints=row_fingerprints)


def build_stratified_split_plan(
    row_ids: tuple[str, ...] | list[str],
    *,
    strata_by_row: Mapping[str, object],
    plan_id: str,
    seed: int,
    fold_count: int = 5,
    final_holdout_ids: tuple[str, ...] | list[str] = (),
    feature_columns: tuple[str, ...] | list[str] = (),
    prohibited_feature_columns: tuple[str, ...] | list[str] = (),
    row_fingerprints: Mapping[str, str] | None = None,
) -> ValidatedSplitPlan:
    """Build deterministic K-fold assignments balanced within each stratum."""

    normalized_rows, normalized_holdout, eligible_rows = _builder_inputs(
        row_ids=row_ids,
        final_holdout_ids=final_holdout_ids,
        fold_count=fold_count,
    )
    _require_complete_mapping(strata_by_row, normalized_rows, "strata_by_row")
    strata: dict[object, list[str]] = {}
    for row_id in eligible_rows:
        value = strata_by_row[row_id]
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError("strata_by_row values must be hashable") from exc
        strata.setdefault(value, []).append(row_id)
    undersized = [
        value for value, members in strata.items() if len(members) < fold_count
    ]
    if undersized:
        raise ValueError(
            f"stratum {undersized[0]!r} has fewer rows than fold_count"
        )
    rng = random.Random(seed)
    buckets = [[] for _ in range(fold_count)]
    for members in strata.values():
        rng.shuffle(members)
        for index, row_id in enumerate(members):
            buckets[index % fold_count].append(row_id)
    plan = SplitPlan(
        plan_id=plan_id,
        strategy="stratified",
        seed=seed,
        row_ids=normalized_rows,
        folds=_folds_from_validation_buckets(eligible_rows, buckets),
        final_holdout_ids=normalized_holdout,
        feature_columns=tuple(feature_columns),
        prohibited_feature_columns=tuple(prohibited_feature_columns),
    )
    return validate_split_plan(plan, row_fingerprints=row_fingerprints)


def build_grouped_split_plan(
    row_ids: tuple[str, ...] | list[str],
    *,
    group_by_row: Mapping[str, object],
    plan_id: str,
    seed: int,
    fold_count: int = 5,
    final_holdout_ids: tuple[str, ...] | list[str] = (),
    feature_columns: tuple[str, ...] | list[str] = (),
    prohibited_feature_columns: tuple[str, ...] | list[str] = (),
    row_fingerprints: Mapping[str, str] | None = None,
) -> ValidatedSplitPlan:
    """Build deterministic K-fold assignments without splitting entities."""

    normalized_rows, normalized_holdout, eligible_rows = _builder_inputs(
        row_ids=row_ids,
        final_holdout_ids=final_holdout_ids,
        fold_count=fold_count,
    )
    _require_complete_mapping(group_by_row, normalized_rows, "group_by_row")
    groups: dict[object, list[str]] = {}
    for row_id in eligible_rows:
        value = group_by_row[row_id]
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError("group_by_row values must be hashable") from exc
        groups.setdefault(value, []).append(row_id)
    if len(groups) < fold_count:
        raise ValueError("fold_count must not exceed eligible group count")
    group_values = list(groups)
    random.Random(seed).shuffle(group_values)
    buckets = [[] for _ in range(fold_count)]
    for index, group in enumerate(group_values):
        buckets[index % fold_count].extend(groups[group])
    plan = SplitPlan(
        plan_id=plan_id,
        strategy="grouped",
        seed=seed,
        row_ids=normalized_rows,
        folds=_folds_from_validation_buckets(eligible_rows, buckets),
        final_holdout_ids=normalized_holdout,
        feature_columns=tuple(feature_columns),
        prohibited_feature_columns=tuple(prohibited_feature_columns),
    )
    return validate_split_plan(
        plan,
        group_by_row=group_by_row,
        row_fingerprints=row_fingerprints,
    )


def validate_split_plan(
    plan: SplitPlan,
    *,
    group_by_row: Mapping[str, object] | None = None,
    time_by_row: Mapping[str, object] | None = None,
    row_fingerprints: Mapping[str, str] | None = None,
) -> ValidatedSplitPlan:
    """Reject entity, future, duplicate-row, and target-derived leakage."""

    if not isinstance(plan, SplitPlan):
        raise TypeError("plan must be a SplitPlan")
    prohibited = set(plan.prohibited_feature_columns)
    leaked_features = sorted(set(plan.feature_columns) & prohibited)
    if leaked_features:
        raise ValueError(
            f"target-derived feature is prohibited: {leaked_features[0]}"
        )

    if plan.strategy in {"grouped", "nested"}:
        if group_by_row is None:
            raise ValueError(f"{plan.strategy} strategy requires group_by_row")
        _require_complete_mapping(group_by_row, plan.row_ids, "group_by_row")
    if plan.strategy == "temporal":
        if time_by_row is None:
            raise ValueError("temporal strategy requires time_by_row")
        _require_complete_mapping(time_by_row, plan.row_ids, "time_by_row")
    if row_fingerprints is not None:
        _require_complete_mapping(
            row_fingerprints,
            plan.row_ids,
            "row_fingerprints",
        )

    holdout_groups = (
        {group_by_row[row_id] for row_id in plan.final_holdout_ids}
        if group_by_row is not None
        else set()
    )
    holdout_fingerprints = (
        {row_fingerprints[row_id] for row_id in plan.final_holdout_ids}
        if row_fingerprints is not None
        else set()
    )
    for fold in plan.folds:
        if group_by_row is not None:
            training_groups = {group_by_row[row_id] for row_id in fold.train_ids}
            validation_groups = {
                group_by_row[row_id] for row_id in fold.validation_ids
            }
            overlap = training_groups & validation_groups
            if overlap:
                raise ValueError(
                    f"entity leakage in fold {fold.fold_id}: "
                    f"{next(iter(overlap))!r}"
                )
            fold_groups = training_groups | validation_groups
            if fold_groups & holdout_groups:
                raise ValueError(
                    f"entity leakage between fold {fold.fold_id} "
                    "and final holdout"
                )
        if time_by_row is not None:
            latest_training = max(time_by_row[row_id] for row_id in fold.train_ids)
            earliest_validation = min(
                time_by_row[row_id] for row_id in fold.validation_ids
            )
            if latest_training >= earliest_validation:
                raise ValueError(
                    f"future leakage in fold {fold.fold_id}: training reaches "
                    "or passes validation time"
                )
        if row_fingerprints is not None:
            training_fingerprints = {
                row_fingerprints[row_id] for row_id in fold.train_ids
            }
            validation_fingerprints = {
                row_fingerprints[row_id] for row_id in fold.validation_ids
            }
            overlap = training_fingerprints & validation_fingerprints
            if overlap:
                raise ValueError(
                    f"duplicate-row leakage in fold {fold.fold_id}"
                )
            fold_fingerprints = (
                training_fingerprints | validation_fingerprints
            )
            if fold_fingerprints & holdout_fingerprints:
                raise ValueError(
                    f"duplicate-row leakage between fold {fold.fold_id} "
                    "and final holdout"
                )

    return ValidatedSplitPlan(plan=plan, fingerprint=plan.fingerprint)
