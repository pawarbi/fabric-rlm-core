"""Engine consolidation — parity / baseline / regression tests.

These tests guard the engine consolidation work described in
``~/.copilot/session-state/.../files/engine_consolidation_plan.md``. They lock
the post-init state of each engine variant so the consolidation work cannot
silently change observable behavior for users who construct ``RLM(...)``
without specifying ``engine=``.

**The fingerprint contract.** Each engine variant must produce a deterministic
``_fingerprint`` dict capturing post-``__init__`` state. Aliases must produce
identical fingerprints to their canonical names. The default constructor
(no ``engine=`` kwarg, no ``tools=``) must produce the v6-custom fingerprint.

**No live LM calls.** All construction uses ``_StubLM`` so tests are
deterministic, fast (<1s), and run offline.
"""

from __future__ import annotations

from typing import Any

import dspy
import pytest

from fabric_rlm import RLM


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubLM(dspy.LM):
    """A no-op dspy.LM. Only used so RLM construction can resolve a backend.
    Never called — these tests stop at __init__ time."""

    def __init__(self) -> None:
        super().__init__(model="stub", model_type="chat")

    def __call__(self, *args: Any, **kwargs: Any) -> list[str]:  # pragma: no cover
        raise AssertionError(
            "_StubLM should not be invoked; parity tests are __init__-only"
        )


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------


# Attributes that define the engine **routing decision** at __init__ time.
# This fingerprint is intentionally narrow — its job is to prove
# alias↔canonical equivalence, NOT full runtime-state equivalence.
# (See duck review of commit 9bce342: a wider fingerprint adds surface area
# without catching anything the routing-only fix can plausibly break.)
_FINGERPRINT_ATTRS: tuple[str, ...] = (
    "engine",
    "inner_engine",
    "skills",
    "max_turns",
    "timeout",
    "enable_verifier",
    "enable_skill_autoloading",
    "enable_router",
    "halve_max_iter_on_retry",
    "stuck_loop_threshold",
    "max_active_skills",
    "reserve_finalize_turns",
    "max_prompt_tokens",
    "digest_after_turn",
)


def _fingerprint(rlm: RLM) -> dict[str, Any]:
    """Capture the post-init **routing-relevant** state of an ``RLM`` instance.

    This is the parity oracle for engine alias resolution: two RLMs that
    differ only in the user-supplied alias name (e.g., ``"v6-custom"`` vs
    ``"default"``) must produce identical fingerprints.

    NOT a full runtime-state oracle. Attributes like ``sub_lm_spec``,
    ``output_validator``, router config lists are trivially preserved by
    Phase 1's name-normalization-only change and are not fingerprinted here.
    If a future phase changes more than name normalization, widen this list.
    """
    fp: dict[str, Any] = {}
    for attr in _FINGERPRINT_ATTRS:
        value = getattr(rlm, attr, None)
        if isinstance(value, (list, tuple)):
            fp[attr] = list(value)
        elif isinstance(value, set):
            fp[attr] = sorted(value)
        else:
            fp[attr] = value

    # Defensive: adaptive RLM may not set self.tools (early return path).
    tools = getattr(rlm, "tools", None)
    fp["tools_count"] = len(list(tools)) if tools else 0
    # Order matters — skill verifier order is observable in _run_skill_verifiers.
    fp["loaded_skill_names"] = [s.name for s in rlm._loaded_skills]
    fp["has_inline_task"] = rlm._inline_task is not None
    fp["has_signature"] = rlm.signature is not None
    return fp


# ---------------------------------------------------------------------------
# Phase 0 — baseline lock: capture the current default behavior
# ---------------------------------------------------------------------------


def test_baseline_default_constructor_uses_v6_custom():
    """Locks the current default. If this fails after a change, the default
    engine has shifted — that's the regression we are guarding against."""
    rlm = RLM(lm=_StubLM())
    assert rlm.engine == "v6-custom", (
        f"Default engine changed: expected 'v6-custom', got {rlm.engine!r}. "
        "If this is intentional (Phase 4 default flip), update this test."
    )
    assert rlm.inner_engine == "v6-custom"


def test_baseline_v7_dspy_engine_is_addressable():
    """Locks the current explicit-opt-in shape for v7-dspy."""
    rlm = RLM(lm=_StubLM(), engine="v7-dspy")
    assert rlm.engine == "v7-dspy"
    assert rlm.inner_engine == "v7-dspy"


def test_baseline_v6_custom_explicit_matches_default():
    """Constructing with explicit ``engine='v6-custom'`` must produce a
    fingerprint identical to the default constructor. This is the strongest
    guard against accidental Phase 4 regressions."""
    rlm_default = RLM(lm=_StubLM())
    rlm_explicit = RLM(lm=_StubLM(), engine="v6-custom")
    assert _fingerprint(rlm_default) == _fingerprint(rlm_explicit)


def test_baseline_v7_dspy_with_tools_accepts_kwarg():
    """Locks the PR #18 contract: tools= is accepted on v7-dspy."""

    def my_tool(x: int) -> int:
        return x + 1

    rlm = RLM(lm=_StubLM(), engine="v7-dspy", tools=[my_tool])
    assert rlm.engine == "v7-dspy"
    assert len(list(rlm.tools)) == 1


# ---------------------------------------------------------------------------
# Phase 1 — alias parity (xfail until aliases are implemented in Phase 1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("default", "v6-custom"),
        ("dspy", "v7-dspy"),
    ],
)
def test_alias_resolves_to_canonical_engine(alias: str, canonical: str):
    """``engine=alias`` must resolve to the canonical engine name and produce
    a fingerprint identical to ``engine=canonical``."""
    rlm_alias = RLM(lm=_StubLM(), engine=alias)
    rlm_canonical = RLM(lm=_StubLM(), engine=canonical)
    assert rlm_alias.engine == canonical, (
        f"Alias {alias!r} did not resolve to canonical {canonical!r}; "
        f"got {rlm_alias.engine!r}"
    )
    assert _fingerprint(rlm_alias) == _fingerprint(rlm_canonical)


def test_alias_works_for_inner_engine_in_adaptive():
    """Adaptive inner_engine must accept aliases too."""
    rlm = RLM(lm=_StubLM(), engine="adaptive", inner_engine="default")
    assert rlm.engine == "adaptive"
    assert rlm.inner_engine == "v6-custom"


def test_non_adaptive_ignores_arbitrary_inner_engine():
    """Pre-Phase-1 behavior: when ``engine != "adaptive"``, the
    ``inner_engine`` kwarg is silently ignored. Config-driven callers that
    always set both must keep working — Phase 1 must NOT add new strictness
    on a value that goes unused. (Per duck review #1 of Phase 1 diff.)"""
    rlm = RLM(lm=_StubLM(), engine="v6-custom", inner_engine="not_a_real_engine")
    # inner_engine is overwritten with the resolved canonical engine for
    # non-adaptive paths. Confirm no exception was raised AND the stored
    # value reflects the actual engine, not the ignored input.
    assert rlm.engine == "v6-custom"
    assert rlm.inner_engine == "v6-custom"


def test_adaptive_factory_passes_public_alias_to_avoid_self_warn_post_phase5():
    """Phase 5: when the outer is constructed with ``inner_engine="default"``
    (or any spelling that resolves to ``"v6-custom"``), the closure that
    builds inner attempts in ``_run_adaptive`` must pass the **public alias**
    (``"default"`` / ``"dspy"``) to the inner ``RLM(**kwargs)`` call — NOT
    the canonical legacy literal — so the inner construction does not
    self-emit a Phase 5 ``DeprecationWarning``.

    End-state ``self.engine`` is identical because ``_normalize_engine_name``
    resolves the alias back to the same canonical name.

    This supersedes the pre-Phase-5 contract that the factory propagated
    the canonical name verbatim — Phase 5 added the deprecation warning
    layer that made the verbatim canonical pass-through a self-warning bug
    (caught by duck review of Phase 5 diff). End behavior is unchanged.

    We don't run the adaptive loop (which requires real LM calls). Instead
    we monkey-patch ``AdaptiveRunner`` to capture the factory the moment
    ``_run_adaptive`` constructs it, then invoke the factory with a
    minimal AttemptConfig and inspect the kwargs it would pass to
    ``RLM(**kwargs)``."""
    from fabric_rlm.experimental.adaptive_policy import AttemptConfig
    import fabric_rlm.runtime as rt

    rlm = RLM(
        lm=_StubLM(),
        engine="adaptive",
        inner_engine="default",
        adaptive={"validator": lambda **_: None},
    )

    captured_factory: list = []

    class _FakeRunner:
        def __init__(self, **kwargs):
            captured_factory.append(kwargs.get("rlm_factory"))

        def run(self, *args, **kwargs):
            return None

    captured_rlm_kwargs: list[dict] = []

    def fake_rlm(**kwargs):
        captured_rlm_kwargs.append(kwargs)

        class _Stub:
            pass

        return _Stub()

    from fabric_rlm.experimental import adaptive_runner as ar_module
    import fabric_rlm.runtime as rt

    original_runner = ar_module.AdaptiveRunner
    original_rlm = rt.RLM
    try:
        ar_module.AdaptiveRunner = _FakeRunner  # type: ignore[assignment]
        rt.RLM = fake_rlm  # type: ignore[assignment]
        try:
            rlm._run_adaptive({"prompt": "test"})
        except Exception:
            pass

        assert captured_factory, "_run_adaptive did not build a factory"
        factory = captured_factory[0]
        cfg = AttemptConfig(rung=0, max_turns=1)
        factory(cfg)
    finally:
        ar_module.AdaptiveRunner = original_runner
        rt.RLM = original_rlm

    assert captured_rlm_kwargs, "factory did not invoke RLM constructor"
    passed_engine = captured_rlm_kwargs[0].get("engine")
    assert passed_engine == "default", (
        f"Phase 5 factory must pass public alias 'default' (was 'v6-custom' "
        f"pre-Phase-5); end behavior is identical because alias resolves to "
        f"the same canonical name. Got: {passed_engine!r}"
    )



def test_unknown_engine_name_raises_with_helpful_message():
    """An unknown engine name must raise ValueError listing valid names.

    Note: ``"auto"`` is mentioned only after Phase 2 ships. Phase 1's helpful
    message lists ``v6-custom``, ``v7-dspy``, ``adaptive``, ``default``, ``dspy``."""
    with pytest.raises(ValueError) as exc_info:
        RLM(lm=_StubLM(), engine="not_a_real_engine")
    msg = str(exc_info.value)
    for valid in ("v6-custom", "v7-dspy", "adaptive", "default", "dspy"):
        assert valid in msg, f"Error message should mention {valid!r}: got {msg!r}"
    # After Phase 2 ships the capability router, "auto" is a valid input
    # and SHOULD appear in the helpful message.
    assert "auto" in msg, (
        f"'auto' should appear in unknown-engine message after Phase 2: {msg!r}"
    )


@pytest.mark.parametrize("bad_engine", [None, 123, 3.14, object()])
def test_non_string_engine_raises_value_error(bad_engine):
    """Non-string engine values must raise ``ValueError`` (not ``TypeError``).

    Pre-Phase-1 callers handled ``ValueError`` for invalid engine values;
    silently changing to ``TypeError`` would break their except clauses.
    """
    with pytest.raises(ValueError, match="engine must be one of"):
        RLM(lm=_StubLM(), engine=bad_engine)


def test_alias_dspy_accepts_tools_kwarg():
    """The ``"dspy"`` alias must accept ``tools=`` (since canonical
    ``"v7-dspy"`` does). Guards against late normalization that would
    reject tools= on the alias."""

    def my_tool(x: int) -> int:
        return x + 1

    rlm = RLM(lm=_StubLM(), engine="dspy", tools=[my_tool])
    assert rlm.engine == "v7-dspy"
    assert len(list(rlm.tools)) == 1


# ---------------------------------------------------------------------------
# Phase 2 — engine="auto" capability router (xfail until Phase 2)
# ---------------------------------------------------------------------------


def test_auto_with_no_tools_picks_v6_custom():
    """``engine='auto'`` with no ``tools=`` must resolve to v6-custom."""
    rlm = RLM(lm=_StubLM(), engine="auto")
    assert rlm.engine == "v6-custom"


def test_auto_with_tools_picks_v7_dspy():
    """``engine='auto'`` with ``tools=[...]`` must resolve to v7-dspy."""

    def my_tool(x: int) -> int:
        return x + 1

    rlm = RLM(lm=_StubLM(), engine="auto", tools=[my_tool])
    assert rlm.engine == "v7-dspy"


def test_auto_with_empty_tools_list_picks_v6_custom():
    """Empty ``tools=[]`` must NOT trigger dspy routing. Guards against the
    duck's edge case (#10): falsy-but-iterable tools."""
    rlm = RLM(lm=_StubLM(), engine="auto", tools=[])
    assert rlm.engine == "v6-custom"


def test_auto_with_generator_tools_routes_correctly():
    """Generator-based tools= must be materialized once, not consumed
    by the routing decision. Guards against the duck's edge case (#10)."""

    def my_tool(x: int) -> int:
        return x + 1

    rlm = RLM(lm=_StubLM(), engine="auto", tools=(t for t in [my_tool]))
    assert rlm.engine == "v7-dspy"
    assert len(list(rlm.tools)) == 1


def test_auto_with_empty_generator_tools_picks_v6_custom():
    """Empty generator tools= must route to v6-custom. A generator object
    is truthy even when it yields nothing, so checking the raw ``tools=``
    parameter would route incorrectly. The router MUST decide on the
    materialized ``tool_list``, not the generator object's truthiness.
    Phase 2 duck review hardening test."""
    rlm = RLM(lm=_StubLM(), engine="auto", tools=(t for t in []))
    assert rlm.engine == "v6-custom"
    assert list(rlm.tools) == []


def test_auto_resolved_engine_attr_is_canonical():
    """``rlm.engine`` exposes the *resolved* engine, never the literal 'auto'."""
    rlm_a = RLM(lm=_StubLM(), engine="auto")
    rlm_b = RLM(lm=_StubLM(), engine="auto", tools=[lambda: None])
    assert rlm_a.engine in ("v6-custom", "v7-dspy")
    assert rlm_b.engine in ("v6-custom", "v7-dspy")
    assert rlm_a.engine != "auto"
    assert rlm_b.engine != "auto"


@pytest.mark.parametrize(
    "user_input, expected_resolved, expected_unresolved",
    [
        ("auto", "v6-custom", "auto"),       # router pseudo-engine, no tools
        ("default", "v6-custom", "default"), # alias preserved literally
        ("dspy", "v7-dspy", "dspy"),         # alias preserved literally
        ("v6-custom", "v6-custom", "v6-custom"),  # canonical preserved
        ("v7-dspy", "v7-dspy", "v7-dspy"),        # canonical preserved
    ],
)
def test_unresolved_engine_preserves_user_literal_input(
    user_input, expected_resolved, expected_unresolved,
):
    """``_unresolved_engine`` MUST capture the literal input string before
    alias normalization or auto-routing. Phase 5 deprecation/debug logic
    will rely on this to distinguish e.g. ``engine="default"`` from
    ``engine="v6-custom"``. Phase 2 duck review hardening test."""
    rlm = RLM(lm=_StubLM(), engine=user_input)
    assert rlm.engine == expected_resolved
    assert rlm._unresolved_engine == expected_unresolved


def test_explicit_default_engine_with_tools_raises_at_init():
    """Passing ``tools=`` with the ``default`` alias (resolves to v6-custom)
    must raise at __init__ time. After Phase 1, the alias normalizes first,
    then the existing tools-on-non-v7 guard fires with NotImplementedError.

    Phase 2 duck review: the error message must point users at the new
    public engine names (``dspy``, ``auto``), not the canonical
    ``v7-dspy``."""

    def my_tool(x: int) -> int:
        return x + 1

    with pytest.raises(NotImplementedError, match="(?i)tool") as exc_info:
        RLM(lm=_StubLM(), engine="default", tools=[my_tool])
    msg = str(exc_info.value)
    assert "dspy" in msg, f"error should suggest engine='dspy': {msg!r}"
    assert "auto" in msg, f"error should suggest engine='auto': {msg!r}"


def test_explicit_v6_custom_with_tools_still_raises_at_init():
    """``engine="v6-custom"`` + ``tools=`` raises at __init__ time today
    (per PR #18). This is a baseline-lock test confirming the existing
    fast-fail behavior is preserved through consolidation."""

    def my_tool(x: int) -> int:
        return x + 1

    with pytest.raises(NotImplementedError, match="(?i)tool") as exc_info:
        RLM(lm=_StubLM(), engine="v6-custom", tools=[my_tool])
    msg = str(exc_info.value)
    assert "dspy" in msg, f"error should suggest engine='dspy': {msg!r}"
    assert "auto" in msg, f"error should suggest engine='auto': {msg!r}"


# ---------------------------------------------------------------------------
# Phase 4 — default flip (xfail until default kwarg becomes "auto")
# ---------------------------------------------------------------------------


def test_default_constructor_post_flip_uses_auto_internally():
    """After Phase 4 flip: the default constructor's *resolved* engine is
    still v6-custom (no tools), but ``_unresolved_engine == "auto"``.

    Phase 4 changed the default kwarg from ``"v6-custom"`` to ``"auto"``.
    The auto-router resolves the empty-tools case to ``v6-custom`` so the
    historic resolved-engine contract (and any log grep / repr shape) is
    preserved. Only ``_unresolved_engine`` reflects the new default."""
    rlm = RLM(lm=_StubLM())
    assert rlm.engine == "v6-custom"
    assert rlm._unresolved_engine == "auto"


def test_default_constructor_with_tools_post_flip_picks_dspy():
    """After Phase 4: ``RLM(lm=lm, tools=[fn])`` (no engine kwarg) must
    auto-route to v7-dspy. Pre-flip this raised during __init__ because
    the default engine was v6-custom and v6 rejects tools."""

    def my_tool(x: int) -> int:
        return x + 1

    rlm = RLM(lm=_StubLM(), tools=[my_tool])
    assert rlm.engine == "v7-dspy"
    assert rlm._unresolved_engine == "auto"
    assert len(list(rlm.tools)) == 1


def test_resolved_engine_attr_unchanged_post_flip_for_log_grep_back_compat():
    """Log scrapers / observability tooling that grep ``rlm.engine`` (or
    its string representation) for ``"v6-custom"`` still hit on the
    default no-tools constructor after the Phase 4 flip — only the
    user-visible kwarg default changed, not the resolved attribute."""
    rlm = RLM(lm=_StubLM())
    assert rlm.engine == "v6-custom"
    # Negative assertion makes the back-compat contract explicit: the
    # resolved attribute must NOT leak the new "auto" sentinel into log
    # scrapers that were grepping for canonical engine names.
    assert rlm.engine != "auto"


def test_init_signature_engine_default_is_auto_post_flip():
    """Constructor introspection contract: any caller that uses
    ``inspect.signature(RLM.__init__).parameters["engine"].default`` to
    detect the public default must see ``"auto"`` after Phase 4."""
    import inspect
    sig = inspect.signature(RLM.__init__)
    assert sig.parameters["engine"].default == "auto"


def test_explicit_v6_custom_emits_deprecation_warning_post_flip():
    """Phase 5: explicit ``engine='v6-custom'`` emits a DeprecationWarning
    pointing at the new alias. Resolution still works — back-compat is
    preserved — but the user is steered toward ``engine='default'`` (or
    ``engine='auto'`` for capability routing)."""
    with pytest.warns(DeprecationWarning, match=r"v6-custom.*deprecated"):
        rlm = RLM(lm=_StubLM(), engine="v6-custom")
    assert rlm.engine == "v6-custom"
    assert rlm._unresolved_engine == "v6-custom"



# ---------------------------------------------------------------------------
# Phase 5 — deprecation warnings (xfail until Phase 5 ships)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("old_name", ["v6-custom", "v7-dspy"])
def test_old_engine_names_emit_deprecation_warning(old_name: str):
    """``engine='v6-custom'`` and ``engine='v7-dspy'`` must emit a
    DeprecationWarning pointing at the new aliases."""
    with pytest.warns(DeprecationWarning) as records:
        RLM(lm=_StubLM(), engine=old_name)
    msgs = [str(r.message) for r in records if issubclass(r.category, DeprecationWarning)]
    assert msgs, f"No DeprecationWarning emitted for engine={old_name!r}"
    combined = " ".join(msgs).lower()
    if old_name == "v6-custom":
        assert "default" in combined
    else:
        assert "dspy" in combined


@pytest.mark.parametrize("new_name", ["default", "dspy"])
def test_new_engine_names_do_not_emit_deprecation_warning(new_name: str):
    """The new alias names must NOT emit any engine-related DeprecationWarning.

    Filter by message content (not filename) to avoid Windows-path false
    positives AND `stacklevel`-induced false negatives where the warning
    points at the test file rather than `fabric_rlm/runtime.py`. (Per duck
    review #8 of commit 9bce342.)"""
    import warnings

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        RLM(lm=_StubLM(), engine=new_name)

    # Look for engine deprecation warnings by message content, not filename.
    engine_dep_warnings = [
        r for r in records
        if issubclass(r.category, DeprecationWarning)
        and "engine" in str(r.message).lower()
        and ("deprecated" in str(r.message).lower() or "use engine=" in str(r.message).lower())
    ]
    assert not engine_dep_warnings, (
        f"engine={new_name!r} emitted unexpected engine DeprecationWarning: "
        f"{[str(r.message) for r in engine_dep_warnings]}"
    )


def test_auto_engine_name_does_not_emit_deprecation_warning():
    """Phase 2's ``"auto"`` value must NOT emit deprecation warnings."""
    import warnings

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        RLM(lm=_StubLM(), engine="auto")

    engine_dep_warnings = [
        r for r in records
        if issubclass(r.category, DeprecationWarning)
        and "engine" in str(r.message).lower()
        and ("deprecated" in str(r.message).lower() or "use engine=" in str(r.message).lower())
    ]
    assert not engine_dep_warnings, (
        f"engine='auto' emitted unexpected engine DeprecationWarning: "
        f"{[str(r.message) for r in engine_dep_warnings]}"
    )


def test_default_constructor_does_not_emit_engine_deprecation_warning():
    """Sentinel test for Phase 5 trap: when Phase 5 wires up deprecation
    warnings, a naive implementation that keys off the *resolved* canonical
    engine name (rather than the user-supplied name) would spuriously warn
    its own users on the default constructor.

    This must NOT happen. Today it passes trivially (no warnings yet);
    after Phase 5 it will catch the trap if hit."""
    import warnings

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        RLM(lm=_StubLM())  # default constructor, no engine kwarg

    engine_dep_warnings = [
        r for r in records
        if issubclass(r.category, DeprecationWarning)
        and "engine" in str(r.message).lower()
    ]
    assert not engine_dep_warnings, (
        "Default constructor should not emit engine deprecation warnings. "
        f"Got: {[str(r.message) for r in engine_dep_warnings]}"
    )


def test_from_task_legacy_engine_warning_points_at_user_callsite():
    """Phase 5 stacklevel contract for ``RLM.from_task``: the deprecation
    warning emitted when a from_task caller passes ``engine='v6-custom'``
    must point at the **caller's** file (this test file), NOT at
    ``fabric_rlm/runtime.py``.

    Catches the duck-flagged stacklevel bug where emitting from
    ``_normalize_engine_name`` produced a warning whose ``filename`` pointed
    at runtime internals for from_task callers (different call depth than
    direct ``RLM(...)`` construction)."""
    import warnings

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        RLM.from_task(
            "echo",
            inputs={"q": "x"},
            outputs=["a"],
            lm=_StubLM(),
            engine="v6-custom",
        )

    dep = [r for r in records if issubclass(r.category, DeprecationWarning)
           and "v6-custom" in str(r.message)]
    assert len(dep) == 1, (
        f"from_task with engine='v6-custom' must emit exactly one Phase 5 "
        f"DeprecationWarning. Got {len(dep)}: {[str(r.message) for r in dep]}"
    )
    assert dep[0].filename.endswith("test_engine_consolidation_parity.py"), (
        f"DeprecationWarning must point at the from_task caller (this test "
        f"file), not at library internals. Got filename={dep[0].filename!r}, "
        f"lineno={dep[0].lineno}"
    )


def test_from_task_legacy_engine_preserves_unresolved_engine_user_literal():
    """Phase 5 v2 duck-blocking regression test: ``RLM.from_task(engine='v6-custom')``
    must preserve the user's literal in ``_unresolved_engine`` to match
    the contract honored by direct construction.

    Phase 5 v1 fix translated ``kwargs['engine']`` to the public alias
    inside from_task to suppress double-warning. That correctly fixed the
    double-warn but accidentally wiped the user-visible debug breadcrumb
    -- ``rlm._unresolved_engine`` became 'default' instead of the user's
    'v6-custom' literal. Phase 5 v2 fix restores the literal after the
    inner ``cls(**kwargs)`` call.

    Locks both directions of the contract: stored ``engine`` resolves
    canonically (unchanged) AND ``_unresolved_engine`` matches what the
    user actually typed (restored)."""
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", DeprecationWarning)
        rlm_v6 = RLM.from_task(
            "echo", inputs={"q": "x"}, outputs=["a"],
            lm=_StubLM(), engine="v6-custom",
        )
        rlm_v7 = RLM.from_task(
            "echo", inputs={"q": "x"}, outputs=["a"],
            lm=_StubLM(), engine="v7-dspy",
        )

    assert rlm_v6.engine == "v6-custom"
    assert rlm_v6._unresolved_engine == "v6-custom", (
        f"from_task must preserve user's literal in _unresolved_engine; "
        f"alias-translation done to suppress double-warn must not wipe "
        f"the debug breadcrumb. Got: {rlm_v6._unresolved_engine!r}"
    )
    assert rlm_v7.engine == "v7-dspy"
    assert rlm_v7._unresolved_engine == "v7-dspy", (
        f"from_task must preserve user's literal in _unresolved_engine. "
        f"Got: {rlm_v7._unresolved_engine!r}"
    )


def test_direct_constructor_legacy_engine_warning_points_at_user_callsite():
    """Phase 5 stacklevel contract for direct ``RLM(...)`` construction:
    matches the from_task contract — warning points at the user's call
    site, not at runtime internals. Belt-and-braces companion to the
    from_task stacklevel test."""
    import warnings

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        RLM(lm=_StubLM(), engine="v7-dspy")

    dep = [r for r in records if issubclass(r.category, DeprecationWarning)
           and "v7-dspy" in str(r.message)]
    assert len(dep) == 1, (
        f"Direct construction with engine='v7-dspy' must emit exactly one "
        f"Phase 5 DeprecationWarning. Got {len(dep)}: "
        f"{[str(r.message) for r in dep]}"
    )
    assert dep[0].filename.endswith("test_engine_consolidation_parity.py"), (
        f"DeprecationWarning must point at the user call site (this test "
        f"file), not at library internals. Got filename={dep[0].filename!r}, "
        f"lineno={dep[0].lineno}"
    )


def test_adaptive_inner_factory_does_not_self_warn_when_invoked():
    """Phase 5: when ``RLM(engine='adaptive', inner_engine='default')`` is
    constructed and the inner-RLM factory is invoked (simulating a real
    adaptive attempt), the inner ``RLM(**kwargs)`` call must NOT emit a
    Phase 5 ``DeprecationWarning``. The factory's own translation of
    ``inner_engine`` (canonical ``"v6-custom"``) -> public alias
    (``"default"``) before passing as ``engine=`` is the mechanism that
    prevents the self-warn.

    Locks the BLOCKING fix from duck review of Phase 5 diff: pre-fix, the
    factory passed ``engine="v6-custom"`` verbatim, which re-entered the
    Phase 5 deprecation path and self-warned the library on every
    adaptive attempt."""
    import warnings as _warnings
    from fabric_rlm.experimental.adaptive_policy import AttemptConfig
    from fabric_rlm.experimental import adaptive_runner as ar_module
    import fabric_rlm.runtime as rt

    rlm = RLM(
        lm=_StubLM(),
        engine="adaptive",
        inner_engine="default",
        adaptive={"validator": lambda **_: None},
    )

    captured_factory: list = []

    class _FakeRunner:
        def __init__(self, **kwargs):
            captured_factory.append(kwargs.get("rlm_factory"))

        def run(self, *args, **kwargs):
            return None

    original_runner = ar_module.AdaptiveRunner
    try:
        ar_module.AdaptiveRunner = _FakeRunner  # type: ignore[assignment]
        try:
            rlm._run_adaptive({"prompt": "test"})
        except Exception:
            pass
        assert captured_factory, "_run_adaptive did not build a factory"
        factory = captured_factory[0]
        cfg = AttemptConfig(rung=0, max_turns=1)
        # Invoke the factory under a strict warning filter that promotes
        # any engine-related DeprecationWarning to an error. If the inner
        # construction self-warns, this raises and the test fails.
        with _warnings.catch_warnings(record=True) as records:
            _warnings.simplefilter("always")
            try:
                factory(cfg)
            except Exception:
                # Inner RLM may error for non-warning reasons (stub LM,
                # missing skills, etc.) — that's fine for this test.
                pass
        engine_dep = [
            r for r in records
            if issubclass(r.category, DeprecationWarning)
            and "engine=" in str(r.message)
            and ("v6-custom" in str(r.message) or "v7-dspy" in str(r.message))
        ]
        assert not engine_dep, (
            "Adaptive inner-RLM factory self-warned on legacy engine "
            "literal — Phase 5 BLOCKING fix regressed. Factory must "
            "translate canonical inner_engine -> public alias before "
            f"passing as engine=. Got: {[str(r.message) for r in engine_dep]}"
        )
    finally:
        ar_module.AdaptiveRunner = original_runner

