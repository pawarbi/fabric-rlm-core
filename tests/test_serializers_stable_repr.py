"""Stable repr in opaque_marker — addresses must not appear in serialized state.

Default Python repr for objects without a custom ``__repr__`` is
``<module.path.Class object at 0xHEXADDR>``. The hex memory address changes
every Python process and even between turn restarts in long-running runtimes.
When ``opaque_marker`` ships this raw repr into the LM-visible state snapshot,
two adjacent turns that share the same workbook/dataframe/connection produce
*different* serialized state (only the address differs), defeating prompt
caching and producing noise in trajectory diffs.

This test file pins the contract that ``opaque_marker`` strips the volatile
``at 0xHEXADDR`` segment so two instances of the same opaque class produce
identical markers.
"""

from __future__ import annotations

import re

from fabric_rlm.serializers import freeze, opaque_marker, snapshot


class _Plain:
    """Class with no custom ``__repr__`` — uses Python default
    ``<module.Class object at 0xADDR>``.
    """


class _CustomRepr:
    def __repr__(self) -> str:
        return "<CustomRepr name='alpha' rows=10>"


class _CustomReprWithFakeAddr:
    """Custom repr that legitimately contains the substring 'at 0x' but the
    hex chars are stable values — must NOT be stripped.
    """

    def __repr__(self) -> str:
        return "ChunkRef(offset_at_0x10, length=64)"


class _CustomReprWithPriceText:
    """Prose-like custom repr where 'at 0xff' is followed by a space, NOT a
    closing delimiter — must NOT be stripped by the lookahead-anchored regex.
    """

    def __repr__(self) -> str:
        return "price at 0xff per unit"


class _NestedAddressRepr:
    """Repr containing nested-form '<function f at 0x...>, 1)' — the address
    IS followed by a closing delimiter, so it MUST be stripped.
    """

    def __repr__(self) -> str:
        return "PartialLike(<function f at 0x7f1234abcd>, 1)"


_HEX_AT_PATTERN = re.compile(r" at 0x[0-9a-fA-F]+")


class TestOpaqueMarkerStableRepr:
    def test_default_repr_address_is_stripped(self):
        marker = opaque_marker(_Plain())
        # No ' at 0x...' substring should remain
        assert not _HEX_AT_PATTERN.search(marker["__repr__"]), marker["__repr__"]
        # But the type information is preserved
        assert "_Plain" in marker["__repr__"]
        assert marker["__type__"] == "_Plain"

    def test_two_instances_of_same_class_produce_identical_markers(self):
        a = opaque_marker(_Plain())
        b = opaque_marker(_Plain())
        # This is the key cache-friendliness contract: two different runtime
        # instances of the same opaque class must serialize identically.
        assert a == b

    def test_custom_repr_without_address_is_preserved(self):
        marker = opaque_marker(_CustomRepr())
        assert marker["__repr__"] == "<CustomRepr name='alpha' rows=10>"

    def test_custom_repr_with_non_address_at_0x_is_preserved(self):
        # 'at_0x10' is not the volatile pattern (no space, underscore present)
        marker = opaque_marker(_CustomReprWithFakeAddr())
        assert marker["__repr__"] == "ChunkRef(offset_at_0x10, length=64)"

    def test_prose_repr_with_at_0xhex_followed_by_space_is_preserved(self):
        # Lookahead requires the hex be followed by ``>``, ``,``, ``)`` or
        # ``]`` — prose like "at 0xff per unit" must survive untouched.
        marker = opaque_marker(_CustomReprWithPriceText())
        assert marker["__repr__"] == "price at 0xff per unit"

    def test_nested_address_in_repr_is_stripped(self):
        # The nested ``<function f at 0x7f1234abcd>`` IS volatile and IS
        # followed by ``>`` — the lookahead allows it to be stripped.
        marker = opaque_marker(_NestedAddressRepr())
        assert not _HEX_AT_PATTERN.search(marker["__repr__"]), marker["__repr__"]
        assert "PartialLike(<function f>, 1)" == marker["__repr__"]

    def test_serializable_flag_still_false(self):
        marker = opaque_marker(_Plain())
        assert marker["__serializable__"] is False

    def test_repr_still_truncated_at_300_chars(self):
        class _Big:
            def __repr__(self) -> str:
                return "x" * 1000

        marker = opaque_marker(_Big())
        assert len(marker["__repr__"]) <= 300


class TestEndToEndCacheFriendliness:
    """End-to-end: snapshot of two namespaces with different instances of the
    same opaque class must produce byte-identical JSON. This is what would
    actually be sent into prompt caches.
    """

    def test_snapshot_byte_identical_across_instances(self):
        import json

        ns1 = {"wb": _Plain(), "answer": 42}
        ns2 = {"wb": _Plain(), "answer": 42}
        s1 = json.dumps(snapshot(ns1), sort_keys=True)
        s2 = json.dumps(snapshot(ns2), sort_keys=True)
        assert s1 == s2

    def test_freeze_routes_to_stable_marker(self):
        # freeze() falls through to opaque_marker for unsupported types
        f = freeze(_Plain())
        assert not _HEX_AT_PATTERN.search(f["__repr__"])

    def test_nested_opaque_in_dict_is_stable(self):
        import json

        d1 = {"items": [_Plain(), _Plain()], "tag": "v1"}
        d2 = {"items": [_Plain(), _Plain()], "tag": "v1"}
        assert json.dumps(freeze(d1), sort_keys=True) == json.dumps(freeze(d2), sort_keys=True)


class TestRegressionExistingBehavior:
    """The fix must NOT regress any of the existing serializer behaviors."""

    def test_freeze_object_still_returns_marker_dict(self):
        frozen = freeze(object())
        assert frozen["__type__"] == "object"
        assert frozen["__serializable__"] is False
        assert "__repr__" in frozen
