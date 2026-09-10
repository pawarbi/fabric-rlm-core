"""F16 repro: the clarification guard is English-only and fails OPEN.

`assert_not_clarification_request` rejects an answer that is really a deferral
("please confirm...", "I need more information..."). It matches English opener
phrases. A deferral in any other language matches nothing, so no assert fires
and the answer is accepted as substantive.

Also checks verify.py's list normalisation, which strips the English "and"/"&"
but no other language's conjunction.

No Fabric, no key, no network. Exits non-zero if the finding stops reproducing,
so it doubles as a regression test for any future fix.

    python repro_clarification_language.py
"""
import sys

from fabric_rlm.validators import _CLARIFICATION_OPENERS
from fabric_rlm.verify import _LEADING_CONJUNCTION


def is_detected(text):
    return any(rx.search(text[:200]) for rx in _CLARIFICATION_OPENERS)


ENGLISH = [
    "Please confirm which fiscal calendar to use.",
    "Could you clarify the SLA definition?",
    "I need more information about the grain.",
    "Before I can answer, the units must be defined.",
]

# The same four deferrals: de / es / fr / it. Identical intent, identical
# uselessness as an answer.
OTHER = [
    "Bitte bestaetigen Sie, welcher Geschaeftskalender gilt.",
    "Podria aclarar la definicion de SLA?",
    "Je dois avoir plus d'informations sur la granularite.",
    "Prima di rispondere, occorre definire le unita.",
]

print("=== deferrals in ENGLISH (should be detected and rejected) ===")
for s in ENGLISH:
    print(f"  detected={str(is_detected(s)):5} {s}")
en = sum(is_detected(s) for s in ENGLISH)

print("\n=== the SAME deferrals, other languages ===")
for s in OTHER:
    print(f"  detected={str(is_detected(s)):5} {s}")
other = sum(is_detected(s) for s in OTHER)

print("\n=== verify.py list normalisation: leading conjunction stripped? ===")
conj = {s: bool(_LEADING_CONJUNCTION.match(s)) for s in
        ["and Contoso", "& Contoso", "und Contoso", "y Contoso", "et Contoso"]}
for s, v in conj.items():
    print(f"  stripped={str(v):5} {s!r}")

print("\n" + "=" * 64)
print(f"English deferrals detected     : {en}/{len(ENGLISH)}")
print(f"Non-English deferrals detected : {other}/{len(OTHER)}")
print("\nDirection of failure: no regex matches -> no assert fires -> the")
print("submission PASSES. The guard fails OPEN, so a non-English deferral is")
print("accepted as a substantive answer.")

ok = (en == len(ENGLISH) and other == 0
      and conj["and Contoso"] and not conj["und Contoso"])
print("\nF16 reproduces exactly." if ok else
      "\nF16 did NOT reproduce -- behaviour changed; re-read the finding.")
sys.exit(0 if ok else 1)
