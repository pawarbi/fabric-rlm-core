"""Minimal stand-in for `az account get-access-token`.

The installed Azure CLI ships only .pyc files compiled for CPython 3.14 and one
of them (charset_normalizer) is corrupt, so every token call dies with
"ValueError: bad marshal data". `az account show` still works because it reads
the profile without importing the HTTP stack, which is why the breakage looks
intermittent.

Repairing that install is out of scope for this evaluation, and every harness
script needs exactly one thing from the CLI:

    az.cmd account get-access-token --resource <R> --query accessToken -o tsv

This reproduces that contract using azure-identity, which lives in the ordinary
Python environment and is unaffected. Tokens are cached on disk so the browser
prompt happens once rather than once per script.
"""

from __future__ import annotations

import argparse
import os
import sys

from azure.identity import (
    AuthenticationRecord,
    DeviceCodeCredential,
    InteractiveBrowserCredential,
    TokenCachePersistenceOptions,
)

CACHE_NAME = "fabric-rlm-eval"
RECORD_PATH = __file__ + ".authrecord.json"

# The workspace under evaluation lives in the fabriccat.net tenant. Without an
# explicit tenant the browser credential silently picks the machine's home
# tenant (sandeeppawar@microsoft.com, 72f988bf-...), which yields a perfectly
# valid token that OneLake answers with "ArtifactNotFound" -- the lakehouse is
# simply invisible to that identity. Pin the tenant so a wrong-identity token
# can never be mistaken for a missing artifact.
TENANT_ID = os.environ.get("FABRIC_TENANT_ID", "4a86d5bb-4173-45ee-bfd5-a3b56ee2d3d5")


def _credential():
    cache = TokenCachePersistenceOptions(name=CACHE_NAME, allow_unencrypted_storage=True)
    record = None
    try:
        with open(RECORD_PATH, "r", encoding="utf-8") as handle:
            record = AuthenticationRecord.deserialize(handle.read())
        if record.tenant_id != TENANT_ID:
            record = None
    except FileNotFoundError:
        pass

    # Device code by default. The browser flow needs a graphical session to
    # hand control back to, and when this runs from an automated shell nothing
    # ever completes the redirect, so it fails only after a 300s timeout. The
    # device code prompt goes to stderr because stdout carries the token that
    # the calling scripts parse.
    if os.environ.get("FABRIC_AUTH_BROWSER") == "1":
        return InteractiveBrowserCredential(
            tenant_id=TENANT_ID,
            cache_persistence_options=cache,
            authentication_record=record,
        )

    def _prompt(verification_uri: str, user_code: str, _expires_on) -> None:
        sys.stderr.write(
            f"\n>>> Sign in as sandeeppawar@fabriccat.net\n"
            f">>> Open {verification_uri} and enter code: {user_code}\n\n"
        )
        sys.stderr.flush()

    return DeviceCodeCredential(
        tenant_id=TENANT_ID,
        cache_persistence_options=cache,
        authentication_record=record,
        prompt_callback=_prompt,
        timeout=600,
    )


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--resource", required=True)
    # Accepted and ignored: the callers always ask for the raw token as tsv.
    parser.add_argument("--query", default=None)
    parser.add_argument("-o", "--output", default="tsv")
    args, _unknown = parser.parse_known_args(sys.argv[1:])

    credential = _credential()
    scope = args.resource.rstrip("/") + "/.default"

    try:
        record = credential.authenticate(scopes=[scope])
        with open(RECORD_PATH, "w", encoding="utf-8") as handle:
            handle.write(record.serialize())
    except Exception:
        # An existing cached token can still satisfy get_token even when a
        # fresh interactive authenticate is not possible.
        pass

    token = credential.get_token(scope)
    sys.stdout.write(token.token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
