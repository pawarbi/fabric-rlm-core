# SPEC: FabricLM Azure AAD Token Refresh

Status: Phase 1 (Specify). Awaiting human review before Phase 2.

## Problem

`FabricLM` (in `fabric_rlm/lm.py:82`) calls
`TokenUtils().get_openai_auth_header()` **once at construction** and bakes
the bearer token into `extra_headers`. Azure AAD tokens expire after ~60
minutes. Any RLM run longer than the token TTL hits 401 mid-run and every
remaining LM call fails with:

    litellm.AuthenticationError: AzureException AuthenticationError -
    Error code: 401 - {'Message': 'User Aad Token is expired.'}

**Confirmed real impact** (from
`comparison_5way_bandit-full-20260502-161343/`): all 5 VLIW questions
(`VLIW_hard_10` through `VLIW_hard_14`) hit token expiry after ~50 minutes
of run time. 4 of the 5 were lost to it. Suppresses the per-template pass
rate from a possible 1–2/5 down to 0/5 — and shows up generically in **any
Fabric job longer than ~1 hour**.

## Goal

When `FabricLM` (or any LM constructed with a token-provider) gets a 401
from the inference endpoint, it must:

1. Refresh the bearer token via the original token-provider callable.
2. Retry the same request once with the new token.
3. If the retry also fails (different cause, or the token-provider itself
   fails), surface the original error to the caller — no infinite loops.

The fix MUST be **provider-agnostic**: any short-lived bearer-token
auth (Azure AAD, GCP IAM, AWS IAM with short tokens, custom OIDC) should
benefit by passing a `token_provider: Callable[[], str]` instead of a
static `extra_headers["Authorization"]` value.

## Non-goals

- Persisting the refreshed token across processes / disk caching
- Proactive refresh on a timer (always lazy on 401)
- Retry on any other litellm error (rate limit, transient 5xx — those have
  their own retry layer in litellm and dspy)
- Changing how non-Fabric LMs (`OpenAILM`, `AnthropicLM`) work — they
  receive a static API key from env and don't have this problem

## Approach

Introduce a thin wrapper class `_RefreshingLM(dspy.LM)` in
`fabric_rlm/lm.py`. Constructor takes the same args as `dspy.LM` plus a
`token_provider: Callable[[], str] | None` and a
`token_header: str = "Authorization"`.

```python
class _RefreshingLM(dspy.LM):
    def __init__(self, *args, token_provider=None, token_header="Authorization", **kwargs):
        self._token_provider = token_provider
        self._token_header = token_header
        super().__init__(*args, **kwargs)

    def _refresh(self):
        if self._token_provider is None:
            return False
        new = self._token_provider()
        hdrs = dict(self.kwargs.get("extra_headers") or {})
        hdrs[self._token_header] = new
        self.kwargs["extra_headers"] = hdrs
        return True

    def __call__(self, *args, **kwargs):
        try:
            return super().__call__(*args, **kwargs)
        except Exception as exc:
            if not self._is_auth_expired(exc) or not self._refresh():
                raise
            return super().__call__(*args, **kwargs)

    @staticmethod
    def _is_auth_expired(exc):
        msg = str(exc)
        cls = type(exc).__name__
        return ("AuthenticationError" in cls
                and ("401" in msg or "expired" in msg.lower() or "CUSTOMER_UNAUTHORIZED" in msg))
```

`_fabric_factory` becomes:

```python
def _fabric_factory(model_name, **overrides):
    from synapse.ml.fabric.token_utils import TokenUtils
    ...
    token_provider = lambda: TokenUtils().get_openai_auth_header()
    auth_header = token_provider()
    ...
    return _RefreshingLM(f"azure/{model}", token_provider=token_provider, **kwargs)
```

`OpenAILM`/`AnthropicLM` are unchanged; they pass `token_provider=None`
implicitly so `_RefreshingLM` degrades to plain `dspy.LM` behaviour.

## Generalization

- **No Fabric-specific assumption in `_RefreshingLM`.** It accepts any
  zero-arg `token_provider` callable. A GCP user passes a lambda that
  reads from `google.auth.default()`; an AWS user passes one that calls
  `boto3.client('sts').get_session_token()`; a Fabric user passes
  `TokenUtils().get_openai_auth_header`. The class itself imports nothing
  Fabric-specific.
- **No model-specific assumption.** Works for `azure/gpt-5`,
  `openai/gpt-4o`, any litellm-supported model. Auth detection is by
  exception class+message, not provider.
- **Header name configurable.** Default `Authorization` covers 95% of
  cases; some providers use `X-Api-Key` or custom — pass `token_header=`.
- **Opt-in for non-Fabric users.** `OpenAILM`/`AnthropicLM` keep their
  current behaviour. Anyone who wants refresh on those calls
  `_RefreshingLM(...)` directly with their token_provider.
- **No change to bandit/adaptive/skill/runtime layers.** The refresh
  happens transparently inside `__call__`, same return type, same error
  semantics on non-auth failures.

## Testing strategy

- **Unit (`tests/test_refreshing_lm.py`)** — uses a mock `dspy.LM`
  subclass that raises a fake `AuthenticationError` on first call and
  returns a normal completion on second. Asserts:
  - On 401: `token_provider` is called, header is updated, second call
    succeeds, return value matches the second call.
  - On non-401 (rate limit, etc.): `token_provider` is NOT called, original
    exception propagates.
  - On 401 + 401 (refresh succeeds but token still rejected): original
    exception propagates after one retry, no infinite loop.
  - On `token_provider=None` + 401: original exception propagates
    immediately (degrades to plain `dspy.LM`).
  - On `token_provider` raising: original 401 propagates (refresh failure
    does not mask the real error).
- **Integration** — `tests/test_fabric_lm_factory.py` (existing? add if
  missing) asserts `FabricLM("gpt-5")` returns a `_RefreshingLM` instance
  with a non-None `_token_provider` (mock the synapse import).
- **Manual validation** — re-run a single VLIW question on Fabric after a
  forced 90-minute idle; expect success where the original run hit 401.

No test should require a real Fabric environment.

## Success criteria

| # | Criterion | How to verify |
|---|---|---|
| 1 | All unit tests pass | `pytest tests/test_refreshing_lm.py` |
| 2 | `FabricLM(...)` returns `_RefreshingLM` with token_provider | Integration test |
| 3 | Existing `tests/test_lm.py` and downstream tests unchanged | Full pytest pass on touched modules |
| 4 | Re-running one VLIW question after 90min completes | Manual notebook run, no 401 |
| 5 | No new public API surface (class is `_RefreshingLM`, underscore-prefixed) | `git diff fabric_rlm/__init__.py` is empty |
| 6 | OpenAILM and AnthropicLM behaviour byte-identical | `tests/test_lm.py` no diff |

## Boundaries

- **Always**: write unit tests first (TDD), keep `_RefreshingLM` private,
  preserve `OpenAILM`/`AnthropicLM` behaviour, run full pytest before
  commit.
- **Ask first**: any change that would expose `_RefreshingLM` publicly,
  any retry-count > 1, any change to litellm callbacks (out of scope for
  this fix).
- **Never**: add a timer/proactive refresh, cache tokens to disk, add a
  Fabric-specific dependency to the wrapper class itself.

## Open questions

1. Does `dspy.LM.__call__` accept `*args, **kwargs` cleanly for delegation?
   → Verify in Phase 2 by reading dspy source (it likely does — it forwards
   to litellm).
2. Does `dspy.LM` expose `self.kwargs` as a mutable dict? → Verify; if not,
   we may need to mutate `extra_headers` via a different attribute.
3. Should we also intercept `dspy.LM.forward` (used by some adapters)? →
   Probably yes — Phase 2 will check both code paths.
4. Should the cost tracker / token usage tracker see the retry as 1 call
   or 2? → 1 call (the retry is bookkeeping). Check `runtime_token_tracking`
   integration in Phase 2.

## Phases (after spec approval)

- **Phase 2 (Plan)**: dspy.LM internals review, decide on `forward` vs
  `__call__` interception, dependency order.
- **Phase 3 (Tasks)**: ≤5-file tasks per `spec-driven-development` skill
  (`_RefreshingLM` impl → factory rewire → unit tests → integration test
  → manual Fabric validation).
- **Phase 4 (Implement)**: TDD per
  `.github/skills/test-driven-development/SKILL.md`.
