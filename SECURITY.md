# Security Policy

## Threat model — read this before running untrusted prompts

fabric-rlm executes **LM-generated Python code in a real CPython subprocess
on your machine**. The default-on security baseline
(`fabric_rlm.security.SecurityPolicy`) provides **guardrails, not isolation**:

1. **Environment scrubbing** — secret-bearing env vars (`*KEY*`, `*TOKEN*`,
   `*SECRET*`, connection strings, cloud-vendor variables, …) are stripped
   from the worker subprocess's environment.
2. **Static AST denylist** — destructive filesystem ops, shell/subprocess
   calls, common network-egress libraries, and targeted dynamic-dispatch
   escapes are rejected before the code reaches the worker.

### What the baseline does NOT protect against

Static analysis is bypassable by adversarial input. Known, documented gaps
(see the `fabric_rlm/security.py` module docstring for the full list):

- Cross-turn aliasing (`rm = os.remove` in turn 1, `rm(...)` in turn 2).
- Reflection/metaclass tricks (`vars(os)["remove"]`, `ctypes`, …).
- Encoded payloads through the deliberately-allowed sandboxed `eval`.
- **File writes/overwrites are not blocked at all** — the LM can overwrite
  any file the calling identity can write.

### Recommended hardening for production

- Run the worker under a low-privilege OS identity that cannot reach
  production secrets.
- Mount the LM's workspace as the only writable path.
- Constrain network egress with an OS-level firewall or container profile.
- Use least-privilege, scoped credentials (SAS / Managed Identity) for any
  lakehouse access.

If your deployment treats the AST denylist as a hard security boundary
against a motivated adversary, that is a misuse of this library — layer OS
isolation on top.

## Supported versions

Only the latest released minor version receives security fixes.

## Reporting a vulnerability

Please **do not open a public issue** for security-sensitive reports.

Use GitHub's private vulnerability reporting on this repository
(*Security → Report a vulnerability*). You can expect an acknowledgement
within 7 days.

Reports about the documented limitations above (e.g. "I bypassed the AST
denylist with reflection") are appreciated but are by-design limitations of
a guardrail layer, not vulnerabilities — feel free to open a regular issue
for those, ideally with the bypass pattern so it can be added to the
denylist or the security-corpus regression tests.
