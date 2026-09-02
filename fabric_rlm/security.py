"""Default-on security baseline for fabric-rlm code execution.

What this module does
---------------------

This module implements two layers of *guardrails* that run inside the parent
process before LM-emitted code is forwarded to the worker subprocess:

1. **Environment scrubbing.** Strips secret-bearing variables
   (``*KEY*``, ``*TOKEN*``, ``*SECRET*``, ``*PASSWORD*``, ``AZURE_CLIENT_*``) from
   the worker's environment so LM-emitted code cannot read them. Core runtime
   variables (``PATH``, ``PYTHONPATH``, ``TEMP``/``TMP``, ``HOME``/``USERPROFILE``,
   ``SystemRoot``) are always preserved so the subprocess can start and import.

2. **Static AST denylist.** Walks the submitted source with the
   :mod:`ast` module and rejects:

   - **Destructive filesystem ops:** ``os.remove``, ``os.unlink``, ``os.rmdir``,
     ``os.removedirs``, ``os.rename``, ``os.replace``, ``shutil.rmtree``,
     ``shutil.move``, ``pathlib.Path.unlink``, and ``pathlib.Path.rmdir`` (the
     two pathlib methods are flagged by bare-method-name match, accepting a
     small false-positive risk on custom classes with the same method name).
     Note: ``Path.rename`` and ``Path.replace`` are **not** blocked because
     ``DataFrame.rename`` / ``Tensor.replace`` etc. are extremely common.
   - **Lakehouse destructive ops:** ``notebookutils.fs.rm`` and
     ``notebookutils.fs.mv``.
   - **Shell/subprocess:** ``os.system``, ``os.popen``, ``subprocess.run``,
     ``subprocess.call``, ``subprocess.check_call``, ``subprocess.check_output``,
     ``subprocess.Popen``, ``subprocess.getoutput``,
     ``subprocess.getstatusoutput``.
   - **Common network-egress libraries:** ``requests``, ``httpx``, ``aiohttp``,
     ``urllib.request``, ``socket``, ``ftplib``, ``smtplib``, ``telnetlib``,
     ``paramiko``. **Not exhaustive** — cloud SDKs, ``grpc``, ``websockets``,
     DB clients, ``sqlalchemy`` etc. can still open sockets. For true egress
     control, layer an OS-level firewall on top.
   - **Targeted dynamic-dispatch escapes:**
     ``importlib.import_module``, ``getattr(<forbidden_module>, ...)``,
     ``__import__("<forbidden_module>")``. The bare builtins ``exec``,
     ``eval``, ``compile``, and benign ``__import__("re")`` are
     **deliberately allowed** — saved trajectories use sandboxed
     ``eval(expr, {"__builtins__": None}, env)`` for math-expression
     evaluation and ``__import__("re").findall(...)`` for inline regex
     helpers ~50+ times across the corpus. Blocking these would break
     no-regression. Users who need the stricter behavior can construct
     ``SecurityPolicy(forbidden_builtins=("exec","eval","compile","__import__"))``
     explicitly.

The visitor also tracks two kinds of aliases within a single submission:

- **Import aliases** — ``import subprocess as sp`` makes ``sp.run`` resolve
  to ``subprocess.run``; ``from subprocess import run as r`` makes ``r(...)``
  resolve to ``subprocess.run``.
- **Assignment aliases** — ``rm = os.remove`` in the same submission makes
  ``rm(...)`` a forbidden call.

What this module does NOT do
----------------------------

Static AST analysis is **bypassable by adversarial input**. The following are
not caught and would require runtime enforcement (PEP 578 audit hooks or
OS-level isolation):

- Cross-turn aliases (state persists between turns). ``rm = os.remove`` in turn
  1 followed by ``rm("file")`` in turn 2 — turn 2's AST sees only ``rm(...)``.
- Reflection/metaclass tricks: ``vars(os)["remove"]``,
  ``__builtins__.eval``, ``type(os).__dict__["remove"]``, ``ctypes`` calls.
- Encoded payloads: e.g. ``eval(base64.b64decode(...))`` or
  ``eval(compile(...))`` chains. Sandboxed ``eval`` is allowed because the
  corpus uses it heavily; an adversary CAN abuse it.
- Write/overwrite prevention: this module does NOT block ``open(path, "w")``,
  ``pathlib.Path.write_text``, dataframe writers, or any other write API. The
  LM can still overwrite files the calling identity has write access to.

Recommended deployment hardening (out of scope for this module)
---------------------------------------------------------------

This baseline is *guardrails*, not isolation. For production deployments
the recommended additional layers are:

- Run the worker under a low-privilege OS identity that cannot reach
  production secrets.
- Mount the LM's workspace as the only writable path; mount everything
  else read-only.
- Use container/seccomp/AppContainer/firewall profiles to constrain
  syscalls and network egress.
- Use scoped Lakehouse credentials (least-privilege SAS / Managed
  Identity) rather than account keys.
- Keep production secrets out of the parent process's env entirely.
"""

from __future__ import annotations

import ast
import fnmatch
import os
from dataclasses import dataclass, field
from typing import Iterable


class SecurityViolation(Exception):
    """Raised when LM-emitted code is rejected by the security policy.

    The legacy :class:`fabric_rlm.interpreter.Interpreter` path converts this
    into an :class:`~fabric_rlm.interpreter.ExecResult` with ``ok=False`` so
    the RLM loop can recover. The v7 ``SubprocessPythonInterpreter`` path
    surfaces it as ``CodeInterpreterError`` so dspy's loop can recover.
    """


# Default forbidden dotted-paths. These are matched on call sites after
# resolving the leftmost name through the per-submission import + assignment
# alias map. Format: ``"module.attr"`` or ``"module.sub.attr"``.
_DEFAULT_FORBIDDEN_CALLS: tuple[str, ...] = (
    # Destructive POSIX filesystem
    "os.system",
    "os.popen",
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "os.removedirs",
    "os.rename",
    "os.renames",
    "os.replace",
    "os.truncate",
    "os.chmod",
    "os.chown",
    # Process replacement / forking the parent
    "os.execv",
    "os.execve",
    "os.execvp",
    "os.execvpe",
    "os.execl",
    "os.execle",
    "os.execlp",
    "os.execlpe",
    "os.spawnv",
    "os.spawnve",
    "os.spawnvp",
    "os.spawnvpe",
    "os.kill",
    "os.killpg",
    "os.forkpty",
    # shutil
    "shutil.rmtree",
    "shutil.move",
    "shutil.chown",
    # subprocess
    "subprocess.run",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.Popen",
    "subprocess.getoutput",
    "subprocess.getstatusoutput",
    # Network egress (kept here so that aliased imports of these dotted names
    # are caught even when the module-level denylist would already cover them).
    "urllib.request.urlopen",
    "urllib.request.urlretrieve",
    "socket.socket",
    "socket.create_connection",
    # Notebookutils destructive ops (Fabric).
    "notebookutils.fs.rm",
    "notebookutils.fs.mv",
    # Dynamic dispatch escapes that defeat the static denylist.
    "importlib.import_module",
    "importlib.__import__",
)

# Modules whose entire surface area is denied. Any import of these (``import X``,
# ``from X import Y``, ``importlib.import_module("X")``) is rejected, and any
# call against a name aliased from one of these modules is rejected.
_DEFAULT_FORBIDDEN_MODULES: tuple[str, ...] = (
    "requests",
    "httpx",
    "aiohttp",
    "ftplib",
    "smtplib",
    "telnetlib",
    "poplib",
    "imaplib",
    "nntplib",
    "paramiko",
    "fabric",
)

# Method names that, by themselves, are too dangerous to leave open regardless
# of the object they're called on. False-positive risk (custom class with the
# same method name) is acceptable for v1 — these names are rare outside their
# canonical pathlib / fs contexts.
_DEFAULT_FORBIDDEN_METHOD_NAMES: tuple[str, ...] = (
    # pathlib.Path destructive — flagged on the bare method name. Custom
    # classes with the same method name will false-positive, which we
    # accept (these names are rare outside their canonical pathlib
    # contexts and the LM can recover via the policy's feedback turn).
    "unlink",
    "rmdir",
    # NOTE: ``rename`` / ``replace`` are NOT in this list — DataFrame.rename
    # is too common to false-positive on. ``os.rename`` / ``os.replace``
    # are still blocked via the dotted-call denylist above.
)

# Built-in names whose blanket-call is rejected. v1 leaves this EMPTY by
# default — saved trajectories use sandboxed ``eval(expr, {"__builtins__":
# None}, env)`` for math-expression evaluation and ``__import__("re")`` for
# inline regex helpers ~50+ times across the corpus. Blocking these at the
# builtin level breaks no-regression. The *targeted* checks remain on:
#
#   * ``__import__("<forbidden_module>")``  is rejected (via the import-
#     decoder in :meth:`_PolicyVisitor.visit_Call`).
#   * ``getattr(<forbidden_module>, ...)``  is rejected (via the getattr
#     decoder in the same method).
#
# So the surface for an LM to bypass the dotted-call denylist via
# ``getattr`` / ``__import__`` is still covered, while sandboxed eval that
# the LM organically writes for arithmetic continues to work.
#
# An operator who wants the stricter behaviour can construct
# ``SecurityPolicy(forbidden_builtins=("exec","eval","compile","__import__"))``
# explicitly.
_DEFAULT_FORBIDDEN_BUILTINS: tuple[str, ...] = ()

# Env keys to strip from the worker's environment by default. Globs are
# matched case-insensitively (Windows env names are case-insensitive). Add
# additional patterns through ``env_strip_globs``.
_DEFAULT_ENV_STRIP_GLOBS: tuple[str, ...] = (
    # Generic secret patterns
    "*KEY*",
    "*TOKEN*",
    "*SECRET*",
    "*PASSWORD*",
    "*PASSWD*",
    "*CREDENTIAL*",
    # Connection-string-style secrets that don't contain the words above.
    # These are common on Azure App Service / Functions and in CI.
    "*CONNECTION_STRING*",
    "*CONNSTR*",
    "SQLCONNSTR_*",
    "CUSTOMCONNSTR_*",
    "MYSQLCONNSTR_*",
    "POSTGRESQLCONNSTR_*",
    "DATABASE_URL",
    "*_DATABASE_URL",
    "POSTGRES_URL",
    "REDIS_URL",
    "MONGO_URL",
    "AMQP_URL",
    "*_DSN",
    # Shared-access-signature URLs
    "*_SAS",
    "*SAS_URL",
    "*SAS_TOKEN",
    # Cloud vendor envs
    "AZURE_CLIENT_*",
    "AZURE_TENANT_*",
    "AZURE_SUBSCRIPTION_*",
    "AZURE_STORAGE_*",
    "AWS_*",
    "GCP_*",
    "GOOGLE_APPLICATION_CREDENTIALS",
    # Common LLM provider envs
    "OPENAI_*",
    "ANTHROPIC_*",
    "GITHUB_TOKEN",
    "GH_TOKEN",
)

# Env keys to ALWAYS preserve even if they match a strip glob. Without these
# the worker subprocess may fail to start or to import standard libraries.
_DEFAULT_ENV_KEEP: tuple[str, ...] = (
    # Runtime
    "PATH",
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONIOENCODING",
    "PYTHONUNBUFFERED",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONUTF8",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    # Windows
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PROGRAMDATA",
    "PROGRAMW6432",
    "ALLUSERSPROFILE",
    "COMMONPROGRAMFILES",
    # POSIX
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TMPDIR",
)


@dataclass(frozen=True)
class SecurityPolicy:
    """Default-on guardrails applied to LM-emitted code.

    Construct with :meth:`default` to get the recommended baseline (everything
    enabled with sensible denylists), or with :meth:`disabled` to turn off all
    enforcement. Custom policies can override any field at construction time.

    Attributes
    ----------
    enabled:
        Master switch. When ``False`` :meth:`validate_code` returns ``None``
        and :meth:`scrub_env` returns the env unchanged.
    forbidden_calls:
        Tuple of dotted-paths (e.g. ``"subprocess.run"``) rejected at call
        sites after alias resolution.
    forbidden_modules:
        Tuple of module names rejected on import (``import X`` or
        ``from X import ...``).
    forbidden_method_names:
        Tuple of bare method names rejected on any method call. Use sparingly
        — false-positive risk on custom classes.
    forbidden_builtins:
        Tuple of built-in names rejected as calls (``exec``, ``eval`` etc.).
    env_strip_globs:
        fnmatch-style globs (case-insensitive) for env variable names to
        remove from the worker's environment.
    env_keep:
        Env variable names always preserved, even if they match a strip glob.
        Matched case-insensitively.
    on_violation:
        Either ``"feedback"`` (the runtime returns the violation as a turn-
        level error so the LM can recover) or ``"raise"`` (the runtime raises
        :class:`SecurityViolation`). Default ``"feedback"`` is what keeps
        ``no regression in behaviour`` true on tasks the LM solves another
        way after rejection.
    """

    enabled: bool = True
    forbidden_calls: tuple[str, ...] = _DEFAULT_FORBIDDEN_CALLS
    forbidden_modules: tuple[str, ...] = _DEFAULT_FORBIDDEN_MODULES
    forbidden_method_names: tuple[str, ...] = _DEFAULT_FORBIDDEN_METHOD_NAMES
    forbidden_builtins: tuple[str, ...] = _DEFAULT_FORBIDDEN_BUILTINS
    env_strip_globs: tuple[str, ...] = _DEFAULT_ENV_STRIP_GLOBS
    env_keep: tuple[str, ...] = _DEFAULT_ENV_KEEP
    on_violation: str = "feedback"

    # ----- Factory helpers -------------------------------------------------

    @classmethod
    def default(cls) -> "SecurityPolicy":
        """Recommended default policy with all guardrails enabled."""
        return cls()

    @classmethod
    def disabled(cls) -> "SecurityPolicy":
        """Policy with all enforcement turned off (escape hatch for tests)."""
        return cls(enabled=False)

    # ----- Env scrub -------------------------------------------------------

    def scrub_env(self, env: dict[str, str] | None = None) -> dict[str, str]:
        """Return a copy of *env* (defaults to :data:`os.environ`) with
        secret-bearing keys removed.

        Matching is case-insensitive — Windows env names are case-insensitive
        and POSIX users often use lowercase aliases.
        """
        source = dict(os.environ) if env is None else dict(env)
        if not self.enabled:
            return source

        keep_upper = {k.upper() for k in self.env_keep}
        result: dict[str, str] = {}
        for k, v in source.items():
            k_upper = k.upper()
            if k_upper in keep_upper:
                result[k] = v
                continue
            if any(fnmatch.fnmatch(k_upper, g.upper()) for g in self.env_strip_globs):
                continue  # strip
            result[k] = v
        return result

    # ----- Code validation -------------------------------------------------

    def validate_code(self, code: str) -> str | None:
        """Inspect *code* statically. Return ``None`` if the code is allowed,
        otherwise a recoverable error message describing the violation.

        Syntax errors are returned as-is — they will fail at runtime anyway,
        but surfacing them at the parent saves a round-trip.
        """
        if not self.enabled or not code or not code.strip():
            return None
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Defer to the worker — it'll produce a proper traceback.
            return None
        visitor = _PolicyVisitor(self)
        try:
            visitor.visit(tree)
        except _PolicyReject as r:
            return r.message
        return None


class _PolicyReject(Exception):
    """Internal control-flow signal raised when the visitor finds a violation."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _dotted_from_attr(node: ast.AST) -> str | None:
    """Return ``"a.b.c"`` for an Attribute chain rooted on a Name, else None.

    Used to recognise ``subprocess.run`` or ``notebookutils.fs.rm`` patterns
    in the AST.
    """
    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    parts.append(cur.id)
    return ".".join(reversed(parts))


# Denied names that collide with something the caller almost certainly wanted.
# Without a hint the model reads "module 'fabric' is disabled (network egress
# off)" as "the fabric module is unavailable here", concludes the whole
# semantic-link path is blocked, and refuses the question - while
# ``import sempy.fabric`` would have worked. Measured on a semantic-model eval:
# 11 refusals across three runs, every one of them citing a blockage that did
# not apply to the path it gave up on.
_COLLISION_HINTS: dict[str, str] = {
    "fabric": (
        " NOTE: this rule targets the 'fabric' SSH automation package, NOT "
        "Microsoft Fabric. For a Power BI semantic model use "
        "`import sempy.fabric as fabric` - sempy is allowed and works here."
    ),
}


def _name_collision_hint(head: str) -> str:
    """Extra guidance when a denied name is one people reach for by mistake."""
    return _COLLISION_HINTS.get(head, "")


class _PolicyVisitor(ast.NodeVisitor):
    """AST visitor that enforces a :class:`SecurityPolicy`.

    Maintains two alias maps per submission:

    - ``module_alias`` maps a local name to the dotted module path it was
      imported from. ``import subprocess as sp`` adds ``sp -> "subprocess"``;
      ``from os import remove as rm`` adds ``rm -> "os.remove"``.
    - ``value_alias`` maps a local name to the dotted call path it was
      assigned to. ``rm = os.remove`` adds ``rm -> "os.remove"``.

    On a Call node, the leftmost name in the dotted chain is rewritten through
    these maps before matching against ``forbidden_calls`` / ``forbidden_modules``.
    """

    def __init__(self, policy: SecurityPolicy) -> None:
        self.policy = policy
        self.module_alias: dict[str, str] = {}
        self.value_alias: dict[str, str] = {}
        self._forbidden_calls = frozenset(policy.forbidden_calls)
        self._forbidden_modules = frozenset(policy.forbidden_modules)
        self._forbidden_method_names = frozenset(policy.forbidden_method_names)
        self._forbidden_builtins = frozenset(policy.forbidden_builtins)

    # ----- Imports ---------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            mod = alias.name
            self._check_forbidden_module(mod, where="import")
            if alias.asname:
                # ``import subprocess as sp`` -> ``sp`` resolves to ``subprocess``
                self.module_alias[alias.asname] = mod
            else:
                # ``import urllib.request`` binds the *top* package name
                # locally (``urllib``), and ``urllib.request.urlopen`` is
                # then a normal attribute walk. Aliasing ``urllib`` to
                # ``urllib.request`` here would incorrectly rewrite the
                # dotted path to ``urllib.request.request.urlopen`` and
                # miss the denylist match.
                head = mod.split(".")[0]
                self.module_alias.setdefault(head, head)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        self._check_forbidden_module(mod, where="from-import")
        for alias in node.names:
            local = alias.asname or alias.name
            dotted = f"{mod}.{alias.name}" if mod else alias.name
            self.module_alias[local] = dotted
            # ``from subprocess import run`` makes ``run -> "subprocess.run"``
            # which we want resolvable when ``run(...)`` is called later.
        self.generic_visit(node)

    # ----- Assignments (intra-submission aliasing) -------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        # ``rm = os.remove`` or ``rm = subprocess.run``.
        if isinstance(node.value, (ast.Attribute, ast.Name)):
            dotted = self._resolve_dotted(node.value)
            if dotted is not None:
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        self.value_alias[tgt.id] = dotted
        self.generic_visit(node)

    # ----- Calls (the main enforcement point) ------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        dotted = self._resolve_dotted(func)
        if dotted is not None:
            self._check_call(dotted, node)

        # ``getattr(os, "remove")(...)`` and ``getattr(__import__("os"), "remove")(...)``.
        # When the outer is a Call whose .func is itself a getattr Call, we
        # already visited the inner getattr Call below and rejected it there.
        # Additionally detect getattr(<forbidden_module>, "name") here.
        if isinstance(func, ast.Call) and self._is_getattr_call(func):
            self._reject_getattr_chain(func)

        # Top-level call that IS getattr(<forbidden>, ...).
        if isinstance(func, ast.Name) and func.id == "getattr":
            self._maybe_reject_getattr(node)

        # Top-level call that IS __import__("forbidden_module").
        if isinstance(func, ast.Name) and func.id == "__import__":
            self._maybe_reject_dunder_import(node)

        # Method-name blacklist: ``some_path.unlink()``.
        if isinstance(func, ast.Attribute) and func.attr in self._forbidden_method_names:
            raise _PolicyReject(
                f"SecurityPolicyViolation: method '.{func.attr}()' is disabled "
                f"by default to prevent file deletion. To override, construct "
                f"RLM(security=SecurityPolicy(forbidden_method_names=())) — but "
                f"prefer non-destructive APIs."
            )

        self.generic_visit(node)

    # ----- Helpers ---------------------------------------------------------

    def _resolve_dotted(self, node: ast.AST) -> str | None:
        """Resolve an Attribute/Name chain to a dotted path, applying aliases.

        ``sp.run`` after ``import subprocess as sp`` returns ``"subprocess.run"``.
        ``rm`` after ``rm = os.remove`` returns ``"os.remove"``.
        """
        raw = _dotted_from_attr(node) if isinstance(node, ast.Attribute) else (
            node.id if isinstance(node, ast.Name) else None
        )
        if raw is None:
            return None
        head, _, rest = raw.partition(".")
        resolved_head = self.value_alias.get(head) or self.module_alias.get(head) or head
        return f"{resolved_head}.{rest}" if rest else resolved_head

    def _check_call(self, dotted: str, node: ast.Call) -> None:
        # Built-in dynamic-dispatch escapes (exec/eval/compile/__import__).
        # These show up as plain Name nodes; ``dotted`` is just the name.
        if dotted in self._forbidden_builtins:
            raise _PolicyReject(
                f"SecurityPolicyViolation: built-in '{dotted}(...)' is disabled "
                f"because it bypasses the static denylist. If you need to "
                f"evaluate a literal, use ast.literal_eval."
            )

        if dotted in self._forbidden_calls:
            raise _PolicyReject(self._explain_call(dotted))

        head = dotted.split(".", 1)[0]
        if head in self._forbidden_modules:
            raise _PolicyReject(
                f"SecurityPolicyViolation: module '{head}' is disabled (network "
                f"egress / external IO is off by default). For local file work "
                f"use pathlib, pandas, or the existing data loaders. To enable, "
                f"pass RLM(security=SecurityPolicy(forbidden_modules=())) — but "
                f"verify the threat model first."
                + _name_collision_hint(head)
            )

    def _check_forbidden_module(self, mod: str, *, where: str) -> None:
        head = mod.split(".", 1)[0]
        if head in self._forbidden_modules:
            raise _PolicyReject(
                f"SecurityPolicyViolation: importing '{mod}' is disabled "
                f"(network/external IO is off by default at the {where} site)."
                + _name_collision_hint(head)
            )

    def _is_getattr_call(self, node: ast.Call) -> bool:
        return isinstance(node.func, ast.Name) and node.func.id == "getattr"

    def _maybe_reject_getattr(self, node: ast.Call) -> None:
        """``getattr(<forbidden_module>, "anything")`` is rejected."""
        if not node.args:
            return
        target_dotted = self._resolve_dotted(node.args[0])
        if target_dotted is None:
            return
        head = target_dotted.split(".", 1)[0]
        if head in self._forbidden_modules or target_dotted in self._forbidden_calls:
            raise _PolicyReject(
                f"SecurityPolicyViolation: getattr() against a disabled module "
                f"('{target_dotted}') is rejected because it bypasses the "
                f"static denylist."
            )
        # ``getattr(os, "remove")`` — second arg is a string literal whose
        # combination with the first arg points to a forbidden call.
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            attr_name = node.args[1].value
            if isinstance(attr_name, str):
                combined = f"{target_dotted}.{attr_name}"
                if combined in self._forbidden_calls:
                    raise _PolicyReject(
                        f"SecurityPolicyViolation: getattr() target resolves to "
                        f"disabled call '{combined}'."
                    )

    def _reject_getattr_chain(self, node: ast.Call) -> None:
        # Defer to _maybe_reject_getattr — same logic.
        self._maybe_reject_getattr(node)

    def _maybe_reject_dunder_import(self, node: ast.Call) -> None:
        if not node.args:
            return
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            head = first.value.split(".", 1)[0]
            if head in self._forbidden_modules:
                raise _PolicyReject(
                    f"SecurityPolicyViolation: __import__('{first.value}') is "
                    f"disabled (importing '{head}' is off by default)."
                )
            # Also reject importing modules whose entire call surface is on
            # the denylist (e.g. ``subprocess`` — every call we care about
            # under it is forbidden, so dynamic import is just a bypass).
            forbidden_call_heads = {c.split(".", 1)[0] for c in self._forbidden_calls}
            if head in forbidden_call_heads and head in {"subprocess", "socket", "urllib"}:
                raise _PolicyReject(
                    f"SecurityPolicyViolation: __import__('{first.value}') is "
                    f"disabled because '{head}' has destructive/network calls "
                    f"that the policy blocks and dynamic import would bypass."
                )

    def _explain_call(self, dotted: str) -> str:
        if dotted.startswith("os.") or dotted.startswith("shutil.") or dotted.startswith("pathlib."):
            return (
                f"SecurityPolicyViolation: call to '{dotted}' is disabled to "
                f"prevent accidental data deletion. Read-only file APIs (open "
                f"for read, pathlib.Path.read_text, etc.) remain available."
            )
        if dotted.startswith("subprocess.") or dotted.startswith("os.system") or dotted.startswith("os.popen"):
            return (
                f"SecurityPolicyViolation: call to '{dotted}' is disabled "
                f"because shell/subprocess execution bypasses the network and "
                f"filesystem guardrails. Use pure Python APIs (pathlib, "
                f"pandas, csv, json) instead."
            )
        if dotted.startswith(("urllib.", "socket.")):
            return (
                f"SecurityPolicyViolation: call to '{dotted}' is disabled "
                f"because network egress is off by default. Work with files "
                f"already on disk."
            )
        if dotted.startswith("notebookutils."):
            return (
                f"SecurityPolicyViolation: call to '{dotted}' is disabled "
                f"because it can delete or move lakehouse files. Use "
                f"notebookutils.fs.ls / .head / .get for read-only inspection."
            )
        if dotted.startswith("importlib."):
            return (
                f"SecurityPolicyViolation: '{dotted}' is disabled because "
                f"dynamic import bypasses the static denylist."
            )
        return (
            f"SecurityPolicyViolation: call to '{dotted}' is disabled by "
            f"default. See fabric_rlm.security.SecurityPolicy to customise."
        )


__all__ = ["SecurityPolicy", "SecurityViolation"]
