"""Inline execution-trace representation from NExT (Ni et al., 2024), Sec. 2 / App. A.1.

A state h_i is represented by *only the variables that changed* relative to h_{i-1},
rendered with Python literal syntax, numbered by execution order. States are grouped
by the statement that produced them; when a statement produced more than 3 states
(loops, recursion) we keep the first two and the last. Each group is joined with "; "
and appended to its source line as a `#` comment.
"""

import re
import reprlib
import signal
import sys
import threading

# Control-flow headers never emit NO_CHANGE. The paper's App. A.1 notes that
# sys.settrace() cannot observe branch conditions, and its Fig. 2 numbering
# confirms bare `if` lines produce no state at all (the counter advances twice
# per loop iteration, not three times). A `for` header still reports the loop
# variable it binds -- only the empty-delta case is suppressed.
_CONTROL_HEADER = re.compile(r"^\s*(if|elif|else|for|while|try|except|finally|with)\b")

MAX_VALUE_REPR = 120
MAX_STATES = 4000  # abort runaway loops before they hang the tracer
# A state cap alone cannot stop `while True: pass`: it fires line events forever
# while recording no state at all. Cap raw events too so tracing always halts.
MAX_EVENTS = 20_000

# Every line event re-reprs the frame's locals, so an unbounded repr() of a large
# collection is the dominant cost of tracing. reprlib truncates without first
# materialising the whole string, which is orders of magnitude faster on the
# handful of tasks that build big lists.
_BOUNDED = reprlib.Repr()
_BOUNDED.maxlevel = 4
_BOUNDED.maxlist = _BOUNDED.maxtuple = _BOUNDED.maxset = _BOUNDED.maxfrozenset = 20
_BOUNDED.maxdict = 12
_BOUNDED.maxstring = _BOUNDED.maxother = MAX_VALUE_REPR
_BOUNDED.maxlong = 40

TRACED_FILENAME = "<traced_program>"


DEFAULT_TRACE_TIMEOUT = 5.0


class TraceAborted(Exception):
    """Raised when a traced program exceeds its state, event, or time budget."""


class _WallClockGuard:
    """Abort tracing after `seconds`, even with no trace events firing.

    Neither MAX_STATES nor MAX_EVENTS can stop a single line that does unbounded
    work inside C code -- HumanEval/160 builds an expression string and eval()s
    it, which can be one enormous exponentiation. Only a real timer catches that.
    SIGALRM is main-thread-only, so this degrades to a no-op elsewhere.
    """

    def __init__(self, seconds):
        self.seconds = seconds
        self.armed = False

    def __enter__(self):
        if self.seconds and threading.current_thread() is threading.main_thread():
            def _fire(signum, frame):
                raise TraceAborted

            self._previous = signal.signal(signal.SIGALRM, _fire)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
            self.armed = True
        return self

    def __exit__(self, *exc):
        if self.armed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._previous)
        return False


def _fmt(value):
    try:
        rendered = _BOUNDED.repr(value)
    except Exception:
        return f"<unreprable {type(value).__name__}>"
    if len(rendered) > MAX_VALUE_REPR:
        rendered = rendered[: MAX_VALUE_REPR - 3] + "..."
    return rendered


def _snapshot(frame):
    """Render the frame's locals as {name: repr}. Comparing reprs rather than
    objects sidesteps values whose __eq__ is absent, expensive, or lying."""
    snap = {}
    for name, value in frame.f_locals.items():
        if name.startswith("__"):
            continue
        snap[name] = _fmt(value)
    return snap


def _changed(previous, current):
    return {k: v for k, v in current.items() if previous.get(k) != v}


class _Tracer:
    def __init__(self, source_lines):
        self.source_lines = source_lines
        self.states = []  # (lineno, order, text)
        self.order = 0
        self.events = 0
        self.frames = {}  # id(frame) -> {"line": int|None, "snap": dict, "raised": bool}

    def _is_control_header(self, lineno):
        if 1 <= lineno <= len(self.source_lines):
            return bool(_CONTROL_HEADER.match(self.source_lines[lineno - 1]))
        return False

    def _record(self, lineno, text):
        if self.order >= MAX_STATES:
            raise TraceAborted
        self.states.append((lineno, self.order, text))
        self.order += 1

    def _record_changes(self, lineno, previous, current):
        delta = _changed(previous, current)
        if delta:
            text = ", ".join(f"{k}={v}" for k, v in delta.items())
        elif self._is_control_header(lineno):
            return  # branch conditions produce no observable state
        else:
            # App. A.1: distinguishes "executed but changed nothing" from "never reached".
            text = "NO_CHANGE"
        self._record(lineno, text)

    def dispatch(self, frame, event, arg):
        if frame.f_code.co_filename != TRACED_FILENAME:
            return None

        self.events += 1
        if self.events > MAX_EVENTS:
            raise TraceAborted

        key = id(frame)

        if event == "call":
            snap = _snapshot(frame)
            # State (0) of a call is the bound arguments, attributed to the `def` line.
            if snap:
                text = ", ".join(f"{k}={v}" for k, v in snap.items())
                self._record(frame.f_code.co_firstlineno, text)
            self.frames[key] = {"line": None, "snap": snap, "raised": False}
            return self.dispatch

        state = self.frames.get(key)
        if state is None:
            return self.dispatch

        if event == "line":
            if state["line"] is not None:
                current = _snapshot(frame)
                self._record_changes(state["line"], state["snap"], current)
                state["snap"] = current
            state["line"] = frame.f_lineno

        elif event == "exception":
            exc_value = arg[1]
            self._record(frame.f_lineno, f"__exception__ = {_fmt(exc_value)}")
            state["raised"] = True
            state["line"] = None  # the raising line is already accounted for

        elif event == "return":
            if not state["raised"] and state["line"] is not None:
                self._record(state["line"], f"__return__={_fmt(arg)}")
            self.frames.pop(key, None)

        return self.dispatch


def _group_and_render(source_lines, states):
    """Attach state groups to their source lines as inline comments."""
    by_line = {}
    for lineno, order, text in states:
        by_line.setdefault(lineno, []).append((order, text))

    rendered = []
    for idx, line in enumerate(source_lines, start=1):
        group = by_line.get(idx)
        if not group:
            rendered.append(line)
            continue
        group.sort(key=lambda pair: pair[0])
        # App. A.1: |H_i| > 3 -> keep the first two states and the last one.
        if len(group) > 3:
            kept = [f"({o}) {t}" for o, t in group[:2]]
            kept.append("...")
            kept.append(f"({group[-1][0]}) {group[-1][1]}")
        else:
            kept = [f"({o}) {t}" for o, t in group]
        rendered.append(f"{line} # {'; '.join(kept)};")
    return "\n".join(rendered)


def trace(program_src, assertion_src, timeout=DEFAULT_TRACE_TIMEOUT):
    """Run `assertion_src` against `program_src` and return the inline trace.

    Returns (trace_text, outcome) where outcome is "pass", "fail", or "error".
    The program is compiled under TRACED_FILENAME so the tracer can ignore
    frames belonging to the harness or the standard library.
    """
    program_src = program_src.rstrip("\n")
    source_lines = program_src.split("\n")

    namespace = {"__name__": "__traced__"}
    try:
        exec(compile(program_src, TRACED_FILENAME, "exec"), namespace)
    except Exception as exc:
        # The program does not even import/define cleanly.
        return f"{program_src}\n\n\n{assertion_src}  # __exception__ = {_fmt(exc)}", "error"

    tracer = _Tracer(source_lines)
    assertion_exc = None
    aborted = False
    previous = sys.gettrace()
    try:
        with _WallClockGuard(timeout):
            sys.settrace(tracer.dispatch)
            exec(compile(assertion_src, "<assertion>", "exec"), namespace)
    except TraceAborted:
        aborted = True
    except Exception as exc:
        assertion_exc = exc
    finally:
        sys.settrace(previous)

    body = _group_and_render(source_lines, tracer.states)

    if aborted:
        # Ran past the step budget -- almost always a non-terminating loop in the
        # buggy program. Say so rather than silently reporting it as a pass.
        return f"{body}\n\n\n{assertion_src} # trace truncated: execution did not terminate", "timeout"

    if assertion_exc is None:
        outcome = "pass"
        assertion_line = assertion_src
    else:
        outcome = "fail" if isinstance(assertion_exc, AssertionError) else "error"
        # Only annotate the assert line when the failure surfaced *there*; an
        # exception raised inside the function is already shown on its own line.
        if isinstance(assertion_exc, AssertionError):
            assertion_line = (
                f"{assertion_src} # ({tracer.order}) __exception__ = {_fmt(assertion_exc)}"
            )
        else:
            assertion_line = assertion_src

    return f"{body}\n\n\n{assertion_line}", outcome


def trace_all(program_src, assertions, separator="-" * 20, timeout=DEFAULT_TRACE_TIMEOUT):
    """Trace a program against several assertions, in the paper's cheatsheet layout."""
    blocks = []
    outcomes = []
    for assertion in assertions:
        text, outcome = trace(program_src, assertion, timeout=timeout)
        blocks.append(text)
        outcomes.append(outcome)
    return f"\n{separator}\n".join(blocks), outcomes
