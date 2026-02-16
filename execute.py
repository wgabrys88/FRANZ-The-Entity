"""Action executor.

Extracts the first python fenced block from VLM output, parses it into
individual AST statements, executes each one independently in a sandboxed
namespace with tool functions, records per-statement results, optionally
sends Win32 input, delegates to capture.py for screenshot, and returns
structured JSON with ordered per-action feedback via stdout.

Tool discovery is self-service via help() in the namespace. Coordinates
outside 0-1000 raise ValueError instead of silent clamping. All settings
are read from config.py.
"""

import ast
import ctypes
import ctypes.wintypes
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Final

import config as franz_config

_MOVE_STEPS: Final = 20
_STEP_DELAY: Final = 0.01
_CLICK_DELAY: Final = 0.12
CAPTURE_SCRIPT: Final = Path(__file__).parent / "capture.py"

INPUT_MOUSE: Final = 0
INPUT_KEYBOARD: Final = 1
MOUSEEVENTF_LEFTDOWN: Final = 0x0002
MOUSEEVENTF_LEFTUP: Final = 0x0004
MOUSEEVENTF_RIGHTDOWN: Final = 0x0008
MOUSEEVENTF_RIGHTUP: Final = 0x0010
MOUSEEVENTF_MOVE: Final = 0x0001
MOUSEEVENTF_ABSOLUTE: Final = 0x8000
KEYEVENTF_KEYUP: Final = 0x0002
KEYEVENTF_UNICODE: Final = 0x0004

ULONG_PTR = ctypes.c_size_t


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long), ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong), ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("u", _INPUTUNION)]


_user32: ctypes.WinDLL | None = None
_screen_w: int = 0
_screen_h: int = 0


def _init_win32() -> None:
    global _user32, _screen_w, _screen_h
    if _user32 is not None:
        return
    ctypes.WinDLL("shcore", use_last_error=True).SetProcessDpiAwareness(2)
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _screen_w = _user32.GetSystemMetrics(0)
    _screen_h = _user32.GetSystemMetrics(1)
    _user32.SendInput.argtypes = (ctypes.c_uint, ctypes.POINTER(INPUT), ctypes.c_int)
    _user32.SendInput.restype = ctypes.c_uint


def _send_inputs(items: list[INPUT]) -> None:
    if not items:
        return
    assert _user32 is not None
    arr = (INPUT * len(items))(*items)
    if _user32.SendInput(len(items), arr, ctypes.sizeof(INPUT)) != len(items):
        raise OSError(ctypes.get_last_error())


def _send_mouse(flags: int, abs_x: int | None = None, abs_y: int | None = None) -> None:
    i = INPUT()
    i.type = INPUT_MOUSE
    f = flags
    dx = dy = 0
    if abs_x is not None and abs_y is not None:
        dx, dy, f = abs_x, abs_y, f | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE
    i.u.mi = MOUSEINPUT(dx, dy, 0, f, 0, 0)
    _send_inputs([i])


def _send_unicode(text: str) -> None:
    items: list[INPUT] = []
    for ch in text:
        if ch == "\r":
            continue
        code = 0x000D if ch == "\n" else ord(ch)
        for fl in (KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP):
            inp = INPUT()
            inp.type = INPUT_KEYBOARD
            inp.u.ki = KEYBDINPUT(0, code, fl, 0, 0)
            items.append(inp)
    _send_inputs(items)


def _to_px(v: int, dim: int) -> int:
    return int((max(0, min(1000, v)) / 1000) * dim)


def _to_abs(x_px: int, y_px: int) -> tuple[int, int]:
    return (
        max(0, min(65535, int((x_px / max(1, _screen_w - 1)) * 65535))),
        max(0, min(65535, int((y_px / max(1, _screen_h - 1)) * 65535))),
    )


def _smooth_move(tx: int, ty: int) -> None:
    assert _user32 is not None
    pt = ctypes.wintypes.POINT()
    _user32.GetCursorPos(ctypes.byref(pt))
    sx, sy = pt.x, pt.y
    dx, dy = tx - sx, ty - sy
    for i in range(_MOVE_STEPS + 1):
        t = i / _MOVE_STEPS
        t = t * t * (3.0 - 2.0 * t)
        _send_mouse(0, *_to_abs(int(sx + dx * t), int(sy + dy * t)))
        time.sleep(_STEP_DELAY)


def _mouse_click(down: int, up: int) -> None:
    _send_mouse(down)
    time.sleep(0.02)
    _send_mouse(up)


def _do_left_click(x: int, y: int) -> None:
    _smooth_move(_to_px(x, _screen_w), _to_px(y, _screen_h))
    time.sleep(_CLICK_DELAY)
    _mouse_click(MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP)


def _do_right_click(x: int, y: int) -> None:
    _smooth_move(_to_px(x, _screen_w), _to_px(y, _screen_h))
    time.sleep(_CLICK_DELAY)
    _mouse_click(MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP)


def _do_double_left_click(x: int, y: int) -> None:
    _do_left_click(x, y)
    time.sleep(0.06)
    _do_left_click(x, y)


def _do_drag(x1: int, y1: int, x2: int, y2: int) -> None:
    _smooth_move(_to_px(x1, _screen_w), _to_px(y1, _screen_h))
    time.sleep(0.08)
    _send_mouse(MOUSEEVENTF_LEFTDOWN)
    time.sleep(0.06)
    _smooth_move(_to_px(x2, _screen_w), _to_px(y2, _screen_h))
    time.sleep(0.06)
    _send_mouse(MOUSEEVENTF_LEFTUP)


_FENCE_RE: Final = re.compile(r"```python[ \t]*\n(.*?)```", re.DOTALL)
_FENCE_ALL_RE: Final = re.compile(r"```python[ \t]*\n.*?```", re.DOTALL)


def _extract_block(raw: str) -> tuple[str | None, int]:
    block_count = len(_FENCE_ALL_RE.findall(raw))
    m = _FENCE_RE.search(raw)
    return (m.group(1) if m else None), block_count


def _validate_coord(name: str, v: object) -> int:
    if not isinstance(v, int | float):
        raise TypeError(f"{name} must be a number, got {type(v).__name__}")
    iv = int(v)
    if iv < 0 or iv > 1000:
        raise ValueError(f"{name}={iv} out of range 0-1000")
    return iv


def _make_namespace(
    master: bool, physical: bool, restricted: bool,
) -> tuple[dict[str, object], list[str], list[str]]:
    executed: list[str] = []
    ignored: list[str] = []

    def gate() -> bool:
        return master

    def left_click(x: int, y: int) -> None:
        """left_click(x, y) -- white dot at position"""
        ix = _validate_coord("x", x)
        iy = _validate_coord("y", y)
        canon = f"left_click({ix}, {iy})"
        if not gate():
            ignored.append(canon)
            return
        if physical:
            _do_left_click(ix, iy)
        executed.append(canon)

    def right_click(x: int, y: int) -> None:
        """right_click(x, y) -- small white square at position"""
        ix = _validate_coord("x", x)
        iy = _validate_coord("y", y)
        canon = f"right_click({ix}, {iy})"
        if not gate():
            ignored.append(canon)
            return
        if physical:
            _do_right_click(ix, iy)
        executed.append(canon)

    def double_left_click(x: int, y: int) -> None:
        """double_left_click(x, y) -- white dot at position"""
        ix = _validate_coord("x", x)
        iy = _validate_coord("y", y)
        canon = f"double_left_click({ix}, {iy})"
        if not gate():
            ignored.append(canon)
            return
        if physical:
            _do_double_left_click(ix, iy)
        executed.append(canon)

    def drag(x1: int, y1: int, x2: int, y2: int) -> None:
        """drag(x1, y1, x2, y2) -- straight white line from (x1,y1) to (x2,y2)"""
        ix1 = _validate_coord("x1", x1)
        iy1 = _validate_coord("y1", y1)
        ix2 = _validate_coord("x2", x2)
        iy2 = _validate_coord("y2", y2)
        canon = f"drag({ix1}, {iy1}, {ix2}, {iy2})"
        if not gate():
            ignored.append(canon)
            return
        if physical:
            _do_drag(ix1, iy1, ix2, iy2)
        executed.append(canon)

    def type_(text: str) -> None:
        """type(text) -- type text at last click position"""
        if not isinstance(text, str):
            raise TypeError(f"type() requires a string, got {type(text).__name__}")
        canon = f"type({json.dumps(text)})"
        if not gate():
            ignored.append(canon)
            return
        if physical:
            _send_unicode(text)
        executed.append(canon)

    def click(x: int, y: int) -> None:
        """click(x, y) -- alias for left_click"""
        ix = _validate_coord("x", x)
        iy = _validate_coord("y", y)
        canon = f"click({ix}, {iy})"
        if not gate():
            ignored.append(canon)
            return
        if physical:
            _do_left_click(ix, iy)
        executed.append(canon)

    ns: dict[str, object] = {
        "left_click": left_click,
        "right_click": right_click,
        "double_left_click": double_left_click,
        "drag": drag,
        "type": type_,
        "click": click,
    }

    def help(fn: object = None) -> str:
        """help() -- list available functions or help(fn) for one function"""
        if fn is not None:
            if callable(fn):
                doc = getattr(fn, "__doc__", None)
                return doc if doc else f"{getattr(fn, '__name__', str(fn))}()"
            return f"help() takes a function, not {type(fn).__name__}"
        lines = []
        for name in sorted(ns):
            obj = ns[name]
            if callable(obj) and not name.startswith("_"):
                doc = getattr(obj, "__doc__", None)
                lines.append(doc if doc else f"{name}()")
        return " | ".join(lines)

    ns["help"] = help

    if restricted:
        ns["__builtins__"] = {}

    return ns, executed, ignored


def _fname(canon: str) -> str:
    idx = canon.find("(")
    return canon[:idx] if idx != -1 else canon


def _exec_statements(
    code: str, ns: dict[str, object],
) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        results.append(("script", f"SyntaxError: {e.msg}"))
        return results
    for node in tree.body:
        fname = ""
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name):
                fname = node.value.func.id
        if not fname:
            source = ast.unparse(node)
            fname = source.split("(")[0] if "(" in source else source
        try:
            if isinstance(node, ast.Expr):
                expr_node = ast.Expression(body=node.value)
                ast.fix_missing_locations(expr_node)
                result = eval(compile(expr_node, "<string>", "eval"), ns)
                if result is not None:
                    results.append((fname, str(result)))
                else:
                    results.append((fname, "OK"))
            else:
                stmt_mod = ast.Module(body=[node], type_ignores=[])
                ast.fix_missing_locations(stmt_mod)
                exec(compile(stmt_mod, "<string>", "exec"), ns)
                results.append((fname, "OK"))
        except Exception:
            typ, val, _ = sys.exc_info()
            type_name = typ.__name__ if typ else "Error"
            results.append((fname, f"{type_name}: {val}"))
    return results


def _build_feedback(
    stmt_results: list[tuple[str, str]],
    no_block: bool,
    block_count: int,
) -> str:
    hint = "Use help() to list available functions."

    if no_block:
        return f"No actions performed.\n{hint}"

    if not stmt_results:
        return f"No actions performed.\n{hint}"

    parts: list[str] = []
    for fname, status in stmt_results:
        parts.append(f"{fname} -> {status}")

    if block_count > 1:
        parts.append("ADHERE TO YOUR INSTRUCTIONS")

    return "\n".join(parts)


def _run_capture(actions: list[str], run_dir: str) -> tuple[str, list[str]]:
    r = subprocess.run(
        [sys.executable, str(CAPTURE_SCRIPT)],
        input=json.dumps({"actions": actions, "run_dir": run_dir}),
        capture_output=True,
        text=True,
    )
    if not r.stdout:
        return "", actions
    try:
        obj = json.loads(r.stdout)
        applied = obj.get("applied", actions)
        if not isinstance(applied, list):
            applied = actions
        return str(obj.get("screenshot_b64", "")), [str(a) for a in applied]
    except json.JSONDecodeError:
        return "", actions


def main() -> None:
    req = json.loads(sys.stdin.read() or "{}")
    raw = str(req.get("raw", ""))
    run_dir = str(req.get("run_dir", ""))

    sandbox = bool(franz_config.SANDBOX)
    master = bool(franz_config.EXECUTE_ACTIONS)
    physical = bool(franz_config.PHYSICAL_EXECUTION) and not sandbox
    restricted = bool(franz_config.RESTRICTED_EXEC)

    if physical:
        _init_win32()

    code, block_count = _extract_block(raw)
    no_block = code is None
    ns, executed, ignored = _make_namespace(master, physical, restricted)

    stmt_results: list[tuple[str, str]] = []
    if code is not None:
        stmt_results = _exec_statements(code, ns)

    screenshot_b64, applied = _run_capture(
        executed + ["timestamp()"], run_dir,
    )

    if sandbox:
        applied_set = set(applied)
        not_applied = [a for a in executed if a not in applied_set]
        if not_applied:
            for a in not_applied:
                fn = _fname(a)
                for i, (f, s) in enumerate(stmt_results):
                    if f == fn and s == "OK":
                        stmt_results[i] = (f, f"RuntimeError: {fn} had no visible effect")
                        break
            executed[:] = [a for a in executed if a in applied_set]

    feedback = _build_feedback(stmt_results, no_block, block_count)

    sys.stdout.write(json.dumps({
        "executed": executed,
        "malformed": [s for _, s in stmt_results if s != "OK"],
        "ignored": ignored,
        "screenshot_b64": screenshot_b64,
        "feedback": feedback,
    }))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
