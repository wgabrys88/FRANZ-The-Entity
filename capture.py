"""Screenshot producer.

Produces a screenshot (real desktop or persistent sandbox canvas), applies
sandbox drawings (white, persistent) and cursor marks (red, ephemeral),
resizes to output dimensions, and returns base64 PNG plus applied action
list to execute.py via stdout JSON.

Cursor marks show the current and previous cursor positions with
normalized coordinate labels. Reads run_dir from the incoming JSON
request to locate sandbox_canvas.bmp and sandbox_state.json. Display
settings are read from config.py.
"""

from __future__ import annotations

import ast
import base64
import ctypes
import ctypes.wintypes
import json
import struct
import sys
import zlib
from datetime import datetime
from pathlib import Path
from typing import Final

import config as franz_config

Color = tuple[int, int, int, int]
Point = tuple[int, int]

MARK_SCALE: Final = 1.8
SANDBOX_LINE_THICKNESS: Final = 8
SANDBOX_CLICK_RADIUS: Final = 10
SANDBOX_RECT_HALF_W: Final = 10
SANDBOX_RECT_HALF_H: Final = 7
_SRCCOPY: Final = 0x00CC0020
_CAPTUREBLT: Final = 0x40000000
_BI_RGB: Final = 0
_DIB_RGB: Final = 0
_HALFTONE: Final = 4

MARK_TEXT: Final[Color] = (255, 255, 255, 255)
SANDBOX_WHITE: Final[Color] = (255, 255, 255, 255)

CURSOR_CURRENT_FILL: Final[Color] = (255, 0, 0, 220)
CURSOR_CURRENT_OUTLINE: Final[Color] = (255, 255, 255, 240)
CURSOR_PREV_FILL: Final[Color] = (255, 0, 0, 70)
CURSOR_PREV_OUTLINE: Final[Color] = (255, 255, 255, 80)
CURSOR_LABEL_BG: Final[Color] = (0, 0, 0, 180)
CURSOR_LABEL_TEXT: Final[Color] = (255, 255, 255, 255)
CURSOR_LABEL_TEXT_FADED: Final[Color] = (255, 255, 255, 90)

ctypes.WinDLL("shcore", use_last_error=True).SetProcessDpiAwareness(2)
_user32 = ctypes.WinDLL("user32", use_last_error=True)
_gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
_screen_w: Final = _user32.GetSystemMetrics(0)
_screen_h: Final = _user32.GetSystemMetrics(1)


def _ms(base: int) -> int:
    return max(1, int(base * MARK_SCALE))


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.wintypes.DWORD), ("biWidth", ctypes.wintypes.LONG),
        ("biHeight", ctypes.wintypes.LONG), ("biPlanes", ctypes.wintypes.WORD),
        ("biBitCount", ctypes.wintypes.WORD), ("biCompression", ctypes.wintypes.DWORD),
        ("biSizeImage", ctypes.wintypes.DWORD), ("biXPelsPerMeter", ctypes.wintypes.LONG),
        ("biYPelsPerMeter", ctypes.wintypes.LONG), ("biClrUsed", ctypes.wintypes.DWORD),
        ("biClrImportant", ctypes.wintypes.DWORD),
    ]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", _BITMAPINFOHEADER), ("bmiColors", ctypes.wintypes.DWORD * 3)]


def _make_bmi(w: int, h: int) -> _BITMAPINFO:
    bmi = _BITMAPINFO()
    hdr = bmi.bmiHeader
    hdr.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
    hdr.biWidth, hdr.biHeight = w, -h
    hdr.biPlanes, hdr.biBitCount, hdr.biCompression = 1, 32, _BI_RGB
    return bmi


def _capture_bgra(w: int, h: int) -> bytes:
    sdc = _user32.GetDC(0)
    memdc = _gdi32.CreateCompatibleDC(sdc)
    bits = ctypes.c_void_p()
    hbmp = _gdi32.CreateDIBSection(
        sdc, ctypes.byref(_make_bmi(w, h)), _DIB_RGB,
        ctypes.byref(bits), None, 0,
    )
    old = _gdi32.SelectObject(memdc, hbmp)
    try:
        _gdi32.BitBlt(memdc, 0, 0, w, h, sdc, 0, 0, _SRCCOPY | _CAPTUREBLT)
        return bytes((ctypes.c_ubyte * (w * h * 4)).from_address(bits.value))
    finally:
        _gdi32.SelectObject(memdc, old)
        _gdi32.DeleteObject(hbmp)
        _gdi32.DeleteDC(memdc)
        _user32.ReleaseDC(0, sdc)


def _resize_bgra(src: bytes, sw: int, sh: int, dw: int, dh: int) -> bytes:
    sdc = _user32.GetDC(0)
    src_dc = _gdi32.CreateCompatibleDC(sdc)
    dst_dc = _gdi32.CreateCompatibleDC(sdc)
    src_bmp = _gdi32.CreateCompatibleBitmap(sdc, sw, sh)
    old_src = _gdi32.SelectObject(src_dc, src_bmp)
    dst_bits = ctypes.c_void_p()
    dst_bmp = _gdi32.CreateDIBSection(
        sdc, ctypes.byref(_make_bmi(dw, dh)), _DIB_RGB,
        ctypes.byref(dst_bits), None, 0,
    )
    old_dst = _gdi32.SelectObject(dst_dc, dst_bmp)
    try:
        _gdi32.SetDIBits(sdc, src_bmp, 0, sh, src, ctypes.byref(_make_bmi(sw, sh)), _DIB_RGB)
        _gdi32.SetStretchBltMode(dst_dc, _HALFTONE)
        _gdi32.SetBrushOrgEx(dst_dc, 0, 0, None)
        _gdi32.StretchBlt(dst_dc, 0, 0, dw, dh, src_dc, 0, 0, sw, sh, _SRCCOPY)
        return bytes((ctypes.c_ubyte * (dw * dh * 4)).from_address(dst_bits.value))
    finally:
        _gdi32.SelectObject(dst_dc, old_dst)
        _gdi32.SelectObject(src_dc, old_src)
        _gdi32.DeleteObject(dst_bmp)
        _gdi32.DeleteObject(src_bmp)
        _gdi32.DeleteDC(dst_dc)
        _gdi32.DeleteDC(src_dc)
        _user32.ReleaseDC(0, sdc)


def _bgra_to_rgba(bgra: bytes) -> bytearray:
    n = len(bgra)
    out = bytearray(n)
    out[0::4] = bgra[2::4]
    out[1::4] = bgra[1::4]
    out[2::4] = bgra[0::4]
    out[3::4] = b"\xff" * (n // 4)
    return out


def _rgba_to_bgra(rgba: bytes) -> bytes:
    n = len(rgba)
    out = bytearray(n)
    out[0::4] = rgba[2::4]
    out[1::4] = rgba[1::4]
    out[2::4] = rgba[0::4]
    out[3::4] = b"\xff" * (n // 4)
    return bytes(out)


def _encode_png(rgba: bytes, w: int, h: int) -> bytes:
    stride = w * 4
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        raw.extend(rgba[y * stride:(y + 1) * stride])

    def chunk(tag: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(tag + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", crc)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


class Canvas:
    __slots__ = ("buf", "w", "h")

    def __init__(self, buf: bytearray, w: int, h: int) -> None:
        self.buf, self.w, self.h = buf, w, h

    def put(self, x: int, y: int, c: Color) -> None:
        if not (0 <= x < self.w and 0 <= y < self.h):
            return
        i = (y * self.w + x) << 2
        sa = c[3]
        if sa >= 255:
            self.buf[i], self.buf[i + 1], self.buf[i + 2], self.buf[i + 3] = c[0], c[1], c[2], 255
            return
        da = 255 - sa
        self.buf[i] = (c[0] * sa + self.buf[i] * da) // 255
        self.buf[i + 1] = (c[1] * sa + self.buf[i + 1] * da) // 255
        self.buf[i + 2] = (c[2] * sa + self.buf[i + 2] * da) // 255
        self.buf[i + 3] = 255

    def put_opaque(self, x: int, y: int, c: Color) -> None:
        if not (0 <= x < self.w and 0 <= y < self.h):
            return
        i = (y * self.w + x) << 2
        self.buf[i], self.buf[i + 1], self.buf[i + 2], self.buf[i + 3] = c[0], c[1], c[2], 255

    def _thick(self, x: int, y: int, c: Color, t: int, opaque: bool) -> None:
        half = t >> 1
        fn = self.put_opaque if opaque else self.put
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                fn(x + dx, y + dy, c)

    def _bresenham(self, x1: int, y1: int, x2: int, y2: int, c: Color, t: int, opaque: bool) -> None:
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        sx, sy = (1 if x1 < x2 else -1), (1 if y1 < y2 else -1)
        err, x, y = dx - dy, x1, y1
        while True:
            self._thick(x, y, c, t, opaque)
            if x == x2 and y == y2:
                break
            e2 = err << 1
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def line(self, x1: int, y1: int, x2: int, y2: int, c: Color, t: int) -> None:
        self._bresenham(x1, y1, x2, y2, c, t, False)

    def line_opaque(self, x1: int, y1: int, x2: int, y2: int, c: Color, t: int) -> None:
        self._bresenham(x1, y1, x2, y2, c, t, True)

    def circle_opaque(self, cx: int, cy: int, r: int, c: Color) -> None:
        r2 = r * r
        for oy in range(-r, r + 1):
            for ox in range(-r, r + 1):
                if ox * ox + oy * oy <= r2:
                    self.put_opaque(cx + ox, cy + oy, c)

    def rect_opaque(self, x: int, y: int, w: int, h: int, c: Color) -> None:
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                self.put_opaque(xx, yy, c)

    def fill_polygon(self, pts: list[Point], c: Color) -> None:
        if len(pts) < 3:
            return
        ys = [p[1] for p in pts]
        n = len(pts)
        for y in range(max(0, min(ys)), min(self.h - 1, max(ys)) + 1):
            nodes: list[int] = []
            j = n - 1
            for i in range(n):
                yi, yj = pts[i][1], pts[j][1]
                if (yi < y <= yj) or (yj < y <= yi):
                    nodes.append(int(pts[i][0] + (y - yi) / (yj - yi) * (pts[j][0] - pts[i][0])))
                j = i
            nodes.sort()
            for k in range(0, len(nodes) - 1, 2):
                for x in range(max(0, nodes[k]), min(self.w - 1, nodes[k + 1]) + 1):
                    self.put(x, y, c)

    def rect_fill(self, x: int, y: int, w: int, h: int, c: Color) -> None:
        for yy in range(max(0, y), min(self.h, y + h)):
            for xx in range(max(0, x), min(self.w, x + w)):
                self.put(xx, yy, c)


_FONT_5X7: Final[dict[str, list[int]]] = {
    " ": [0, 0, 0, 0, 0, 0, 0],
    "0": [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
    "1": [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    "2": [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
    "3": [0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110],
    "4": [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
    "5": [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
    "6": [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
    "7": [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
    "8": [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
    "9": [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
    "A": [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    "B": [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
    "C": [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
    "D": [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
    "E": [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
    "F": [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
    "G": [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
    "H": [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    "I": [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    "J": [0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100],
    "K": [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
    "L": [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
    "M": [0b10001, 0b11011, 0b10101, 0b10001, 0b10001, 0b10001, 0b10001],
    "N": [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
    "O": [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    "P": [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
    "Q": [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
    "R": [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
    "S": [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
    "T": [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
    "U": [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    "V": [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
    "W": [0b10001, 0b10001, 0b10001, 0b10001, 0b10101, 0b11011, 0b10001],
    "X": [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
    "Y": [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
    "Z": [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
    ".": [0, 0, 0, 0, 0, 0b00100, 0b00100],
    ",": [0, 0, 0, 0, 0b00100, 0b00100, 0b01000],
    "!": [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0, 0b00100],
    "?": [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0, 0b00100],
    "-": [0, 0, 0, 0b11111, 0, 0, 0],
    ":": [0, 0b00100, 0b00100, 0, 0b00100, 0b00100, 0],
    "/": [0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0, 0],
    "(": [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
    ")": [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
}

_CURSOR_SHAPE: Final[list[Point]] = [
    (0, 0), (0, 20), (5, 16), (9, 24), (12, 23), (8, 15), (14, 14), (0, 0),
]


def _draw_text(cv: Canvas, x: int, y: int, text: str, c: Color, scale: int) -> None:
    px, py = x, y
    for ch in text:
        if ch == "\n":
            py += 8 * scale
            px = x
            continue
        pat = _FONT_5X7.get(ch.upper())
        if pat is None:
            cv.rect_opaque(px, py, 5 * scale, 7 * scale, c)
        else:
            for row in range(7):
                bits = pat[row]
                for col in range(5):
                    if bits & (1 << (4 - col)):
                        for sy in range(scale):
                            for sx in range(scale):
                                cv.put_opaque(px + col * scale + sx, py + row * scale + sy, c)
        px += 6 * scale


def _draw_text_alpha(cv: Canvas, x: int, y: int, text: str, c: Color, scale: int) -> None:
    px, py = x, y
    for ch in text:
        if ch == "\n":
            py += 8 * scale
            px = x
            continue
        pat = _FONT_5X7.get(ch.upper())
        if pat is None:
            cv.rect_fill(px, py, 5 * scale, 7 * scale, c)
        else:
            for row in range(7):
                bits = pat[row]
                for col in range(5):
                    if bits & (1 << (4 - col)):
                        for sy in range(scale):
                            for sx in range(scale):
                                cv.put(px + col * scale + sx, py + row * scale + sy, c)
        px += 6 * scale


def _draw_cursor_icon(cv: Canvas, tip_x: int, tip_y: int, fill: Color, outline: Color, scale: float) -> None:
    pts = [(int(tip_x + p[0] * scale), int(tip_y + p[1] * scale)) for p in _CURSOR_SHAPE]
    for i in range(len(pts) - 1):
        cv.line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], outline, max(1, int(2 * scale)))
    cv.fill_polygon(pts, fill)


def _draw_cursor_label(
    cv: Canvas, tip_x: int, tip_y: int, norm_x: int, norm_y: int,
    bg: Color, text_color: Color, scale: int, cursor_scale: float,
) -> None:
    label = f"({norm_x},{norm_y})"
    char_w = 6 * scale
    char_h = 7 * scale
    lw = len(label) * char_w
    lh = char_h
    pad_x, pad_y = scale * 3, scale * 2
    total_w = lw + pad_x * 2
    total_h = lh + pad_y * 2
    cursor_h = int(24 * cursor_scale)
    cursor_w = int(14 * cursor_scale)

    if tip_x + cursor_w + total_w + 4 <= cv.w:
        lx = tip_x + cursor_w + 4
    elif tip_x - total_w - 4 >= 0:
        lx = tip_x - total_w - 4
    else:
        lx = max(0, min(cv.w - total_w, tip_x - total_w // 2))

    if tip_y + cursor_h + 4 + total_h <= cv.h:
        ly = tip_y + cursor_h + 4
    elif tip_y - total_h - 4 >= 0:
        ly = tip_y - total_h - 4
    else:
        ly = max(0, min(cv.h - total_h, tip_y))

    cv.rect_fill(lx, ly, total_w, total_h, bg)
    if text_color[3] >= 250:
        _draw_text(cv, lx + pad_x, ly + pad_y, label, text_color, scale)
    else:
        _draw_text_alpha(cv, lx + pad_x, ly + pad_y, label, text_color, scale)


def _parse_action(line: str) -> tuple[str, list[object], dict[str, object]] | None:
    s = line.strip()
    if not s:
        return None
    try:
        node = ast.parse(s, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None
    args: list[object] = []
    for a in node.args:
        if not isinstance(a, ast.Constant):
            return None
        args.append(a.value)
    kwargs: dict[str, object] = {}
    for kw in node.keywords:
        if kw.arg is None or not isinstance(kw.value, ast.Constant):
            return None
        kwargs[kw.arg] = kw.value.value
    return node.func.id, args, kwargs


def _arg_int(args: list[object], kw: dict[str, object], idx: int, key: str) -> int | None:
    v = args[idx] if idx < len(args) else kw.get(key)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _arg_str(args: list[object], kw: dict[str, object], idx: int, key: str) -> str | None:
    v = args[idx] if idx < len(args) else kw.get(key)
    return str(v) if v is not None else None


def _norm(v: int, extent: int) -> int:
    return int((max(0, min(1000, v)) / 1000.0) * extent)


def _denorm(px: int, extent: int) -> int:
    if extent <= 0:
        return 0
    return int((px / extent) * 1000.0)


def _bmp_write_black(path: Path, w: int, h: int) -> None:
    stride = ((w * 3 + 3) // 4) * 4
    si = stride * h
    hdr = struct.pack("<2sIHHI", b"BM", 54 + si, 0, 0, 54)
    ihdr = struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, si, 2835, 2835, 0, 0)
    row = b"\x00" * stride
    _atomic_write(path, hdr + ihdr + row * h)


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)


def _bmp_load_rgba(path: Path, w: int, h: int) -> bytearray:
    try:
        data = path.read_bytes()
        if len(data) < 54 or data[:2] != b"BM":
            return bytearray()
        off = struct.unpack_from("<I", data, 10)[0]
        if struct.unpack_from("<I", data, 14)[0] < 40:
            return bytearray()
        bw, bh = struct.unpack_from("<ii", data, 18)
        planes, bpp = struct.unpack_from("<HH", data, 26)
        comp = struct.unpack_from("<I", data, 30)[0]
        if planes != 1 or comp != 0 or bpp not in (24, 32):
            return bytearray()
        ah = abs(bh)
        if bw != w or ah != h:
            return bytearray()
        bytespp = bpp // 8
        stride = ((w * bytespp + 3) // 4) * 4
        if len(data) < off + stride * h:
            return bytearray()
        out = bytearray(w * h * 4)
        top_down = bh < 0
        for y in range(h):
            sy = y if top_down else (h - 1 - y)
            row = data[off + sy * stride: off + (sy + 1) * stride]
            di = y * w * 4
            for x in range(w):
                si2 = x * bytespp
                out[di + x * 4] = row[si2 + 2]
                out[di + x * 4 + 1] = row[si2 + 1]
                out[di + x * 4 + 2] = row[si2]
                out[di + x * 4 + 3] = 255
        return out
    except Exception:
        return bytearray()


def _bmp_save_rgba(path: Path, buf: bytes, w: int, h: int) -> None:
    stride = ((w * 3 + 3) // 4) * 4
    si = stride * h
    pad = b"\x00" * (stride - w * 3)
    out = bytearray()
    out.extend(struct.pack("<2sIHHI", b"BM", 54 + si, 0, 0, 54))
    out.extend(struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, si, 2835, 2835, 0, 0))
    for y in range(h - 1, -1, -1):
        row = buf[y * w * 4: (y + 1) * w * 4]
        for x in range(w):
            i = x * 4
            out.append(row[i + 2])
            out.append(row[i + 1])
            out.append(row[i])
        out.extend(pad)
    _atomic_write(path, bytes(out))


def _sandbox_state_load(path: Path) -> dict[str, int | None]:
    try:
        o = json.loads(path.read_text(encoding="utf-8"))
        result: dict[str, int | None] = {"last_x": None, "last_y": None, "prev_x": None, "prev_y": None}
        for key in result:
            v = o.get(key)
            if isinstance(v, int):
                result[key] = v
        return result
    except Exception:
        return {"last_x": None, "last_y": None, "prev_x": None, "prev_y": None}


def _sandbox_state_save(path: Path, st: dict[str, int | None]) -> None:
    _atomic_write_text(path, json.dumps(st))


def _sandbox_load(canvas_path: Path, w: int, h: int) -> bytearray:
    if not canvas_path.is_file():
        _bmp_write_black(canvas_path, w, h)
    buf = _bmp_load_rgba(canvas_path, w, h)
    if not buf:
        _bmp_write_black(canvas_path, w, h)
        return bytearray(b"\x00\x00\x00\xff" * (w * h))
    return buf


def _sandbox_apply(
    buf: bytearray, w: int, h: int, actions: list[str], state_path: Path,
) -> tuple[bool, list[str]]:
    cv = Canvas(buf, w, h)
    dirty = False
    applied: list[str] = []
    st = _sandbox_state_load(state_path)
    st["prev_x"] = st["last_x"]
    st["prev_y"] = st["last_y"]

    for line in actions:
        parsed = _parse_action(line)
        if parsed is None:
            continue
        name, args, kw = parsed
        if name == "click":
            name = "left_click"
        if name == "timestamp":
            continue

        if name == "drag":
            x1, y1 = _arg_int(args, kw, 0, "x1"), _arg_int(args, kw, 1, "y1")
            x2, y2 = _arg_int(args, kw, 2, "x2"), _arg_int(args, kw, 3, "y2")
            if x1 is None or y1 is None or x2 is None or y2 is None:
                continue
            px1, py1 = _norm(x1, w), _norm(y1, h)
            px2, py2 = _norm(x2, w), _norm(y2, h)
            cv.line_opaque(px1, py1, px2, py2, SANDBOX_WHITE, SANDBOX_LINE_THICKNESS)
            st["last_x"], st["last_y"] = px2, py2
            dirty = True
            applied.append(line)
        elif name in ("left_click", "double_left_click"):
            x, y = _arg_int(args, kw, 0, "x"), _arg_int(args, kw, 1, "y")
            if x is None or y is None:
                continue
            px, py = _norm(x, w), _norm(y, h)
            cv.circle_opaque(px, py, SANDBOX_CLICK_RADIUS, SANDBOX_WHITE)
            st["last_x"], st["last_y"] = px, py
            dirty = True
            applied.append(line)
        elif name == "right_click":
            x, y = _arg_int(args, kw, 0, "x"), _arg_int(args, kw, 1, "y")
            if x is None or y is None:
                continue
            px, py = _norm(x, w), _norm(y, h)
            cv.rect_opaque(
                px - SANDBOX_RECT_HALF_W, py - SANDBOX_RECT_HALF_H,
                SANDBOX_RECT_HALF_W * 2, SANDBOX_RECT_HALF_H * 2, SANDBOX_WHITE,
            )
            st["last_x"], st["last_y"] = px, py
            dirty = True
            applied.append(line)
        elif name == "type":
            t = _arg_str(args, kw, 0, "text")
            if t is None:
                continue
            lx, ly = st.get("last_x"), st.get("last_y")
            if not isinstance(lx, int) or not isinstance(ly, int):
                continue
            cv.line_opaque(lx + 6, ly - 7, lx + 6, ly + 7, SANDBOX_WHITE, 2)
            _draw_text(cv, lx + 9, ly - 6, t, SANDBOX_WHITE, 2)
            dirty = True
            applied.append(line)

    if dirty:
        _sandbox_state_save(state_path, st)
    return dirty, applied


def _apply_marks_cursor(
    buf: bytearray, w: int, h: int, actions: list[str], state_path: Path,
) -> None:
    cv = Canvas(buf, w, h)
    st = _sandbox_state_load(state_path)
    cursor_scale = MARK_SCALE * 1.5
    label_scale = _ms(2)

    cur_x, cur_y = st.get("last_x"), st.get("last_y")
    prev_x, prev_y = st.get("prev_x"), st.get("prev_y")

    if isinstance(prev_x, int) and isinstance(prev_y, int):
        norm_px = _denorm(prev_x, w)
        norm_py = _denorm(prev_y, h)
        _draw_cursor_icon(cv, prev_x, prev_y, CURSOR_PREV_FILL, CURSOR_PREV_OUTLINE, cursor_scale)
        _draw_cursor_label(
            cv, prev_x, prev_y, norm_px, norm_py,
            (0, 0, 0, 100), CURSOR_LABEL_TEXT_FADED, label_scale, cursor_scale,
        )

    if isinstance(cur_x, int) and isinstance(cur_y, int):
        norm_cx = _denorm(cur_x, w)
        norm_cy = _denorm(cur_y, h)
        _draw_cursor_icon(cv, cur_x, cur_y, CURSOR_CURRENT_FILL, CURSOR_CURRENT_OUTLINE, cursor_scale)
        _draw_cursor_label(
            cv, cur_x, cur_y, norm_cx, norm_cy,
            CURSOR_LABEL_BG, CURSOR_LABEL_TEXT, label_scale, cursor_scale,
        )

    for line in actions:
        parsed = _parse_action(line)
        if parsed is None:
            continue
        name = parsed[0]
        if name == "timestamp":
            stamp = datetime.now().strftime("CAPTURED %Y-%m-%d %H:%M:%S")
            sc = _ms(2)
            cw = 6 * sc
            ch = 7 * sc
            stw = len(stamp) * cw
            tx, ty = (w - stw) // 2, int(h * 0.85)
            padx, pady = _ms(6), _ms(4)
            backdrop: Color = (0, 0, 0, 140)
            for yy in range(ty - pady, ty + ch + pady):
                for xx in range(tx - padx, tx + stw + padx):
                    cv.put(xx, yy, backdrop)
            _draw_text(cv, tx, ty, stamp, MARK_TEXT, sc)
            break


def capture(actions: list[str], run_dir: str) -> tuple[str, list[str]]:
    sw, sh = _screen_w, _screen_h
    sandbox = bool(franz_config.SANDBOX)
    marks = bool(franz_config.VISUAL_MARKS)
    width = int(franz_config.WIDTH)
    height = int(franz_config.HEIGHT)

    applied = list(actions)
    state_path = Path(run_dir) / "sandbox_state.json" if run_dir else None

    if sandbox:
        rd = Path(run_dir)
        canvas_path = rd / "sandbox_canvas.bmp"
        sp = rd / "sandbox_state.json"
        state_path = sp
        base = _sandbox_load(canvas_path, sw, sh)
        dirty, applied = _sandbox_apply(base, sw, sh, actions, sp)
        if dirty:
            _bmp_save_rgba(canvas_path, bytes(base), sw, sh)
        rgba = bytearray(base)
    else:
        rgba = _bgra_to_rgba(_capture_bgra(sw, sh))

    if marks and actions and state_path is not None:
        _apply_marks_cursor(rgba, sw, sh, actions, state_path)

    dw = sw if width <= 0 else width
    dh = sh if height <= 0 else height
    if (dw, dh) != (sw, sh):
        rgba = _bgra_to_rgba(_resize_bgra(_rgba_to_bgra(bytes(rgba)), sw, sh, dw, dh))

    return base64.b64encode(_encode_png(bytes(rgba), dw, dh)).decode("ascii"), applied


def main() -> None:
    req = json.loads(sys.stdin.read() or "{}")
    raw_actions = req.get("actions", [])
    actions = [str(a) for a in raw_actions] if isinstance(raw_actions, list) else []
    run_dir = str(req.get("run_dir", ""))
    b64, applied = capture(actions, run_dir)
    sys.stdout.write(json.dumps({"screenshot_b64": b64, "applied": applied}))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
