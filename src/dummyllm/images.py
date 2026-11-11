"""Procedural cat pictures (pure-Python PNG encoder) and image header parsing."""

from __future__ import annotations

import base64
import colorsys
import random
import re
import struct
import zlib

# Images are rasterised at this resolution (on the longer side) and then
# nearest-neighbour upscaled, so even 1792x1024 renders in milliseconds.
_BASE_RES = 96

MAX_IMAGE_SIDE = 4096


def parse_size(size: str | None, default: tuple[int, int] = (1024, 1024)) -> tuple[int, int] | None:
    """Parse ``"1024x768"``; ``None``/``"auto"`` -> default; invalid -> ``None``."""
    if size is None or size == "auto":
        return default
    m = re.fullmatch(r"\s*(\d{1,5})\s*[xX]\s*(\d{1,5})\s*", size)
    if not m:
        return None
    w, h = int(m.group(1)), int(m.group(2))
    if not (16 <= w <= MAX_IMAGE_SIDE and 16 <= h <= MAX_IMAGE_SIDE):
        return None
    return w, h


def _hsv(h: float, s: float, v: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t), int(a[2] + (b[2] - a[2]) * t))


def _in_triangle(px: float, py: float, a, b, c) -> bool:
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    p = (px, py)
    d1, d2, d3 = sign(p, a, b), sign(p, b, c), sign(p, c, a)
    has_neg = d1 < 0 or d2 < 0 or d3 < 0
    has_pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (has_neg and has_pos)


def _near_segment(px: float, py: float, a, b, width: float) -> bool:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    cx, cy = ax + t * dx, ay + t * dy
    return (px - cx) ** 2 + (py - cy) ** 2 <= width * width


def render_cat(width: int, height: int, rng: random.Random) -> bytes:
    """Render a random cat face as a PNG of exactly ``width`` x ``height``."""
    # Rasterise with ~_BASE_RES px on the shorter side (never more than the
    # requested size, and capped for extreme aspect ratios).
    scale = min(1.0, _BASE_RES / min(width, height), (4 * _BASE_RES) / max(width, height))
    bw, bh = max(8, round(width * scale)), max(8, round(height * scale))

    hue = rng.random()
    bg_top = _hsv(hue, 0.25 + rng.random() * 0.2, 0.95)
    bg_bottom = _hsv(hue + 0.08 + rng.random() * 0.2, 0.35 + rng.random() * 0.2, 0.85)
    fur_choices = [
        (240, 160, 70), (60, 60, 60), (230, 230, 230), (170, 120, 80),
        (120, 120, 130), (250, 200, 140), (35, 35, 40),
    ]
    fur = rng.choice(fur_choices)
    fur_dark = _mix(fur, (0, 0, 0), 0.25)
    inner_ear = (240, 170, 180)
    eye = rng.choice([(110, 200, 80), (230, 200, 50), (80, 160, 230), (200, 140, 40)])
    pupil = (15, 15, 20)
    nose = (230, 120, 140)
    whisker = (250, 250, 250) if sum(fur) < 300 else (40, 40, 40)
    striped = rng.random() < 0.5

    # Geometry in normalised units relative to the shorter side, centred.
    unit = min(bw, bh)
    cx, cy = bw / 2, bh / 2 + unit * 0.06
    r = unit * (0.30 + rng.random() * 0.04)
    ear_h = r * (0.75 + rng.random() * 0.25)
    left_ear = ((cx - r * 0.95, cy - r * 0.35), (cx - r * 0.25, cy - r * 0.85), (cx - r * 0.85, cy - r - ear_h * 0.6))
    right_ear = ((cx + r * 0.95, cy - r * 0.35), (cx + r * 0.25, cy - r * 0.85), (cx + r * 0.85, cy - r - ear_h * 0.6))

    def shrink(tri, k):
        mx = sum(p[0] for p in tri) / 3
        my = sum(p[1] for p in tri) / 3
        return tuple((mx + (p[0] - mx) * k, my + (p[1] - my) * k) for p in tri)

    left_inner, right_inner = shrink(left_ear, 0.55), shrink(right_ear, 0.55)
    eye_dx, eye_y = r * 0.40, cy - r * 0.12
    eye_rx, eye_ry = r * 0.17, r * (0.12 + rng.random() * 0.08)
    pupil_rx = eye_rx * (0.25 + rng.random() * 0.3)
    nose_tri = ((cx - r * 0.1, cy + r * 0.12), (cx + r * 0.1, cy + r * 0.12), (cx, cy + r * 0.25))
    whiskers = []
    for side in (-1, 1):
        for k in (-1, 0, 1):
            whiskers.append(((cx + side * r * 0.3, cy + r * 0.28 + k * r * 0.05), (cx + side * r * 1.15, cy + r * 0.2 + k * r * 0.18)))
    wwidth = max(0.35, unit * 0.006)

    rows: list[bytes] = []
    for y in range(bh):
        t = y / max(1, bh - 1)
        bg = _mix(bg_top, bg_bottom, t)
        row = bytearray()
        for x in range(bw):
            px, py = x + 0.5, y + 0.5
            color = bg
            dx, dy = (px - cx) / r, (py - cy) / (r * 0.9)
            in_head = dx * dx + dy * dy <= 1.0
            if _in_triangle(px, py, *left_ear) or _in_triangle(px, py, *right_ear):
                color = fur
                if _in_triangle(px, py, *left_inner) or _in_triangle(px, py, *right_inner):
                    color = inner_ear
            if in_head:
                color = fur
                if striped and abs(dx) < 0.5 and dy < -0.35 and int((dx + 1) * 6) % 2 == 0:
                    color = fur_dark
                for ex in (cx - eye_dx, cx + eye_dx):
                    edx, edy = (px - ex) / eye_rx, (py - eye_y) / eye_ry
                    if edx * edx + edy * edy <= 1.0:
                        color = eye
                        if abs(px - ex) <= pupil_rx:
                            color = pupil
                if _in_triangle(px, py, *nose_tri):
                    color = nose
                mdx = abs(px - cx)
                if r * 0.25 <= py - cy <= r * 0.40 and mdx <= r * 0.18:
                    # little "w" mouth
                    my = cy + r * 0.30 + abs((mdx % (r * 0.18)) - r * 0.09) * 0.6
                    if abs(py - my) <= max(0.5, r * 0.025):
                        color = pupil
            for a, b in whiskers:
                if _near_segment(px, py, a, b, wwidth) and not (in_head and abs(px - cx) < r * 0.3):
                    color = whisker
                    break
            row += bytes(color)
        rows.append(bytes(row))

    # Nearest-neighbour upscale to the requested size.
    xmap = [min(bw - 1, int(x * bw / width)) for x in range(width)]
    scaled_rows: list[bytes] = []
    cache: dict[int, bytes] = {}
    for y in range(height):
        sy = min(bh - 1, int(y * bh / height))
        if sy not in cache:
            src = rows[sy]
            cache[sy] = b"".join(src[i * 3 : i * 3 + 3] for i in xmap)
        scaled_rows.append(cache[sy])
    return encode_png(width, height, scaled_rows)


def encode_png(width: int, height: int, rows: list[bytes]) -> bytes:
    """Encode 8-bit RGB rows as PNG."""
    raw = b"".join(b"\x00" + row for row in rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    text = b"Software\x00dummyllm (meow)"
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"tEXt", text)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def render_cat_b64(width: int, height: int, rng: random.Random) -> str:
    return base64.b64encode(render_cat(width, height, rng)).decode("ascii")


# ---------------------------------------------------------------------------
# Image header parsing (to compute vision token usage for data: URLs).


def decode_data_url(url: str) -> bytes | None:
    if not url.startswith("data:"):
        return None
    header, _, payload = url.partition(",")
    if not payload:
        return None
    try:
        if header.endswith(";base64"):
            # Only the first chunk is needed for header sniffing.
            head = payload[:65536]
            head = head[: len(head) - len(head) % 4]
            return base64.b64decode(head, validate=False)
        return payload.encode()
    except (ValueError, TypeError):
        return None


def image_size(data: bytes) -> tuple[int, int] | None:
    """Return (width, height) for PNG / GIF / JPEG / WebP / BMP bytes, else None."""
    try:
        if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
            w, h = struct.unpack(">II", data[16:24])
            return w, h
        if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
            w, h = struct.unpack("<HH", data[6:10])
            return w, h
        if data[:2] == b"BM" and len(data) >= 26:
            w, h = struct.unpack("<ii", data[18:26])
            return abs(w), abs(h)
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP" and len(data) >= 30:
            kind = data[12:16]
            if kind == b"VP8X":
                w = 1 + int.from_bytes(data[24:27], "little")
                h = 1 + int.from_bytes(data[27:30], "little")
                return w, h
            if kind == b"VP8 ":
                w, h = struct.unpack("<HH", data[26:30])
                return w & 0x3FFF, h & 0x3FFF
            if kind == b"VP8L":
                b = data[21:25]
                w = 1 + (((b[1] & 0x3F) << 8) | b[0])
                h = 1 + (((b[3] & 0xF) << 10) | (b[2] << 2) | ((b[1] & 0xC0) >> 6))
                return w, h
        if data[:2] == b"\xff\xd8":
            i = 2
            while i + 9 < len(data):
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                    i += 2
                    continue
                seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
                if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    h, w = struct.unpack(">HH", data[i + 5 : i + 9])
                    return w, h
                i += 2 + seg_len
    except (struct.error, IndexError):
        return None
    return None
