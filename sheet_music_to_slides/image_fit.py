"""Tight content bounds and fit to 16:9 without cropping ink."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

# Google Slides default 16:9 page size in EMU (matches API docs)
SLIDE_WIDTH_EMU = 9144000
SLIDE_HEIGHT_EMU = 5143500
# Pixel canvas for exported PNGs (16:9)
DEFAULT_CANVAS_W = 1920
DEFAULT_CANVAS_H = 1080

# Pixels with L *strictly below* this are "ink". Light grey "Preview" watermarks are
# often L≈240–250 and are excluded so the bbox tightens around real notation.
DEFAULT_INK_THRESHOLD = 232
# Small binary dilation to close anti-aliasing gaps (not grayscale blur — that smears watermarks)
MASK_DILATE_SIZE = 3
PAD_PX = 2
PAD_TOP_PX = 12
PAD_BOTTOM_PX = 14

# Cover-page detection: staff lines form short horizontal runs in the note area.
STAFF_X0_FRAC = 0.35
STAFF_X1_FRAC = 0.85
_MIN_STAFF_RUN_ROWS = 5
_MIN_STAFF_CLUSTERS = 1
# Stricter thresholds for the first 1–2 PDF pages (cover/title vs real score).
COVER_MIN_STAFF_CLUSTERS = 40
COVER_MIN_STAFF_ROWS = 450
# Segment filter: drop text/copyright strips after crop.
SEGMENT_MIN_STAFF_CLUSTERS = 15
SEGMENT_MIN_STAFF_ROWS = 80
# Gap filter: below region must look like a full system, not piano/footer.
MIN_BELOW_STAFF_CLUSTERS = 30


@dataclass(frozen=True)
class FitResult:
    image: Image.Image
    """RGBA or RGB image, exact canvas size."""


def _np_gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.uint8)


def _ink_mask_array(gray: np.ndarray, ink_threshold: int) -> np.ndarray:
    """Boolean mask: True = ink (darker than threshold)."""
    return gray < ink_threshold


def _dilate_mask_bool(mask: np.ndarray) -> np.ndarray:
    m = (mask.astype(np.uint8)) * 255
    pil = Image.fromarray(m, mode="L")
    pil = pil.filter(ImageFilter.MaxFilter(MASK_DILATE_SIZE))
    return np.asarray(pil) > 128


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y = np.flatnonzero(rows)
    x = np.flatnonzero(cols)
    t, b = int(y[0]), int(y[-1]) + 1
    l, r = int(x[0]), int(x[-1]) + 1
    return (l, t, r, b)


def _pad_bbox(
    bbox: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    *,
    pad: int = PAD_PX,
    pad_top: int = PAD_TOP_PX,
    pad_bottom: int = PAD_BOTTOM_PX,
) -> tuple[int, int, int, int]:
    l, t, r, b = bbox
    l = max(0, l - pad)
    t = max(0, t - pad_top)
    r = min(img_w, r + pad)
    b = min(img_h, b + pad_bottom)
    return (l, t, r, b)


def count_staff_metrics(
    mask: np.ndarray,
    y0: int = 0,
    y1: int | None = None,
    *,
    x0_frac: float = STAFF_X0_FRAC,
    x1_frac: float = STAFF_X1_FRAC,
) -> tuple[int, int]:
    """Return (staff_clusters, staff_rows) for a boolean ink mask slice."""
    h, w = mask.shape
    if y1 is None:
        y1 = h
    y0 = max(0, min(y0, h))
    y1 = max(y0, min(y1, h))
    if y1 <= y0:
        return (0, 0)

    x0 = int(round(w * x0_frac))
    x1 = int(round(w * x1_frac))
    x0 = max(0, min(x0, w - 2))
    x1 = max(x0 + 1, min(x1, w))
    core_w = x1 - x0

    staff_row = np.zeros(y1 - y0, dtype=bool)
    for i, y in enumerate(range(y0, y1)):
        span = float(np.sum(mask[y, x0:x1])) / core_w
        staff_row[i] = 0.06 <= span <= 0.45

    clusters = 0
    i = 0
    n = len(staff_row)
    while i < n:
        if staff_row[i]:
            ys = i
            while i < n and staff_row[i]:
                i += 1
            if i - ys >= _MIN_STAFF_RUN_ROWS:
                clusters += 1
        else:
            i += 1
    return clusters, int(staff_row.sum())


def count_staff_metrics_img(
    img: Image.Image,
    *,
    white_threshold: int = DEFAULT_INK_THRESHOLD,
) -> tuple[int, int]:
    gray = _np_gray(img)
    m = _ink_mask_array(gray, white_threshold)
    return count_staff_metrics(m)


def page_has_staff_notation(
    img: Image.Image,
    *,
    white_threshold: int = DEFAULT_INK_THRESHOLD,
    min_staff_clusters: int = _MIN_STAFF_CLUSTERS,
    min_staff_rows: int = 0,
) -> bool:
    """
    True if the page has enough staff-like horizontal lines in the center band.
    Cover/title pages fail when called with COVER_MIN_* thresholds.
    """
    clusters, staff_rows = count_staff_metrics_img(img, white_threshold=white_threshold)
    if staff_rows < min_staff_rows:
        return False
    return clusters >= min_staff_clusters


def _is_barcode_row(row: np.ndarray) -> bool:
    """Dense vertical edges + enough dark pixels (typical 1D barcode strip at bottom)."""
    if row.size < 8:
        return False
    d = np.abs(np.diff(row.astype(np.int16)))
    edge_frac = float(np.mean(d > 22))
    dark_frac = float(np.mean(row < 200))
    return edge_frac > 0.13 and dark_frac > 0.07


def _trim_bottom_barcode_and_padding(gray: np.ndarray, mask: np.ndarray) -> int:
    """
    Exclusive bottom row index for the crop [0:ret, :].
    Drops barcode rows from the bottom, then rows with no mask ink (white padding).
    """
    h = gray.shape[0]
    y = h - 1
    while y >= 0 and _is_barcode_row(gray[y]):
        y -= 1
    while y >= 0 and not mask[y].any():
        y -= 1
    return y + 1


def content_bbox(
    img: Image.Image,
    *,
    white_threshold: int = DEFAULT_INK_THRESHOLD,
    min_ink_fraction: float = 0.00015,
    trim_bottom: bool = True,
) -> tuple[int, int, int, int] | None:
    """
    Tight bounding box of printed music (excludes light watermarks, trims barcode/footer).
    """
    gray = _np_gray(img)
    h, w = gray.shape
    m = _ink_mask_array(gray, white_threshold)
    m = _dilate_mask_bool(m)

    ink_count = int(np.sum(m))
    if ink_count / (h * w) < min_ink_fraction:
        return None

    bbox = _bbox_from_mask(m)
    if bbox is None:
        return None
    l, t, r, b = bbox
    sub_g = gray[t:b, l:r]
    sub_m = m[t:b, l:r]
    if trim_bottom:
        new_b = _trim_bottom_barcode_and_padding(sub_g, sub_m)
        if new_b < sub_g.shape[0]:
            b = t + new_b
    sub_m = m[t:b, l:r]
    bbox2 = _bbox_from_mask(sub_m)
    if bbox2 is None:
        return None
    ll, tt, rr, bb = bbox2
    l, t, r, b = l + ll, t + tt, l + rr, t + bb
    return _pad_bbox((l, t, r, b), w, h)


def is_mostly_empty(
    img: Image.Image,
    *,
    white_threshold: int = DEFAULT_INK_THRESHOLD,
    min_ink_fraction: float = 0.0012,
) -> bool:
    """True if region should be skipped (no meaningful music)."""
    gray = _np_gray(img)
    h, w = gray.shape
    m = _ink_mask_array(gray, white_threshold)
    m = _dilate_mask_bool(m)
    ink_count = int(np.sum(m))
    return (ink_count / (w * h)) < min_ink_fraction


def fit_to_slide_canvas(
    img: Image.Image,
    *,
    canvas_w: int = DEFAULT_CANVAS_W,
    canvas_h: int = DEFAULT_CANVAS_H,
    white_threshold: int = DEFAULT_INK_THRESHOLD,
    vertical_align: str = "bottom",
) -> FitResult:
    """
    Crop to tight score bbox, then scale uniformly to fit inside 16:9 (no cropping of content).
    Extra canvas is filled with white; by default the score is **bottom-aligned** so letterboxing
    from wide crops appears mostly above the music (not below).
    """
    bbox = content_bbox(img, white_threshold=white_threshold)
    if bbox is None:
        cropped = img
    else:
        cropped = img.crop(bbox)

    cw, ch = cropped.size
    if cw == 0 or ch == 0:
        out = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        return FitResult(image=out)

    scale = min(canvas_w / cw, canvas_h / ch)
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
    if resized.mode not in ("RGB", "RGBA"):
        resized = resized.convert("RGB")
    elif resized.mode == "RGBA":
        bg = Image.new("RGB", resized.size, (255, 255, 255))
        bg.paste(resized, mask=resized.split()[3])
        resized = bg

    if vertical_align not in ("top", "center", "bottom"):
        raise ValueError("vertical_align must be 'top', 'center', or 'bottom'")

    out = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    x = (canvas_w - new_w) // 2
    if vertical_align == "top":
        y = 0
    elif vertical_align == "bottom":
        y = canvas_h - new_h
    else:
        y = (canvas_h - new_h) // 2
    out.paste(resized, (x, y))
    return FitResult(image=out)
