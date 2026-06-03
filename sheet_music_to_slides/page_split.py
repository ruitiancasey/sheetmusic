"""Split a full-page score at ink-free horizontal bands (between systems)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from sheet_music_to_slides.image_fit import (
    DEFAULT_INK_THRESHOLD,
    MIN_BELOW_STAFF_CLUSTERS,
    SEGMENT_MIN_STAFF_CLUSTERS,
    SEGMENT_MIN_STAFF_CLUSTERS_STRICT,
    SEGMENT_MIN_STAFF_ROWS,
    count_staff_metrics,
    count_staff_metrics_img,
    _dilate_mask_bool,
    _ink_mask_array,
    _np_gray,
)

# Tight crop still uses the wide core (see image_fit / piano_strip).
CORE_X0_FRAC = 0.14
CORE_X1_FRAC = 0.98

# Gap detection: center band, strict no-ink rows.
GAP_X0_FRAC = 0.35
GAP_X1_FRAC = 0.85
DEFAULT_MIN_GAP_PX = 12
STRICT_GAP_MAX_INK_FRAC = 0.0

# Valley fallback when strict gaps merge systems.
_VALLEY_SMOOTH_WINDOW = 7
_VALLEY_MIN_DEPTH_FRAC = 0.35
_VALLEY_SEARCH_Y0_FRAC = 0.10
_VALLEY_SEARCH_Y1_FRAC = 0.92

# Between-system blocks should be tall; piano / footer blocks are shorter.
_MIN_SYSTEM_BLOCK_FRAC = 0.12
_MIN_SYSTEM_BLOCK_PX = 260
# Piano/footer slivers are short with few staff-line clusters.
_MAX_PIANO_SLIVER_CLUSTERS = 18
_MAX_PIANO_SLIVER_HEIGHT_PX = 280
# Side-margin // system dividers (outside center gap band).
_DIVIDER_LEFT_X0_FRAC = 0.10
_DIVIDER_LEFT_X1_FRAC = 0.16
_DIVIDER_RIGHT_X0_FRAC = 0.84
_DIVIDER_RIGHT_X1_FRAC = 0.90
_MIN_DIVIDER_MARGIN_INK_PX = 3
_MIN_DIVIDER_GAP_PX = 3
# Smaller ensembles (no piano): lower cluster bar when not divider-backed.
_MIN_SYSTEM_STAFF_CLUSTERS = 20
_MIN_DIVIDER_SYSTEM_STAFF_CLUSTERS = 18
_MIN_DIVIDER_SYSTEM_BLOCK_PX = 200
# Valley fallback: inter-system gaps usually land in this vertical band.
_VALLEY_SYSTEM_Y0_FRAC = 0.38
_VALLEY_SYSTEM_Y1_FRAC = 0.52
_MAX_ASPECT_WIDTH_OVER_HEIGHT = 5.5
_MIN_JUNK_STRIP_HEIGHT_PX = 220
_MIN_MUSIC_INK_FRAC = 0.0018


def _row_ink_fraction(mask: np.ndarray, x0: int, x1: int) -> np.ndarray:
    core = mask[:, x0:x1]
    w = x1 - x0
    if w <= 0:
        return np.zeros(mask.shape[0], dtype=np.float64)
    return np.mean(core, axis=1)


def _gap_band_columns(w: int, x0_frac: float, x1_frac: float) -> tuple[int, int]:
    x0 = int(round(w * x0_frac))
    x1 = int(round(w * x1_frac))
    x0 = max(0, min(x0, w - 2))
    x1 = max(x0 + 1, min(x1, w))
    return x0, x1


def _divider_band_columns(w: int) -> tuple[int, int, int, int]:
    lx0 = int(round(w * _DIVIDER_LEFT_X0_FRAC))
    lx1 = int(round(w * _DIVIDER_LEFT_X1_FRAC))
    rx0 = int(round(w * _DIVIDER_RIGHT_X0_FRAC))
    rx1 = int(round(w * _DIVIDER_RIGHT_X1_FRAC))
    return lx0, lx1, rx0, rx1


def _row_has_system_divider_mark(
    mask: np.ndarray,
    y: int,
    *,
    gap_x0_frac: float = GAP_X0_FRAC,
    gap_x1_frac: float = GAP_X1_FRAC,
) -> bool:
    """True when center is clear and both side margins carry // divider ink."""
    _h, w = mask.shape
    cx0, cx1 = _gap_band_columns(w, gap_x0_frac, gap_x1_frac)
    lx0, lx1, rx0, rx1 = _divider_band_columns(w)
    if mask[y, cx0:cx1].any():
        return False
    left_ink = int(np.sum(mask[y, lx0:lx1]))
    right_ink = int(np.sum(mask[y, rx0:rx1]))
    return left_ink >= _MIN_DIVIDER_MARGIN_INK_PX and right_ink >= _MIN_DIVIDER_MARGIN_INK_PX


def gap_has_system_divider_mark(
    mask: np.ndarray,
    gs: int,
    ge: int,
) -> bool:
    """True if any row in [gs, ge) looks like a // system divider band."""
    if ge <= gs:
        return False
    for y in range(gs, ge):
        if _row_has_system_divider_mark(mask, y):
            return True
    return False


def find_system_divider_gap_intervals(
    mask: np.ndarray,
    *,
    min_gap_px: int = _MIN_DIVIDER_GAP_PX,
) -> list[tuple[int, int]]:
    """Gaps at publisher // system dividers (clear center, ink in both side margins)."""
    h, _w = mask.shape
    gaps: list[tuple[int, int]] = []
    y = 0
    while y < h:
        if _row_has_system_divider_mark(mask, y):
            ys = y
            while y < h and _row_has_system_divider_mark(mask, y):
                y += 1
            if y - ys >= min_gap_px:
                gaps.append((ys, y))
        else:
            y += 1
    return gaps


def _merge_nearby_gaps(
    gaps: list[tuple[int, int]],
    *,
    max_separation_px: int = 48,
) -> list[tuple[int, int]]:
    """Merge gap intervals that overlap or sit very close together."""
    if not gaps:
        return []
    ordered = sorted(gaps, key=lambda g: g[0])
    merged: list[tuple[int, int]] = [ordered[0]]
    for gs, ge in ordered[1:]:
        pgs, pge = merged[-1]
        if gs <= pge + max_separation_px:
            merged[-1] = (pgs, max(pge, ge))
        else:
            merged.append((gs, ge))
    return merged


def find_vertical_gap_intervals(
    mask: np.ndarray,
    *,
    min_gap_px: int = DEFAULT_MIN_GAP_PX,
    gap_x0_frac: float = GAP_X0_FRAC,
    gap_x1_frac: float = GAP_X1_FRAC,
    max_ink_frac: float = STRICT_GAP_MAX_INK_FRAC,
) -> list[tuple[int, int]]:
    """Intervals [y_start, y_end) of consecutive no-ink rows in the center gap band."""
    h, w = mask.shape
    x0, x1 = _gap_band_columns(w, gap_x0_frac, gap_x1_frac)
    frac = _row_ink_fraction(mask, x0, x1)
    is_gap_row = frac <= max_ink_frac

    gaps: list[tuple[int, int]] = []
    y = 0
    while y < h:
        if is_gap_row[y]:
            ys = y
            while y < h and is_gap_row[y]:
                y += 1
            if y - ys >= min_gap_px:
                gaps.append((ys, y))
        else:
            y += 1
    return gaps


def _music_extent_rows(mask: np.ndarray, y0: int, y1: int) -> int:
    """Vertical span of rows that contain any ink in [y0, y1)."""
    if y1 <= y0:
        return 0
    rows = np.any(mask[y0:y1], axis=1)
    if not rows.any():
        return 0
    ys = np.flatnonzero(rows)
    return int(ys[-1] - ys[0]) + 1


def looks_like_sheet_music_region(
    mask: np.ndarray,
    y0: int,
    y1: int,
) -> bool:
    """True if a crop band looks like notation (not a thin text/footer strip)."""
    if y1 <= y0:
        return False
    h, w = mask.shape
    sub = mask[y0:y1, :]
    hh = y1 - y0
    if hh < 20:
        return False
    ink_frac = float(np.sum(sub)) / (hh * w)
    if ink_frac < _MIN_MUSIC_INK_FRAC:
        return False
    if hh < _MIN_JUNK_STRIP_HEIGHT_PX and (w / hh) >= _MAX_ASPECT_WIDTH_OVER_HEIGHT:
        return False
    return True


def _min_system_block_px(page_h: int) -> int:
    return max(_MIN_SYSTEM_BLOCK_PX, int(round(page_h * _MIN_SYSTEM_BLOCK_FRAC)))


def _is_piano_sliver_region(
    mask: np.ndarray,
    y0: int,
    y1: int,
) -> bool:
    """True for a short piano or footer block (not a full string-system region)."""
    h = y1 - y0
    if h <= 0:
        return True
    clusters, _ = count_staff_metrics(mask, y0, y1)
    return clusters < _MAX_PIANO_SLIVER_CLUSTERS and h < _MAX_PIANO_SLIVER_HEIGHT_PX


def _is_bass_piano_gap(
    mask: np.ndarray,
    regions: list[tuple[int, int]],
    gap_index: int,
) -> bool:
    """True when a gap separates strings from piano within one system."""
    if gap_index + 1 >= len(regions):
        return False
    r_above = regions[gap_index]
    r_below = regions[gap_index + 1]
    below_c, _ = count_staff_metrics(mask, r_below[0], r_below[1])
    below_h = r_below[1] - r_below[0]
    if (
        below_c < _MAX_PIANO_SLIVER_CLUSTERS
        and below_h < _MAX_PIANO_SLIVER_HEIGHT_PX
    ):
        return True
    above_c, _ = count_staff_metrics(mask, r_above[0], r_above[1])
    above_h = r_above[1] - r_above[0]
    return (
        below_c < MIN_BELOW_STAFF_CLUSTERS
        and above_h >= 500
        and above_c >= 25
    )


def _expand_above_merged(
    regions: list[tuple[int, int]],
    mask: np.ndarray,
    gap_index: int,
) -> tuple[int, int]:
    """Merge the region above a gap with any attached piano sliver(s) below strings."""
    end_idx = gap_index
    start_idx = gap_index
    while start_idx > 0 and _is_piano_sliver_region(
        mask, regions[start_idx][0], regions[start_idx][1]
    ):
        start_idx -= 1
    return regions[start_idx][0], regions[end_idx][1]


def _expand_below_merged(
    regions: list[tuple[int, int]],
    mask: np.ndarray,
    gap_index: int,
) -> tuple[int, int]:
    """Merge the region below a gap with an attached piano block if present."""
    start_idx = gap_index + 1
    end_idx = gap_index + 1
    if end_idx + 1 < len(regions):
        nxt = regions[end_idx + 1]
        nc, _ = count_staff_metrics(mask, nxt[0], nxt[1])
        nh = nxt[1] - nxt[0]
        if _is_piano_sliver_region(mask, nxt[0], nxt[1]) or (
            nc < MIN_BELOW_STAFF_CLUSTERS and nh < 520
        ):
            end_idx += 1
    return regions[start_idx][0], regions[end_idx][1]


def _gap_passes_system_split_check(
    mask: np.ndarray,
    gaps: list[tuple[int, int]],
    gap_index: int,
    *,
    min_below_staff_clusters: int,
    page_h: int,
    has_divider: bool = False,
) -> bool:
    """True when a gap separates two full systems (strings + piano each side)."""
    min_block = (
        _MIN_DIVIDER_SYSTEM_BLOCK_PX
        if has_divider
        else _min_system_block_px(page_h)
    )
    min_clusters = (
        _MIN_DIVIDER_SYSTEM_STAFF_CLUSTERS
        if has_divider
        else min_below_staff_clusters
    )
    regions = regions_between_gaps(page_h, gaps)
    if gap_index + 1 >= len(regions):
        return False
    if not has_divider and _is_bass_piano_gap(mask, regions, gap_index):
        return False

    above_t, above_b = _expand_above_merged(regions, mask, gap_index)
    below_t, below_b = _expand_below_merged(regions, mask, gap_index)
    above_h = above_b - above_t
    below_h = below_b - below_t
    if above_h < min_block or below_h < min_block:
        return False
    above_clusters, _ = count_staff_metrics(mask, above_t, above_b)
    below_clusters, _ = count_staff_metrics(mask, below_t, below_b)
    return (
        above_clusters >= min_clusters
        and below_clusters >= min_clusters
    )


def filter_gaps_for_system_splits(
    mask: np.ndarray,
    gaps: list[tuple[int, int]],
    *,
    min_below_staff_clusters: int = MIN_BELOW_STAFF_CLUSTERS,
    page_h: int | None = None,
) -> list[tuple[int, int]]:
    """
    Keep only gaps that separate two systems (not bass/piano or footer text).
    Merges string + piano blocks on each side before measuring staff evidence.
    """
    if not gaps:
        return []
    h, _w = mask.shape
    ph = page_h if page_h is not None else h
    out: list[tuple[int, int]] = []
    for i, gap in enumerate(gaps):
        gs, ge = gap
        if _gap_passes_system_split_check(
            mask,
            gaps,
            i,
            min_below_staff_clusters=min_below_staff_clusters,
            page_h=ph,
            has_divider=gap_has_system_divider_mark(mask, gs, ge),
        ):
            out.append(gap)
    return out


def drop_edge_margin_gaps(
    page_h: int, gaps: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Remove gaps that touch the top or bottom of the page (page margins)."""
    out: list[tuple[int, int]] = []
    for gs, ge in gaps:
        if gs <= 0:
            continue
        if ge >= page_h:
            continue
        out.append((gs, ge))
    return out


def regions_between_gaps(page_h: int, gaps: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Turn gap bands into vertical slices: [0, g0_s), [g0_e, g1_s), … [gN_e, page_h)."""
    if not gaps:
        return [(0, page_h)]
    regions: list[tuple[int, int]] = []
    y0 = 0
    for gs, ge in gaps:
        if gs > y0:
            regions.append((y0, gs))
        y0 = ge
    if y0 < page_h:
        regions.append((y0, page_h))
    return regions


def _smooth_1d(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr.astype(np.float64)
    k = np.ones(window, dtype=np.float64) / window
    return np.convolve(arr.astype(np.float64), k, mode="same")


def find_valley_gap_interval(
    mask: np.ndarray,
    *,
    min_gap_px: int = DEFAULT_MIN_GAP_PX,
    gap_x0_frac: float = GAP_X0_FRAC,
    gap_x1_frac: float = GAP_X1_FRAC,
    y_search0_frac: float = _VALLEY_SEARCH_Y0_FRAC,
    y_search1_frac: float = _VALLEY_SEARCH_Y1_FRAC,
) -> tuple[int, int] | None:
    """Deepest ink-density valley in the center band (fallback for short/bridged gaps)."""
    h, w = mask.shape
    x0, x1 = _gap_band_columns(w, gap_x0_frac, gap_x1_frac)
    row_ink = np.sum(mask[:, x0:x1], axis=1, dtype=np.float64)
    if row_ink.max() <= 0:
        return None

    smooth = _smooth_1d(row_ink, _VALLEY_SMOOTH_WINDOW)
    y_search0 = int(h * y_search0_frac)
    y_search1 = int(h * y_search1_frac)
    if y_search1 <= y_search0 + min_gap_px:
        return None

    sub = smooth[y_search0:y_search1]
    peak = float(sub.max())
    if peak <= 0:
        return None

    y_min = int(np.argmin(sub)) + y_search0
    valley = float(smooth[y_min])
    if valley > peak * (1.0 - _VALLEY_MIN_DEPTH_FRAC):
        return None

    lo, hi = y_min, y_min + 1
    thresh = valley + (peak - valley) * 0.25
    while lo > 0 and smooth[lo - 1] <= thresh:
        lo -= 1
    while hi < h and smooth[hi] <= thresh:
        hi += 1
    if hi - lo < min_gap_px:
        return None
    return (lo, hi)


def is_junk_segment(
    img: Image.Image,
    *,
    white_threshold: int = DEFAULT_INK_THRESHOLD,
) -> bool:
    """True for thin text-only strips with little notation."""
    w, h = img.size
    if h <= 0 or w <= 0:
        return True
    aspect = w / h
    if h < _MIN_JUNK_STRIP_HEIGHT_PX and aspect >= _MAX_ASPECT_WIDTH_OVER_HEIGHT:
        return True
    clusters, staff_rows = count_staff_metrics_img(img, white_threshold=white_threshold)
    if clusters < SEGMENT_MIN_STAFF_CLUSTERS:
        return True
    if (
        clusters < SEGMENT_MIN_STAFF_CLUSTERS_STRICT
        and staff_rows < SEGMENT_MIN_STAFF_ROWS
    ):
        return True
    if clusters < 20 and staff_rows < SEGMENT_MIN_STAFF_ROWS:
        return True
    gray = _np_gray(img)
    hh, ww = gray.shape
    m = _ink_mask_array(gray, white_threshold)
    if int(np.sum(m)) / (hh * ww) < _MIN_MUSIC_INK_FRAC:
        return True
    return False


def page_vertical_segments(
    img: Image.Image,
    *,
    white_threshold: int = DEFAULT_INK_THRESHOLD,
    min_gap_px: int = DEFAULT_MIN_GAP_PX,
) -> list[tuple[int, int, int, int]]:
    """
    Return pixel crop boxes (l, t, r, b) for each vertical segment on the page.
    """
    gray = _np_gray(img)
    h, w = gray.shape
    m_gap = _ink_mask_array(gray, white_threshold)

    gaps = find_vertical_gap_intervals(m_gap, min_gap_px=min_gap_px)
    divider_gaps = find_system_divider_gap_intervals(m_gap)
    gaps = _merge_nearby_gaps(gaps + divider_gaps)
    gaps = drop_edge_margin_gaps(h, gaps)
    m_any = _dilate_mask_bool(m_gap)
    gaps = filter_gaps_for_system_splits(
        m_any,
        gaps,
        min_below_staff_clusters=_MIN_SYSTEM_STAFF_CLUSTERS,
        page_h=h,
    )

    slices = regions_between_gaps(h, gaps)
    if len(slices) == 1 and h > 300 and not divider_gaps:
        valley = find_valley_gap_interval(
            m_gap,
            min_gap_px=min_gap_px,
            y_search0_frac=_VALLEY_SYSTEM_Y0_FRAC,
            y_search1_frac=_VALLEY_SYSTEM_Y1_FRAC,
        )
        if valley is not None:
            vg = filter_gaps_for_system_splits(
                m_any,
                drop_edge_margin_gaps(h, [valley]),
                min_below_staff_clusters=_MIN_SYSTEM_STAFF_CLUSTERS,
                page_h=h,
            )
            if vg:
                slices = regions_between_gaps(h, vg)

    out: list[tuple[int, int, int, int]] = []
    for t, b in slices:
        if b <= t:
            continue
        if not m_any[t:b, :].any():
            continue
        out.append((0, t, w, b))

    if not out:
        return [(0, 0, w, h)]
    return out
