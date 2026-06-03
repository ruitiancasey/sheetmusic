"""Remove the piano staff block from the bottom of a single-system raster crop."""

from __future__ import annotations

import numpy as np
from PIL import Image

from sheet_music_to_slides.image_fit import (
    DEFAULT_INK_THRESHOLD,
    _ink_mask_array,
    _np_gray,
)
from sheet_music_to_slides.page_split import GAP_X0_FRAC, GAP_X1_FRAC

_DENSE_FRAC = 0.018
_GAP_FRAC = 0.005
_MIN_GAP_ABOVE_PIANO_PX = 6
_MIN_PIANO_DENSE_BLOCK_PX = 92
_MIN_KEEP_FRAC = 0.22
_PAD_BELOW_STRINGS_PX = 10


def _row_ink_fraction(mask: np.ndarray, x0: int, x1: int) -> np.ndarray:
    core = mask[:, x0:x1]
    w = x1 - x0
    if w <= 0:
        return np.zeros(mask.shape[0], dtype=np.float64)
    return np.mean(core, axis=1)


def strip_piano_block(
    img: Image.Image,
    *,
    white_threshold: int = DEFAULT_INK_THRESHOLD,
) -> Image.Image:
    """
    Cut off everything from the first sufficiently empty horizontal band below the
    strings (gap above piano) through the bottom of the image. If no safe cut is
    found, returns the original image.
    """
    gray = _np_gray(img)
    h, w = gray.shape
    if h < 80 or w < 80:
        return img

    m = _ink_mask_array(gray, white_threshold)
    if not m.any():
        return img

    x0 = int(round(w * GAP_X0_FRAC))
    x1 = int(round(w * GAP_X1_FRAC))
    x0 = max(0, min(x0, w - 2))
    x1 = max(x0 + 1, min(x1, w))

    frac = _row_ink_fraction(m, x0, x1)

    y = h - 1
    while y >= 0 and frac[y] <= _GAP_FRAC:
        y -= 1
    if y < 0:
        return img

    piano_dense_h = 0
    while y >= 0 and frac[y] > _DENSE_FRAC:
        y -= 1
        piano_dense_h += 1
    if y < 0:
        return img
    if piano_dense_h < _MIN_PIANO_DENSE_BLOCK_PX:
        return img

    if frac[y] > _GAP_FRAC:
        return img

    gap_h = 0
    while y >= 0 and frac[y] <= _GAP_FRAC:
        y -= 1
        gap_h += 1

    if gap_h < _MIN_GAP_ABOVE_PIANO_PX:
        return img
    if y < 0:
        return img

    new_h = min(h, y + 1 + _PAD_BELOW_STRINGS_PX)
    if new_h < int(h * _MIN_KEEP_FRAC):
        return img

    return img.crop((0, 0, w, new_h))
