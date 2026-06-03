"""Detect text-only segments (copyright, title strips) vs real notation."""

from __future__ import annotations

from PIL import Image

from sheet_music_to_slides.image_fit import (
    DEFAULT_INK_THRESHOLD,
    SEGMENT_MIN_STAFF_CLUSTERS,
    SEGMENT_MIN_STAFF_ROWS,
    count_staff_metrics_img,
)

# OCR runs only when staff evidence is borderline.
_OCR_CLUSTER_LO = 15
_OCR_CLUSTER_HI = 30
_OCR_MIN_TEXT_AREA_FRAC = 0.35
_OCR_MAX_STAFF_CLUSTERS = 25


def _ocr_text_area_fraction(img: Image.Image) -> float | None:
    """Fraction of image covered by OCR word boxes, or None if OCR unavailable."""
    try:
        import pytesseract
    except ImportError:
        return None

    try:
        data = pytesseract.image_to_data(
            img.convert("L"), output_type=pytesseract.Output.DICT
        )
    except Exception:
        return None

    w, h = img.size
    if w <= 0 or h <= 0:
        return 0.0

    text_area = 0
    n = len(data["text"])
    for i in range(n):
        text = (data["text"][i] or "").strip()
        conf = int(data["conf"][i]) if str(data["conf"][i]).lstrip("-").isdigit() else -1
        if not text or conf < 30:
            continue
        tw = int(data["width"][i])
        th = int(data["height"][i])
        if tw > 0 and th > 0:
            text_area += tw * th
    return text_area / (w * h)


def is_text_only_segment(
    img: Image.Image,
    *,
    white_threshold: int = DEFAULT_INK_THRESHOLD,
) -> bool:
    """
    True if the crop is mostly words with little staff notation.
    Fast staff check first; optional Tesseract on borderline crops.
    """
    clusters, staff_rows = count_staff_metrics_img(img, white_threshold=white_threshold)

    if clusters < SEGMENT_MIN_STAFF_CLUSTERS:
        return True
    if clusters < 20 and staff_rows < SEGMENT_MIN_STAFF_ROWS:
        return True

    if _OCR_CLUSTER_LO <= clusters < _OCR_CLUSTER_HI:
        text_frac = _ocr_text_area_fraction(img)
        if text_frac is not None:
            if text_frac > _OCR_MIN_TEXT_AREA_FRAC and clusters < _OCR_MAX_STAFF_CLUSTERS:
                return True

    return False
