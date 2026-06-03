"""Render PDF pages into tight PNG sections (system-aware vertical split)."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import fitz  # PyMuPDF

from PIL import Image

from sheet_music_to_slides.image_fit import (
    COVER_MIN_STAFF_CLUSTERS,
    COVER_MIN_STAFF_ROWS,
    DEFAULT_INK_THRESHOLD,
    content_bbox,
    is_mostly_empty,
    page_has_staff_notation,
)
from sheet_music_to_slides.page_split import is_junk_segment, page_vertical_segments
from sheet_music_to_slides.piano_strip import strip_piano_block
from sheet_music_to_slides.text_detect import is_text_only_segment

_PIANO_HINT = re.compile(
    r"\bpiano\b|\bpno\.?\b|\bpf\.?\b|\bkbd\.?\b|keyboard",
    re.IGNORECASE,
)

_MIN_VERTICAL_SLICE_PX = 48
_MIN_CROPPED_SIDE_PX = 48


@dataclass
class Section:
    """One slide candidate after processing."""

    index: int
    """0-based order in output deck."""
    page_index: int
    """0-based PDF page."""
    segment: int
    """0-based segment index within this PDF page."""
    png_bytes: bytes


def _page_to_pil(doc: fitz.Document, page_index: int, dpi: float) -> Image.Image:
    page = doc.load_page(page_index)
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    mode = "RGB" if pix.n == 3 else "RGBA"
    return Image.frombytes(mode, (pix.width, pix.height), pix.samples)


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _document_has_piano_part(doc: fitz.Document) -> bool:
    for i in range(len(doc)):
        text = doc.load_page(i).get_text("text") or ""
        if _PIANO_HINT.search(text):
            return True
    return False


def _document_has_extractable_text(doc: fitz.Document, *, min_chars: int = 24) -> bool:
    n = 0
    for i in range(len(doc)):
        n += len((doc.load_page(i).get_text("text") or "").strip())
        if n >= min_chars:
            return True
    return False


def _should_strip_piano(
    doc: fitz.Document,
    *,
    strip_piano: bool,
    force_piano_strip: bool = False,
) -> bool:
    if not strip_piano:
        return False
    if force_piano_strip:
        return True
    if not _document_has_extractable_text(doc):
        return True
    return _document_has_piano_part(doc)


def extract_sections(
    pdf_path: str,
    *,
    dpi: float = 200.0,
    skip_blank_pages: bool = True,
    ink_threshold: int | None = None,
    min_gap_px: int | None = None,
    only_page_indices: set[int] | None = None,
    strip_piano: bool = False,
    force_piano_strip: bool = False,
    skip_cover_pages: int = 2,
) -> list[Section]:
    """
    Split at center-band whitespace between systems, tight-crop, drop text slivers,
    optionally remove piano from each segment.
    """
    thr = ink_threshold if ink_threshold is not None else DEFAULT_INK_THRESHOLD
    gap = min_gap_px

    doc = fitz.open(pdf_path)
    out: list[Section] = []
    try:
        do_strip_piano = _should_strip_piano(
            doc,
            strip_piano=strip_piano,
            force_piano_strip=force_piano_strip,
        )
        for page_index in range(len(doc)):
            if only_page_indices is not None and page_index not in only_page_indices:
                continue

            page_img = _page_to_pil(doc, page_index, dpi)
            if skip_blank_pages and is_mostly_empty(
                page_img, min_ink_fraction=0.0015, white_threshold=thr
            ):
                continue
            if page_index < skip_cover_pages and not page_has_staff_notation(
                page_img,
                white_threshold=thr,
                min_staff_clusters=COVER_MIN_STAFF_CLUSTERS,
                min_staff_rows=COVER_MIN_STAFF_ROWS,
            ):
                continue

            kwargs: dict = {"white_threshold": thr}
            if gap is not None:
                kwargs["min_gap_px"] = gap
            boxes = page_vertical_segments(page_img, **kwargs)

            for segment, box in enumerate(boxes):
                _l, t, _r, b = box
                if b - t < _MIN_VERTICAL_SLICE_PX:
                    continue
                region = page_img.crop(box)
                if is_mostly_empty(region, min_ink_fraction=0.0012, white_threshold=thr):
                    continue
                bbox = content_bbox(region, white_threshold=thr)
                cropped = region if bbox is None else region.crop(bbox)
                if do_strip_piano:
                    cropped = strip_piano_block(cropped, white_threshold=thr)
                    bbox2 = content_bbox(
                        cropped, white_threshold=thr, trim_bottom=False
                    )
                    if bbox2 is not None:
                        cropped = cropped.crop(bbox2)
                if is_junk_segment(cropped, white_threshold=thr):
                    continue
                if is_text_only_segment(cropped, white_threshold=thr):
                    continue
                if cropped.width < _MIN_CROPPED_SIDE_PX or cropped.height < _MIN_CROPPED_SIDE_PX:
                    continue
                png = _pil_to_png_bytes(cropped)
                out.append(
                    Section(
                        index=len(out),
                        page_index=page_index,
                        segment=segment,
                        png_bytes=png,
                    )
                )
    finally:
        doc.close()

    return out
