"""Tests for staff detection and cover-page logic."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from sheet_music_to_slides.image_fit import (
    COVER_MIN_STAFF_CLUSTERS,
    COVER_MIN_STAFF_ROWS,
    page_has_staff_notation,
)


class TestPageHasStaffNotation(unittest.TestCase):
    def test_blank_page_is_false(self) -> None:
        img = Image.new("RGB", (400, 600), (255, 255, 255))
        self.assertFalse(page_has_staff_notation(img))

    def test_staff_cluster_is_true(self) -> None:
        arr = np.ones((400, 500), dtype=np.uint8) * 255
        line_l, line_r = 250, 310
        base = 80
        for line in range(5):
            arr[base + line, line_l:line_r] = 10
        img = Image.fromarray(arr, mode="L").convert("RGB")
        self.assertTrue(page_has_staff_notation(img, min_staff_clusters=1))

    def test_title_block_without_staff_runs_is_false(self) -> None:
        arr = np.ones((300, 500), dtype=np.uint8) * 255
        arr[50:200, 100:400] = 10
        img = Image.fromarray(arr, mode="L").convert("RGB")
        self.assertFalse(page_has_staff_notation(img))

    def test_cover_thresholds_reject_sparse_staff(self) -> None:
        arr = np.ones((400, 500), dtype=np.uint8) * 255
        line_l, line_r = 250, 310
        base = 80
        for line in range(5):
            arr[base + line, line_l:line_r] = 10
        img = Image.fromarray(arr, mode="L").convert("RGB")
        self.assertTrue(page_has_staff_notation(img, min_staff_clusters=1))
        self.assertFalse(
            page_has_staff_notation(
                img,
                min_staff_clusters=COVER_MIN_STAFF_CLUSTERS,
                min_staff_rows=COVER_MIN_STAFF_ROWS,
            )
        )


if __name__ == "__main__":
    unittest.main()
