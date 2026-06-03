"""Tests for piano block stripping."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from sheet_music_to_slides.image_fit import DEFAULT_INK_THRESHOLD
from sheet_music_to_slides.piano_strip import strip_piano_block


class TestPianoStrip(unittest.TestCase):
    def test_removes_bottom_block_after_gap(self) -> None:
        w, h = 300, 500
        arr = np.ones((h, w), dtype=np.uint8) * 255
        # Strings block (top)
        arr[40:180, 80:220] = 10
        # Piano block (bottom) — gap rows 180–280 in core are white
        arr[300:460, 80:220] = 10
        img = Image.fromarray(arr, mode="L").convert("RGB")
        out = strip_piano_block(img, white_threshold=DEFAULT_INK_THRESHOLD)
        self.assertLess(out.height, img.height)
        self.assertEqual(out.width, img.width)
        self.assertGreater(out.height, 100)

    def test_no_cut_when_no_clear_gap(self) -> None:
        w, h = 100, 200
        arr = np.ones((h, w), dtype=np.uint8) * 255
        arr[20:180, 30:70] = 10
        img = Image.fromarray(arr, mode="L").convert("RGB")
        out = strip_piano_block(img, white_threshold=DEFAULT_INK_THRESHOLD)
        self.assertEqual(out.size, img.size)


if __name__ == "__main__":
    unittest.main()
