"""Tests for text-only segment detection."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from sheet_music_to_slides.text_detect import is_text_only_segment


class TestTextOnlySegment(unittest.TestCase):
    def test_blank_is_text_only(self) -> None:
        img = Image.new("RGB", (400, 200), (255, 255, 255))
        self.assertTrue(is_text_only_segment(img))

    def test_dense_text_block_is_text_only(self) -> None:
        arr = np.ones((200, 500), dtype=np.uint8) * 255
        arr[40:160, 80:420] = 10
        img = Image.fromarray(arr, mode="L").convert("RGB")
        self.assertTrue(is_text_only_segment(img))

    def test_staff_lines_are_not_text_only(self) -> None:
        arr = np.ones((400, 500), dtype=np.uint8) * 255
        cx0, cx1 = 175, 325
        line_half = 30
        mid = (cx0 + cx1) // 2
        for block in range(22):
            base = 20 + block * 16
            for line in range(5):
                y = base + line
                arr[y, mid - line_half : mid + line_half] = 10
        img = Image.fromarray(arr, mode="L").convert("RGB")
        self.assertFalse(is_text_only_segment(img))


if __name__ == "__main__":
    unittest.main()
