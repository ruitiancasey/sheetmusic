"""Tests for vertical gap detection between score systems."""

from __future__ import annotations

import unittest

import numpy as np

from sheet_music_to_slides.page_split import (
    drop_edge_margin_gaps,
    filter_gaps_for_system_splits,
    find_system_divider_gap_intervals,
    find_vertical_gap_intervals,
    gap_has_system_divider_mark,
    regions_between_gaps,
)


def _draw_staff_clusters(
    mask: np.ndarray,
    y0: int,
    y1: int,
    count: int,
) -> None:
    """Draw thin horizontal staff-line clusters in the center band."""
    h, w = mask.shape
    cx0 = int(round(w * 0.35))
    cx1 = int(round(w * 0.85))
    line_half = max(3, int((cx1 - cx0) * 0.1))
    mid = (cx0 + cx1) // 2
    span = max(1, (y1 - y0) // (count + 1))
    for i in range(count):
        base = y0 + span * (i + 1)
        for line in range(5):
            y = base + line
            if y >= y1:
                return
            mask[y, mid - line_half : mid + line_half] = True


class TestPageSplit(unittest.TestCase):
    def test_regions_no_gaps_is_full_height(self) -> None:
        self.assertEqual(regions_between_gaps(100, []), [(0, 100)])

    def test_regions_one_gap_splits(self) -> None:
        self.assertEqual(regions_between_gaps(100, [(40, 55)]), [(0, 40), (55, 100)])

    def test_drop_top_bottom_margin_gaps(self) -> None:
        g = drop_edge_margin_gaps(200, [(0, 30), (80, 100), (180, 200)])
        self.assertEqual(g, [(80, 100)])

    def test_find_gap_requires_min_height(self) -> None:
        m = np.zeros((30, 200), dtype=bool)
        m[0:10, 50:150] = True  # block A
        m[20:30, 50:150] = True  # block B, gap rows 10–19 only 10px tall
        gaps = find_vertical_gap_intervals(m, min_gap_px=12)
        self.assertEqual(gaps, [])

    def test_find_gap_between_blocks(self) -> None:
        m = np.zeros((50, 200), dtype=bool)
        m[0:15, 50:150] = True
        m[35:50, 50:150] = True
        gaps = find_vertical_gap_intervals(m, min_gap_px=15)
        self.assertEqual(len(gaps), 1)
        gs, ge = gaps[0]
        self.assertGreaterEqual(ge - gs, 15)
        self.assertGreaterEqual(gs, 15)
        self.assertLessEqual(ge, 35)

    def test_filter_rejects_short_below_block(self) -> None:
        h, w = 500, 200
        m = np.zeros((h, w), dtype=bool)
        m[0:350, 50:150] = True
        m[370:470, 50:150] = True
        gaps = [(350, 370)]
        kept = filter_gaps_for_system_splits(m, gaps)
        self.assertEqual(kept, [])

    def test_filter_rejects_tall_piano_with_few_staves(self) -> None:
        h, w = 2200, 200
        m = np.zeros((h, w), dtype=bool)
        _draw_staff_clusters(m, 0, 976, 40)
        _draw_staff_clusters(m, 1002, 1709, 52)
        _draw_staff_clusters(m, 1722, 2200, 25)
        gaps = [(976, 1002), (1709, 1722)]
        kept = filter_gaps_for_system_splits(m, gaps)
        self.assertEqual(kept, [(976, 1002)])

    def test_filter_rejects_footer_gap(self) -> None:
        h, w = 2200, 200
        m = np.zeros((h, w), dtype=bool)
        _draw_staff_clusters(m, 0, 1018, 40)
        _draw_staff_clusters(m, 1038, 1851, 40)
        _draw_staff_clusters(m, 1867, 2200, 12)
        gaps = [(1018, 1038), (1851, 1867)]
        kept = filter_gaps_for_system_splits(m, gaps)
        self.assertEqual(kept, [(1018, 1038)])

    def test_filter_accepts_two_tall_systems(self) -> None:
        h, w = 2200, 200
        m = np.zeros((h, w), dtype=bool)
        _draw_staff_clusters(m, 0, 900, 40)
        _draw_staff_clusters(m, 1000, 1900, 40)
        gaps = [(900, 1000)]
        kept = filter_gaps_for_system_splits(m, gaps)
        self.assertEqual(kept, [(900, 1000)])

    def test_divider_gap_detected_in_side_margins(self) -> None:
        h, w = 2200, 1700
        m = np.zeros((h, w), dtype=bool)
        lx0, lx1 = int(w * 0.10), int(w * 0.16)
        rx0, rx1 = int(w * 0.84), int(w * 0.90)
        cx0, cx1 = int(w * 0.35), int(w * 0.85)
        for y in range(700, 730):
            m[y, lx0:lx1] = True
            m[y, rx0:rx1] = True
        m[:, cx0:cx1] = False
        gaps = find_system_divider_gap_intervals(m)
        self.assertEqual(gaps, [(700, 730)])
        self.assertTrue(gap_has_system_divider_mark(m, 700, 730))

    def test_divider_gap_not_confused_with_center_ink(self) -> None:
        h, w = 500, 1700
        m = np.zeros((h, w), dtype=bool)
        lx0, lx1 = int(w * 0.10), int(w * 0.16)
        rx0, rx1 = int(w * 0.84), int(w * 0.90)
        cx0, cx1 = int(w * 0.35), int(w * 0.85)
        for y in range(100, 120):
            m[y, lx0:lx1] = True
            m[y, rx0:rx1] = True
            m[y, cx0:cx1] = True
        self.assertEqual(find_system_divider_gap_intervals(m), [])


if __name__ == "__main__":
    unittest.main()
