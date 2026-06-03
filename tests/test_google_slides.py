"""Mocked test: Slides + Drive flow without network or OAuth."""

from __future__ import annotations

import io
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from sheet_music_to_slides.google_slides import EMU_H, EMU_W, build_presentation


def _png(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=(255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


class TestBuildPresentation(unittest.TestCase):
    def test_creates_deck_and_batch_update_with_two_images(self) -> None:
        creds_path = Path("/fake/credentials.json")
        token_path = Path("/fake/token.json")

        drive = MagicMock()
        slides = MagicMock()

        def build_side_effect(service: str, version: str, **kwargs):
            if service == "drive":
                return drive
            if service == "slides":
                return slides
            raise AssertionError(service)

        file_execute = MagicMock(
            side_effect=[
                {"id": "folder_ABC"},
                {"id": "drive_img_0"},
                {"id": "drive_img_1"},
            ]
        )
        drive.files.return_value.create.return_value.execute = file_execute
        drive.permissions.return_value.create.return_value.execute = MagicMock(
            return_value={"id": "perm"}
        )

        slides.presentations.return_value.create.return_value.execute.return_value = {
            "presentationId": "pres_xyz"
        }
        slides.presentations.return_value.get.return_value.execute.return_value = {
            "slides": [{"objectId": "default_slide_1"}]
        }
        batch = slides.presentations.return_value.batchUpdate
        batch.return_value.execute.return_value = {}

        with patch("sheet_music_to_slides.google_slides.build", side_effect=build_side_effect):
            with patch("sheet_music_to_slides.google_slides._load_credentials"):
                url = build_presentation(
                    title="Test deck",
                    image_png_bytes=[_png(1920, 1080), _png(1920, 1080)],
                    credentials_path=creds_path,
                    token_path=token_path,
                )

        self.assertIn("pres_xyz", url)
        self.assertTrue(url.startswith("https://docs.google.com/presentation/d/"))

        batch.assert_called_once()
        call_kw = batch.call_args.kwargs
        self.assertEqual(call_kw["presentationId"], "pres_xyz")
        requests = call_kw["body"]["requests"]
        # blank slide + delete default + image0 + createSlide + image1
        self.assertEqual(len(requests), 5)
        self.assertIn("createSlide", requests[0])
        self.assertIn("deleteObject", requests[1])
        self.assertEqual(requests[1]["deleteObject"]["objectId"], "default_slide_1")
        img0 = requests[2]["createImage"]
        self.assertIn("drive.google.com", img0["url"])
        self.assertIn("drive_img_0", img0["url"])
        sz = img0["elementProperties"]["size"]
        self.assertEqual(sz["width"]["magnitude"], EMU_W)
        self.assertEqual(sz["height"]["magnitude"], EMU_H)

        self.assertEqual(file_execute.call_count, 3)
        drive.permissions.return_value.create.assert_called()


if __name__ == "__main__":
    unittest.main()
