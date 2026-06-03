"""CLI: PDF → Google Slides."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sheet_music_to_slides import __version__
from sheet_music_to_slides.google_slides import build_presentation
from sheet_music_to_slides.pdf_sections import extract_sections


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="sheet-music-to-slides",
        description=(
            "Convert sheet music PDF to a Google Slides deck (system-aware vertical split, OAuth)."
        ),
    )
    p.add_argument("pdf", type=Path, help="Path to input PDF")
    p.add_argument(
        "--title",
        default=None,
        help="Presentation title (default: stem of PDF file)",
    )
    p.add_argument("--dpi", type=float, default=200.0, help="Rasterization DPI (default: 200)")
    p.add_argument(
        "--ink-threshold",
        type=int,
        default=None,
        metavar="L",
        help=(
            "Grayscale cutoff for ink (0–255): lower = tighter crop, excludes light watermarks. "
            "Default 232; raise toward 250 if light measure numbers vanish."
        ),
    )
    p.add_argument(
        "--min-gap-px",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Minimum height (pixels at render DPI) of a no-ink band in the center "
            "of the page used to split systems. Default 12."
        ),
    )
    p.add_argument(
        "--pages",
        type=str,
        default=None,
        metavar="N,M,...",
        help="Only process these 0-based PDF page indices (e.g. 2,3).",
    )
    p.add_argument(
        "--strip-piano",
        action="store_true",
        help=(
            "Remove piano from the bottom of each slide (default: keep piano). "
            "On image-only PDFs, also applies when the score appears to include piano."
        ),
    )
    p.add_argument(
        "--credentials",
        type=Path,
        default=Path("credentials.json"),
        help="OAuth client JSON from Google Cloud (default: ./credentials.json)",
    )
    p.add_argument(
        "--token",
        type=Path,
        default=Path("token.json"),
        help="Saved OAuth token (default: ./token.json)",
    )
    p.add_argument(
        "--save-images",
        type=Path,
        default=None,
        metavar="DIR",
        help="Write slide PNGs to this directory",
    )
    p.add_argument(
        "--images-only",
        action="store_true",
        help="Only extract and save PNGs; do not upload to Google Slides (requires --save-images).",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = p.parse_args(argv)

    if args.images_only and not args.save_images:
        print("Error: --images-only requires --save-images DIR", file=sys.stderr)
        return 1

    pdf = args.pdf.expanduser().resolve()
    if not pdf.is_file():
        print(f"Error: PDF not found: {pdf}", file=sys.stderr)
        return 1

    title = args.title or pdf.stem

    only_pages: set[int] | None = None
    if args.pages is not None:
        try:
            only_pages = {int(p.strip()) for p in args.pages.split(",") if p.strip()}
        except ValueError:
            print("Error: --pages must be comma-separated integers", file=sys.stderr)
            return 1

    sections = extract_sections(
        str(pdf),
        dpi=args.dpi,
        skip_blank_pages=True,
        ink_threshold=args.ink_threshold,
        min_gap_px=args.min_gap_px,
        only_page_indices=only_pages,
        strip_piano=args.strip_piano,
        force_piano_strip=args.strip_piano,
    )
    if not sections:
        print(
            "No slide images produced (all pages empty or filtered as blank).",
            file=sys.stderr,
        )
        return 2

    out_dir: Path | None = None
    if args.save_images:
        out_dir = args.save_images.expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        for s in sections:
            if only_pages is not None:
                path = out_dir / f"slide_p{s.page_index}_s{s.segment:02d}.png"
            else:
                path = out_dir / f"slide_{s.index:04d}_p{s.page_index}_s{s.segment:02d}.png"
            path.write_bytes(s.png_bytes)

    if args.images_only:
        assert out_dir is not None
        print(f"Saved {len(sections)} images to {out_dir}")
        return 0

    creds = args.credentials.expanduser().resolve()
    token = args.token.expanduser().resolve()

    try:
        url = build_presentation(
            title=title,
            image_png_bytes=[s.png_bytes for s in sections],
            credentials_path=creds,
            token_path=token,
        )
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 3
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 4

    print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
