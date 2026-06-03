# Sheet music → Google Slides

Local CLI: rasterize a PDF, split pages at system boundaries, skip text/cover junk, tight-crop notation, optionally upload to a **new** Google Slides deck via **OAuth**.

## Setup

1. **Python 3.10+**

2. **Google Cloud**
   - Create a project → **APIs & Services** → **Library** → enable **Google Slides API** and **Google Drive API**.
   - **OAuth consent screen** → External (or Internal for Workspace) → add scopes:
     - `.../auth/presentations`
     - `.../auth/drive.file`
   - If the app is in **Testing**, add your Google account under **Test users**.
   - **Credentials** → **Create credentials** → **OAuth client ID** → **Desktop app**.
   - Download JSON and save as `credentials.json` in the project directory (or pass `--credentials`).

3. **Install**

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

## Usage

```bash
sheet-music-to-slides path/to/score.pdf
```

First run opens a browser to sign in; a `token.json` is saved for later runs.

**Upload flow:** the CLI uploads each slide PNG to **Drive** (new folder per run), sets each file to **Anyone with the link can view** so **Google Slides** can embed them by URL, then creates a **new presentation** and prints its **edit URL**.

Options:

- `--title "My Deck"` — presentation title (default: PDF filename stem).
- `--dpi 200` — render DPI for scanned pages.
- `--ink-threshold L` — grayscale cutoff for “ink” (default **232**). Lower excludes light grey overlays (e.g. “Preview” watermarks) so crops tighten; if measure numbers disappear, try **238–245**.
- `--valign top|center|bottom` — placement in the 16:9 frame after scaling (default **bottom**: letterboxing from wide crops sits **above** the staff, not under it). Does not change how the PDF is cropped—only where the image sits on the slide.
- `--save-images ./out` — write slide PNGs for inspection.
- `--images-only` — save PNGs only (no Google upload); requires `--save-images`.
- `--strip-piano` — remove piano from each slide (default: keep piano).
- `--credentials` / `--token` — paths to OAuth client JSON and saved token.

On success, the **presentation URL** is printed to stdout.

## Notes

- Source images live in **Drive** under `sheet_music_to_slides_<random>`; they are **link-shared** for Slides. Do not delete them if you want images to keep showing in the deck.
- To **re-auth** (e.g. new scopes), delete `token.json` and run again.
- **Workspace** orgs may block “Anyone with the link”; if images don’t appear in Slides, ask an admin or use an account without that restriction.
