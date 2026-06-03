"""OAuth, Drive upload, and Slides deck creation."""

from __future__ import annotations

import io
import uuid
from pathlib import Path

from PIL import Image

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseUpload

# Slides + Drive (files created by this app)
SCOPES = [
    "https://www.googleapis.com/auth/presentations",
    "https://www.googleapis.com/auth/drive.file",
]

# Default 16:9 slide size (EMU)
EMU_W = 9144000
EMU_H = 5143500


def _element_size_and_origin(png_bytes: bytes) -> tuple[float, float, float, float]:
    """
    Size (EMU) and top-left translation so the image fits inside the slide
    without distortion and is centered.
    """
    try:
        with Image.open(io.BytesIO(png_bytes)) as im:
            w_px, h_px = im.size
    except Exception:
        w_px, h_px = 1920, 1080
    if w_px <= 0 or h_px <= 0:
        w_px, h_px = 1920, 1080
    aspect = w_px / h_px
    slide_aspect = EMU_W / EMU_H
    if aspect >= slide_aspect:
        emu_w = float(EMU_W)
        emu_h = float(EMU_W / aspect)
    else:
        emu_h = float(EMU_H)
        emu_w = float(EMU_H * aspect)
    tx = (EMU_W - emu_w) / 2.0
    ty = (EMU_H - emu_h) / 2.0
    return emu_w, emu_h, tx, ty


def _load_credentials(
    credentials_path: Path,
    token_path: Path,
) -> Credentials:
    creds: Credentials | None = None
    if token_path.is_file():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    if not creds or not creds.valid:
        if not credentials_path.is_file():
            raise FileNotFoundError(
                f"Missing OAuth client file: {credentials_path}\n"
                "Create a Desktop OAuth client in Google Cloud Console and save JSON as credentials.json."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
        creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json())
    return creds


def drive_image_url(file_id: str) -> str:
    """URL that Google Slides can fetch for a Drive file visible to the user."""
    return f"https://drive.google.com/uc?export=view&id={file_id}"


def upload_png_bytes(
    drive,
    png_bytes: bytes,
    filename: str,
    folder_id: str | None = None,
) -> str:
    """Upload PNG to Drive; returns file id."""
    meta: dict = {"name": filename, "mimeType": "image/png"}
    if folder_id:
        meta["parents"] = [folder_id]
    media = MediaIoBaseUpload(
        io.BytesIO(png_bytes),
        mimetype="image/png",
        resumable=False,
    )
    file = drive.files().create(body=meta, media_body=media, fields="id").execute()
    file_id = file["id"]
    # Slides fetches image URLs without user cookies; link-readable PNGs load reliably.
    drive.permissions().create(
        fileId=file_id,
        body={"type": "anyone", "role": "reader"},
        fields="id",
    ).execute()
    return file_id


def create_folder(drive, name: str) -> str:
    body = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    f = drive.files().create(body=body, fields="id").execute()
    return f["id"]


def build_presentation(
    *,
    title: str,
    image_png_bytes: list[bytes],
    credentials_path: Path,
    token_path: Path,
) -> str:
    """
    Create a new Google Slides presentation with one full-slide image per slide.
    Returns the presentation URL (document URL).
    """
    if not image_png_bytes:
        raise ValueError("No slides to create")

    creds = _load_credentials(credentials_path, token_path)
    drive = build("drive", "v3", credentials=creds, cache_discovery=False)
    slides = build("slides", "v1", credentials=creds, cache_discovery=False)

    folder_name = f"sheet_music_to_slides_{uuid.uuid4().hex[:12]}"
    folder_id = create_folder(drive, folder_name)

    file_ids: list[str] = []
    for i, blob in enumerate(image_png_bytes):
        fid = upload_png_bytes(drive, blob, f"slide_{i:04d}.png", folder_id=folder_id)
        file_ids.append(fid)

    try:
        pres = slides.presentations().create(body={"title": title}).execute()
    except HttpError as e:
        raise RuntimeError(f"Failed to create presentation: {e}") from e

    presentation_id = pres["presentationId"]
    pres_full = slides.presentations().get(presentationId=presentation_id).execute()
    first_slides = pres_full.get("slides") or []
    if not first_slides:
        raise RuntimeError("Presentation has no slides")
    default_slide_id = first_slides[0]["objectId"]

    first_blank_id = f"blank_first_{uuid.uuid4().hex[:8]}"

    def image_req(slide_object_id: str, file_id: str, blob: bytes) -> dict:
        ew, eh, tx, ty = _element_size_and_origin(blob)
        return {
            "createImage": {
                "url": drive_image_url(file_id),
                "elementProperties": {
                    "pageObjectId": slide_object_id,
                    "size": {
                        "width": {"magnitude": ew, "unit": "EMU"},
                        "height": {"magnitude": eh, "unit": "EMU"},
                    },
                    "transform": {
                        "scaleX": 1,
                        "scaleY": 1,
                        "translateX": tx,
                        "translateY": ty,
                        "unit": "EMU",
                    },
                },
            }
        }

    requests: list[dict] = [
        {
            "createSlide": {
                "objectId": first_blank_id,
                "insertionIndex": 0,
                "slideLayoutReference": {"predefinedLayout": "BLANK"},
            }
        },
        {"deleteObject": {"objectId": default_slide_id}},
        image_req(first_blank_id, file_ids[0], image_png_bytes[0]),
    ]

    for i in range(1, len(file_ids)):
        sid = f"music_slide_{i}_{uuid.uuid4().hex[:8]}"
        requests.append(
            {
                "createSlide": {
                    "objectId": sid,
                    "slideLayoutReference": {"predefinedLayout": "BLANK"},
                }
            }
        )
        requests.append(image_req(sid, file_ids[i], image_png_bytes[i]))

    try:
        slides.presentations().batchUpdate(
            presentationId=presentation_id, body={"requests": requests}
        ).execute()
    except HttpError as e:
        raise RuntimeError(f"Failed to populate slides: {e}") from e

    return f"https://docs.google.com/presentation/d/{presentation_id}/edit"
