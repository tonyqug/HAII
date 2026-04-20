from __future__ import annotations

import html
import subprocess
import textwrap
from pathlib import Path
from typing import Iterable, Optional

import fitz
from PIL import Image, ImageDraw, ImageFont

from .utils import ensure_parent, safe_filename


PAGE_WIDTH = 1280
PAGE_HEIGHT = 720
MARGIN = 56
BG_COLOR = (255, 255, 255)
FG_COLOR = (20, 20, 20)
ACCENT_COLOR = (90, 90, 90)
LINE_SPACING = 8


def _get_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def render_text_image(
    text: str,
    output_path: Path,
    *,
    title: Optional[str] = None,
    footer: Optional[str] = None,
    width: int = PAGE_WIDTH,
    height: int = PAGE_HEIGHT,
) -> None:
    ensure_parent(output_path)
    image = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(image)
    title_font = _get_font(34)
    body_font = _get_font(26)
    footer_font = _get_font(20)

    y = MARGIN
    if title:
        title_lines = textwrap.wrap(title, width=50)[:3]
        for line in title_lines:
            draw.text((MARGIN, y), line, font=title_font, fill=FG_COLOR)
            y += 44
        y += 10
        draw.line((MARGIN, y, width - MARGIN, y), fill=(180, 180, 180), width=2)
        y += 24

    max_width_chars = 72
    lines: list[str] = []
    for para in (text or "").splitlines() or [""]:
        para = para.rstrip()
        if not para:
            lines.append("")
            continue
        wrapped = textwrap.wrap(para, width=max_width_chars, replace_whitespace=False, drop_whitespace=False) or [para]
        lines.extend(wrapped)

    max_lines = max(1, (height - y - 2 * MARGIN) // (body_font.size + LINE_SPACING))
    shown_lines = lines[:max_lines]
    if len(lines) > max_lines and shown_lines:
        shown_lines[-1] = shown_lines[-1].rstrip()[: max(0, len(shown_lines[-1]) - 1)] + "…"

    for line in shown_lines:
        draw.text((MARGIN, y), line, font=body_font, fill=FG_COLOR)
        y += body_font.size + LINE_SPACING

    if footer:
        footer_text = footer[:140]
        footer_y = height - MARGIN - 24
        draw.line((MARGIN, footer_y - 16, width - MARGIN, footer_y - 16), fill=(220, 220, 220), width=1)
        draw.text((MARGIN, footer_y), footer_text, font=footer_font, fill=ACCENT_COLOR)

    image.save(output_path, format="PNG")


def render_pdf_page_to_png(pdf_path: Path, page_number: int, output_path: Path, zoom: float = 1.6) -> None:
    ensure_parent(output_path)
    with fitz.open(pdf_path) as doc:
        page = doc[page_number]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pix.save(output_path)


def convert_pptx_to_pdf(pptx_path: Path, output_dir: Path, soffice_bin: str = "soffice", timeout_seconds: int = 120) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        soffice_bin,
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(output_dir),
        str(pptx_path),
    ]
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"LibreOffice conversion failed with code {completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    output_pdf = output_dir / (pptx_path.stem + ".pdf")
    if not output_pdf.exists():
        raise RuntimeError("LibreOffice reported success but no PDF was created.")
    return output_pdf


def html_page(title: str, body_html: str) -> str:
    escaped_title = html.escape(title)
    return f"""
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{escaped_title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.5; color: #111; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; box-shadow: 0 1px 6px rgba(0,0,0,.08); }}
    pre {{ white-space: pre-wrap; background: #f8f8f8; padding: 12px; border-radius: 8px; border: 1px solid #e5e5e5; }}
    .meta {{ color: #555; margin-bottom: 18px; }}
    .thumb-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 12px; }}
    a {{ color: #0a58ca; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
{body_html}
</body>
</html>
""".strip()


def relative_path(base_dir: Path, target: Path) -> str:
    return str(target.resolve().relative_to(base_dir.resolve()))
