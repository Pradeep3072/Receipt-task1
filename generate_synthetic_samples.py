"""Create synthetic PDFs from the expected-results table in the assessment brief."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample_docs"
EXPECTED_CSV = ROOT / "expected_results.csv"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _description(row: dict[str, str]) -> str:
    vendor = row["vendor"]
    category = row["category"]
    if vendor == "CarPark Central":
        return "Parking fee"
    if row["document_type"] == "Travel ticket":
        return "Air travel ticket"
    if vendor in {"CityCab Singapore", "Metro Cabs Bengaluru"}:
        return "Taxi / ride-hailing fare"
    if row["document_type"] == "Hotel folio":
        return "Room and accommodation charges"
    if category == "Food & Beverage":
        return "Food and beverage charges"
    if category == "Office Supplies" and vendor == "TechHub Pte Ltd":
        return "Office equipment and accessories"
    if category == "Office Supplies":
        return "Stationery and office supplies"
    if vendor == "StarLink Mobile":
        return "Mobile phone service charges"
    return "Expense charges"


def _document_lines(row: dict[str, str]) -> list[str]:
    if row["document_type"] == "Not an expense document":
        return [
            "COMMUNITY NEWSLETTER",
            "Announcements and event notices",
            "This document contains informational updates.",
        ]
    title = row["document_type"].upper()
    vendor = row["vendor"]
    lines = [title, vendor, f"Date: {row['date']}", ""]
    lines.append(_description(row))
    lines.append("")
    amount = row["amount"]
    if amount:
        lines.append(f"TOTAL: {row['currency']} {float(amount):.2f}")
    else:
        lines.append(f"TOTAL: {row['currency']} [amount unreadable]")
    return lines


def _draw_scan(lines: list[str]) -> Image.Image:
    image = Image.new("RGB", (1224, 1584), "white")
    draw = ImageDraw.Draw(image)
    left, right = 120, 1104
    top = 120
    draw.rectangle((left, top, right, 1464), outline=(170, 170, 170), width=3)

    y = top + 70
    for line_index, line in enumerate(lines):
        if not line:
            y += 34
            continue
        is_title = line_index == 0
        font = _font(54 if is_title else (44 if line_index == 1 else 34), bold=is_title)
        color = (25, 25, 25) if line_index < 3 else (45, 45, 45)
        draw.text((left + 60, y), line, fill=color, font=font)
        y += 86 if is_title or line_index == 1 else 74
        if line.startswith("Date:"):
            draw.line((left + 55, y + 10, right - 55, y + 10), fill=(150, 150, 150), width=2)
            y += 40
    return image


def _draw_digital(pdf: canvas.Canvas, lines: list[str]) -> None:
    page_width, page_height = letter
    left = 64
    y = page_height - 96
    pdf.setStrokeColorRGB(0.72, 0.76, 0.80)
    pdf.roundRect(44, 44, page_width - 88, page_height - 88, 12, stroke=1, fill=0)
    for index, line in enumerate(lines):
        if not line:
            y -= 22
            continue
        if index == 0:
            pdf.setFont("Helvetica-Bold", 19)
            pdf.drawString(left, y, line)
            y -= 40
        elif index == 1:
            pdf.setFont("Helvetica-Bold", 16)
            pdf.drawString(left, y, line)
            y -= 36
        elif line.startswith("TOTAL:"):
            pdf.setFont("Helvetica-Bold", 14)
            pdf.drawString(left, y, line)
            y -= 32
        else:
            pdf.setFont("Helvetica", 12)
            pdf.drawString(left, y, line)
            y -= 26


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    with EXPECTED_CSV.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    for row in rows:
        path = SAMPLE_DIR / row["file"]
        pdf = canvas.Canvas(str(path), pagesize=letter, pageCompression=1)
        lines = _document_lines(row)
        if row["source"] == "OCR":
            scan = _draw_scan(lines)
            buffer = io.BytesIO()
            scan.save(buffer, format="PNG", optimize=True)
            buffer.seek(0)
            pdf.drawImage(ImageReader(buffer), 0, 0, width=letter[0], height=letter[1])
        else:
            _draw_digital(pdf, lines)
        pdf.showPage()
        pdf.save()

    print(f"Created {len(rows)} synthetic PDFs in {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
