from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
import io
import os


def generate_pdf_report(outpath: str, filename: str, metrics: dict, image_paths: list):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    c = canvas.Canvas(outpath, pagesize=letter)
    width, height = letter

    c.setFont('Helvetica-Bold', 16)
    c.drawString(72, height - 72, f"CardioScanX - Sample Report")
    c.setFont('Helvetica', 12)
    c.drawString(72, height - 92, f"Filename: {filename}")

    y = height - 120
    for k, v in metrics.items():
        c.drawString(72, y, f"{k}: {v}")
        y -= 18

    # Insert images
    x = 72
    for img in image_paths:
        if os.path.exists(img):
            with Image.open(img) as im:
                # Resize to fit
                ratio = min((width - 144) / im.width, 0.4)
                w, h = int(im.width * ratio), int(im.height * ratio)
                img_r = ImageReader(im.resize((w, h)))
                y -= h + 12
                c.drawImage(img_r, x, y, width=w, height=h)
                if y < 120:
                    c.showPage()
                    y = height - 72
    c.save()
    return outpath
