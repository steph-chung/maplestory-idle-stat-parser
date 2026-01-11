# app/ocr.py
from app.extractor import extract_stats


def extract_stats(image_path: str, template_path: str) -> dict:
    return extract_stats(
        image_path=image_path,
        cp_template_path=template_path,
    )
