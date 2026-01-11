# app/tasks.py
from __future__ import annotations

import base64
import os
import tempfile
from typing import Any, Dict, List

from app.celery_app import celery_app
from app.extractor import extract_stats

TEMPLATE_PATH = os.environ.get("CP_TEMPLATE_PATH", "cp_template.png")


@celery_app.task(bind=True)
def extract_multi_task(self, items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    items: [{ "filename": "...", "content_b64": "..." }, ...]

    Returns:
      [
        {"filename": "...", "stats": {...}, "error": None},
        {"filename": "...", "stats": {}, "error": "message"},
        ...
      ]
    """
    out: List[Dict[str, Any]] = []

    for it in items:
        filename = it.get("filename") or "upload.png"
        content_b64 = it.get("content_b64")
        tmp_path = None

        try:
            if not content_b64:
                raise ValueError("No image content provided")

            # Decode and write to temp file
            content = base64.b64decode(content_b64)
            ext = os.path.splitext(filename)[1] or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            stats = extract_stats(
                image_path=tmp_path,
                cp_template_path=TEMPLATE_PATH,
            )
            out.append({"filename": filename, "stats": stats, "error": None})

        except Exception as e:
            out.append({"filename": filename, "stats": {}, "error": str(e)})

        finally:
            # Cleanup temp file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    return out
