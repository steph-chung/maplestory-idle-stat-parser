# app/tasks.py
from __future__ import annotations

import os
from typing import Any, Dict, List

from app.celery_app import celery_app
from app.extractor import extract_stats

TEMPLATE_PATH = os.environ.get("CP_TEMPLATE_PATH", "cp_template.png")


@celery_app.task(bind=True)
def extract_multi_task(self, items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    items: [{ "filename": "...", "path": "/shared/uploads/..." }, ...]

    Returns:
      [
        {"filename": "...", "stats": {...}, "error": None},
        {"filename": "...", "stats": {}, "error": "message"},
        ...
      ]

    Notes:
      - This task attempts each file independently.
      - It best-effort deletes the uploaded file after processing.
    """
    out: List[Dict[str, Any]] = []

    for it in items:
        filename = it.get("filename") or "upload.png"
        path = it.get("path")

        try:
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

            stats = extract_stats(
                image_path=path,
                cp_template_path=TEMPLATE_PATH,
            )
            out.append({"filename": filename, "stats": stats, "error": None})

        except Exception as e:
            out.append({"filename": filename, "stats": {}, "error": str(e)})

        finally:
            # Cleanup upload
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

    return out
