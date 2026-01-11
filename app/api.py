# app/api.py
import os
import uuid
from typing import List

from celery.result import AsyncResult
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.celery_app import celery_app
from app.merge import merge_job_result
from app.tasks import extract_multi_task

# -------------------------
# Configuration
# -------------------------

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
}

MAX_FILES_PER_JOB = int(os.environ.get("MAX_FILES_PER_JOB", "20"))


# -------------------------
# FastAPI app
# -------------------------

app = FastAPI(title="OCR Stats API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# Routes
# -------------------------


@app.post("/v1/jobs")
async def create_job(files: List[UploadFile] = File(...)):
    """
    Submit one or more screenshots for OCR.

    Returns:
      {"job_id": "<celery-task-id>", "status_url": "/v1/jobs/<id>", "summary_url": "/v1/jobs/<id>/summary"}
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    if len(files) > MAX_FILES_PER_JOB:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Max is {MAX_FILES_PER_JOB}.",
        )

    items = []

    for f in files:
        if f.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=415, detail=f"Unsupported file type: {f.content_type}"
            )

        filename = f.filename or "upload.png"
        job_filename = f"{uuid.uuid4().hex}_{filename}"
        job_path = os.path.join(UPLOAD_DIR, job_filename)

        try:
            contents = await f.read()
            with open(job_path, "wb") as out:
                out.write(contents)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

        items.append({"filename": filename, "path": job_path})

    async_result = extract_multi_task.delay(items)

    return {
        "job_id": async_result.id,
        "status_url": f"/v1/jobs/{async_result.id}",
        "summary_url": f"/v1/jobs/{async_result.id}/summary",
    }


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str):
    """
    Poll job status or retrieve raw per-file results.
    """
    res = AsyncResult(job_id, app=celery_app)

    response = {"job_id": job_id, "state": res.state}

    if res.state == "SUCCESS":
        response["result"] = res.result
    elif res.state == "FAILURE":
        response["error"] = str(res.result)

    return response


@app.get("/v1/jobs/{job_id}/summary")
def get_job_summary(job_id: str):
    """
    Returns a merged/aggregated view of the job output.

    If job isn't finished, returns state only.
    If finished, returns:
      - merged_stats: {stat: value}
      - merged_detail: {stat: {value, method, coverage}}
      - files: per-file stats/errors
    """
    res = AsyncResult(job_id, app=celery_app)

    if res.state != "SUCCESS":
        payload = {"job_id": job_id, "state": res.state}
        if res.state == "FAILURE":
            payload["error"] = str(res.result)
        return payload

    try:
        summary = merge_job_result(res.result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to summarize result: {e}")

    return {
        "job_id": job_id,
        "state": "SUCCESS",
        "summary": summary,
    }


@app.get("/health")
def health():
    return {"ok": True}
