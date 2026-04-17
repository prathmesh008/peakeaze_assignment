"""
main.py — FastAPI application for the Customer Support Triage & Action system.

Endpoints
---------
POST /input            — Submit raw customer text for processing.
POST /input/upload     — Upload a PDF or text file for processing.
POST /process/{id}     — Execute the multi-step agentic workflow synchronously.
GET  /results/{id}     — Retrieve stored processing results.
GET  /health           — Liveness probe.
"""

from __future__ import annotations

import io
import logging
import uuid
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from PyPDF2 import PdfReader

from database import store
from workflow import run_workflow


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Customer Support Triage & Action API",
    description=(
        "An agentic, multi-step AI pipeline that classifies customer "
        "messages, extracts structured data, and returns confident results."
    ),
    version="1.0.0",
)


# Models


class InputRequest(BaseModel):
    """Payload accepted by ``POST /input``."""

    text: str = Field(
        ...,
        min_length=1,
        description="Raw customer support message to process.",
        examples=["My laptop screen is flickering after the latest Windows update."],
    )


class InputResponse(BaseModel):
    """Returned by ``POST /input`` and ``POST /input/upload`` after storing the record."""

    input_id: str
    status: str


class ProcessResponse(BaseModel):
    """Returned by ``POST /process/{input_id}`` after workflow completion."""

    input_id: str
    status: str
    result: dict
    steps: list


class ResultsResponse(BaseModel):
    """Returned by ``GET /results/{input_id}``."""

    input_id: str
    status: str
    text: str
    result: Optional[dict]
    steps: Optional[list]
    created_at: str
    completed_at: Optional[str]


# Endpoints


@app.post("/input", response_model=InputResponse, status_code=201)
def submit_input(payload: InputRequest):
    input_id = uuid.uuid4()
    store.create(input_id, payload.text)
    logger.info("New input stored: %s", input_id)
    return InputResponse(input_id=str(input_id), status="pending")


@app.post("/input/upload", response_model=InputResponse, status_code=201)
async def upload_document(file: UploadFile = File(...)):
    """Accept a ``.pdf`` or ``.txt`` file, extract its text, and store with status *pending*."""
    if file.filename is None:
        raise HTTPException(status_code=400, detail="Filename is required.")

    suffix = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in ("pdf", "txt"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{suffix}'. Accepted: .pdf, .txt",
        )

    raw = await file.read()

    if suffix == "txt":
        text = raw.decode("utf-8", errors="replace").strip()
    else:  # pdf
        reader = PdfReader(io.BytesIO(raw))
        text = "\n".join(
            page.extract_text() or "" for page in reader.pages
        ).strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Could not extract any text from the uploaded file.",
        )

    input_id = uuid.uuid4()
    store.create(input_id, text)
    logger.info("Document uploaded (%s) — stored as %s", file.filename, input_id)
    return InputResponse(input_id=str(input_id), status="pending")


@app.post("/process/{input_id}", response_model=ProcessResponse)
def process_input(input_id: uuid.UUID, force: bool = False):
    """Retrieve the pending input and run the full agentic workflow.

    Pass ``?force=true`` to re-process an already-completed input.
    """
    record = store.get(input_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Input ID not found.")

    if record["status"] == "completed" and not force:
        logger.info("Input %s already processed — returning cached result.", input_id)
        return ProcessResponse(
            input_id=str(input_id),
            status="completed",
            result=record["result"],
            steps=record["steps"] or [],
        )

    store.update(input_id, status="processing")
    logger.info("Processing %sstarted for input %s", "RE-" if force else "", input_id)

    try:
        workflow_output = run_workflow(record["text"])
    except Exception as exc:
        store.update(input_id, status="failed")
        logger.exception("Workflow failed for input %s", input_id)
        raise HTTPException(
            status_code=500,
            detail=f"Workflow processing failed: {exc}",
        ) from exc

    result = workflow_output["result"]
    steps = workflow_output["steps"]
    store.update(input_id, status="completed", result=result, steps=steps)
    logger.info("Processing completed for input %s", input_id)

    return ProcessResponse(
        input_id=str(input_id),
        status="completed",
        result=result,
        steps=steps,
    )


@app.get("/results/{input_id}", response_model=ResultsResponse)
def get_results(input_id: uuid.UUID):
    """Return the full stored record, including results if available."""
    record = store.get(input_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Input ID not found.")

    return ResultsResponse(
        input_id=record["input_id"],
        status=record["status"],
        text=record["text"],
        result=record["result"],
        steps=record["steps"],
        created_at=record["created_at"],
        completed_at=record["completed_at"],
    )


@app.get("/health")
def health_check():
    """Simple liveness probe."""
    return {"status": "ok"}
