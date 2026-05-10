"""FastAPI wrapper around vectorize.vectorize_bgr.

Endpoints:
  GET  /healthz         liveness probe
  POST /vectorize       multipart upload `file=<image>` -> JSON {lines, ...}

Run locally:
  uvicorn api:app --reload --port 10000
"""
from __future__ import annotations

import os
import logging
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import vectorize

logger = logging.getLogger("vectorize.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Reject uploads above this many bytes — protects the worker from OOM on a
# 50 MP photograph. 12 MB covers any sane floorplan PNG.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 12 * 1024 * 1024))

# Downscale very large images to keep peak memory bounded on small Render
# instances. The longer side is clamped to this; smaller images pass through.
MAX_LONG_SIDE = int(os.environ.get("MAX_LONG_SIDE", 2500))

app = FastAPI(title="floorplan vectorizer", version="1.0")

# Wide-open CORS for now — tighten to your frontend origin before going public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.post("/vectorize")
async def vectorize_endpoint(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file too large (>{MAX_UPLOAD_BYTES} bytes)")

    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "could not decode image (supported: png/jpg/webp/bmp)")

    h, w = bgr.shape[:2]
    long_side = max(h, w)
    if long_side > MAX_LONG_SIDE:
        ratio = MAX_LONG_SIDE / long_side
        bgr = cv2.resize(bgr, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)
        logger.info("resized %dx%d -> %dx%d", w, h, bgr.shape[1], bgr.shape[0])

    t0 = time.perf_counter()
    try:
        result = vectorize.vectorize_bgr(bgr, verbose=False)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:
        logger.exception("vectorize failed")
        raise HTTPException(500, "vectorize failed")
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info("vectorized %s (%dx%d) in %d ms -> %d segments",
                file.filename, bgr.shape[1], bgr.shape[0],
                elapsed_ms, result["stats"]["segment_count"])
    result["stats"]["elapsed_ms"] = elapsed_ms
    return result
