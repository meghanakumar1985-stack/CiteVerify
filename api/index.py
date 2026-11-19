"""CiteVerify FastAPI backend and production web interface."""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import run_pipeline

app = FastAPI(
    title="CiteVerify Engine",
    description="Automated scientific claim verification, citation grounding, and reference validity auditor.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HTML_PATH = BASE_DIR / "index.html"
STATIC_DIR = BASE_DIR / "static"
ASSETS_DIR = BASE_DIR / "assets"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


class VerificationResponse(BaseModel):
    total_claims: int
    supported_count: int
    contradicted_count: int
    outdated_count: int
    needs_review_count: int
    results: List[Dict[str, Any]]
    diagnostics: Dict[str, Any]


@app.get("/api/health")
async def health_check():
    """Health check endpoint reporting backend and external service statuses."""
    gemini_ok = bool(os.getenv("GEMINI_API_KEY"))
    groq_ok = bool(os.getenv("GROQ_API_KEY"))

    active_nli = "symbolic_rule_engine"
    if gemini_ok and groq_ok:
        active_nli = "dual_cascade (gemini_primary + groq_fallback)"
    elif gemini_ok:
        active_nli = "gemini_2.0_flash (1,000,000 TPM tier)"
    elif groq_ok:
        active_nli = "groq_lpu_fast_inference"

    return {
        "status": "operational",
        "engine": "CiteVerify-v2",
        "pdf_parser": "native_pymupdf (zero-docker)",
        "services": {
            "nli_intelligence": active_nli,
            "scholarly_apis": "crossref_semantic_scholar_arxiv",
        },
    }


@app.get("/api/demo")
async def run_demo():
    """Runs the verification pipeline on benchmark demonstration academic claims."""
    df, diagnostics = run_pipeline(b"%PDF-demo", use_stub=True)
    records = df.to_dict(orient="records")

    supported = sum(1 for r in records if "supported" in str(r.get("Verdict", "")).lower() and "partially" not in str(r.get("Verdict", "")).lower())
    partial = sum(1 for r in records if "partially" in str(r.get("Verdict", "")).lower())
    contradicted = sum(1 for r in records if "contradict" in str(r.get("Verdict", "")).lower())
    outdated = sum(1 for r in records if "outdated" in str(r.get("Verdict", "")).lower())
    needs_review = sum(1 for r in records if "review" in str(r.get("Review Status", "")).lower())

    return {
        "total_claims": len(records),
        "supported_count": supported,
        "partial_count": partial,
        "contradicted_count": contradicted,
        "outdated_count": outdated,
        "needs_review_count": needs_review,
        "results": records,
        "diagnostics": diagnostics,
    }


@app.post("/api/verify")
async def verify_pdf(
    file: Optional[UploadFile] = File(None),
    use_demo: bool = Form(False),
):
    """Processes an uploaded PDF research paper and verifies citations."""
    if use_demo or not file:
        pdf_bytes = b"%PDF-demo"
        stub_mode = True
    else:
        pdf_bytes = await file.read()
        stub_mode = False

    df, diagnostics = run_pipeline(pdf_bytes, use_stub=stub_mode)
    records = df.to_dict(orient="records")

    supported = sum(1 for r in records if "supported" in str(r.get("Verdict", "")).lower() and "partially" not in str(r.get("Verdict", "")).lower())
    partial = sum(1 for r in records if "partially" in str(r.get("Verdict", "")).lower())
    contradicted = sum(1 for r in records if "contradict" in str(r.get("Verdict", "")).lower())
    outdated = sum(1 for r in records if "outdated" in str(r.get("Verdict", "")).lower())
    needs_review = sum(1 for r in records if "review" in str(r.get("Review Status", "")).lower())

    return {
        "total_claims": len(records),
        "supported_count": supported,
        "partial_count": partial,
        "contradicted_count": contradicted,
        "outdated_count": outdated,
        "needs_review_count": needs_review,
        "results": records,
        "diagnostics": diagnostics,
    }


@app.get("/static/css/styles.css")
async def serve_static_css():
    """Explicit fallback server for CSS asset."""
    candidates = [
        STATIC_DIR / "css" / "styles.css",
        BASE_DIR / "public" / "static" / "css" / "styles.css",
        Path("static/css/styles.css"),
    ]
    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return Response(content=f.read(), media_type="text/css")
    return Response(content="/* CSS not found */", media_type="text/css", status_code=404)


@app.get("/static/js/app.js")
async def serve_static_js():
    """Explicit fallback server for JS asset."""
    candidates = [
        STATIC_DIR / "js" / "app.js",
        BASE_DIR / "public" / "static" / "js" / "app.js",
        Path("static/js/app.js"),
    ]
    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return Response(content=f.read(), media_type="application/javascript")
    return Response(content="// JS not found", media_type="application/javascript", status_code=404)


@app.get("/favicon.png")
async def get_favicon_png():
    fav = STATIC_DIR / "favicon.png"
    if not fav.exists():
        fav = BASE_DIR / "public" / "favicon.png"
    if fav.exists():
        return FileResponse(fav, media_type="image/png")
    return HTMLResponse(status_code=404)


@app.get("/favicon.ico")
async def get_favicon_ico():
    fav = STATIC_DIR / "favicon.png"
    if fav.exists():
        return FileResponse(fav, media_type="image/png")
    return HTMLResponse(status_code=404)


@app.get("/robots.txt")
async def get_robots_txt():
    rob = STATIC_DIR / "robots.txt"
    if rob.exists():
        return FileResponse(rob, media_type="text/plain")
    return HTMLResponse(status_code=404)


@app.get("/", response_class=HTMLResponse)
async def serve_webapp():
    """Serves the standalone HTML web interface directly from disk for instant live editing."""
    if HTML_PATH.exists():
        with open(HTML_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return "<html><body><h1>CiteVerify Ready</h1></body></html>"
