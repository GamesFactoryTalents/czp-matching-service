from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

import anthropic
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware

from models import BatchMatchRequest, BatchMatchResponse, MatchRequest, MatchResponse
from matching import match_candidate


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_client: Optional[anthropic.Anthropic] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is required")
    _client = anthropic.Anthropic(api_key=api_key)
    yield
    _client = None


app = FastAPI(
    title="CZP Matching Service",
    description="AI-powered candidate ↔ job matching for Careers Zone Portal",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Auth — simple bearer token (set SERVICE_API_KEY env var)
# ---------------------------------------------------------------------------

security = HTTPBearer(auto_error=False)


def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)):
    service_key = os.environ.get("SERVICE_API_KEY")
    if not service_key:
        return  # no key configured → open (dev mode)
    if not credentials or credentials.credentials != service_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "czp-matching"}


@app.post("/match", response_model=MatchResponse)
def match_single(
    req: MatchRequest,
    _: None = Security(verify_token),
):
    """Match a single candidate against a single job."""
    if not req.cv_text.strip():
        raise HTTPException(status_code=400, detail="cv_text is required")
    if not req.job_description.strip():
        raise HTTPException(status_code=400, detail="job_description is required")

    try:
        return match_candidate(_client, req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/match/batch", response_model=BatchMatchResponse)
def match_batch(
    req: BatchMatchRequest,
    _: None = Security(verify_token),
):
    """Match multiple candidates against one job (sequential, rate-limit safe)."""
    results = []
    for cand in req.candidates:
        single_req = MatchRequest(
            candidate_id=cand.get("candidate_id", ""),
            job_id=req.job_id,
            cv_text=cand.get("cv_text", ""),
            job_description=req.job_description,
            job_title=req.job_title,
            required_skills=req.required_skills,
            candidate_tags=cand.get("tags", []),
            location=cand.get("location"),
            job_location=req.job_location,
            job_seniority=req.job_seniority,
        )
        if not single_req.cv_text.strip():
            continue
        try:
            result = match_candidate(_client, single_req)
            results.append(result)
        except Exception:
            continue  # skip failed candidates, don't abort batch

    # Sort by score descending
    results.sort(key=lambda x: x.score, reverse=True)

    return BatchMatchResponse(
        job_id=req.job_id,
        matches=results,
        total=len(results),
    )
