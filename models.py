from pydantic import BaseModel
from typing import Optional, List


class MatchRequest(BaseModel):
    candidate_id: str
    job_id: str
    cv_text: str
    job_description: str
    job_title: str
    required_skills: List[str] = []
    candidate_tags: List[str] = []
    location: Optional[str] = None
    job_location: Optional[str] = None
    job_seniority: Optional[str] = None


class MatchResponse(BaseModel):
    candidate_id: str
    job_id: str
    score: int                    # 0-100
    seniority_match: bool
    location_match: bool
    strengths: List[str]
    gaps: List[str]
    summary: str
    recommendation: str           # Shortlist | Review | Pass
    calculated_at: str


class BatchMatchRequest(BaseModel):
    job_id: str
    job_title: str
    job_description: str
    required_skills: list[str] = []
    job_location: Optional[str] = None
    job_seniority: Optional[str] = None
    candidates: List[dict]        # [{candidate_id, cv_text, tags, location}]


class BatchMatchResponse(BaseModel):
    job_id: str
    matches: List[MatchResponse]
    total: int
