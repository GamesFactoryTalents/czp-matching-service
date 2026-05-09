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
    location: Optional[str] = None              # candidate's city/country
    job_location: Optional[str] = None
    job_seniority: Optional[str] = None
    # Additional Zoho candidate fields
    current_job_title: Optional[str] = None     # Current_Job_Title in Zoho
    current_employer: Optional[str] = None      # Current_Employer in Zoho
    experience_years: Optional[float] = None    # Experience_in_Years in Zoho
    skill_set: Optional[str] = None             # Skill_Set text field in Zoho
    highest_qualification: Optional[str] = None # Highest_Qualification_Held in Zoho
    linkedin_url: Optional[str] = None          # LinkedIn profile URL
    attendance_mode: Optional[str] = None       # "Onsite" | "Online" for the conference


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
    required_skills: List[str] = []
    job_location: Optional[str] = None
    job_seniority: Optional[str] = None
    candidates: List[dict]        # [{candidate_id, cv_text, tags, location, current_job_title, current_employer, experience_years, skill_set, highest_qualification, linkedin_url, attendance_mode}]


class BatchMatchResponse(BaseModel):
    job_id: str
    matches: List[MatchResponse]
    total: int
