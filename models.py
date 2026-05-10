from pydantic import BaseModel
from typing import Optional, List


class MatchRequest(BaseModel):
    # Core identifiers
    candidate_id: str
    job_id: str

    # Text content for semantic matching
    cv_text: str
    job_description: str

    # Job fields
    job_title: str
    required_skills: List[str] = []
    job_location: Optional[str] = None          # city + country combined
    job_country: Optional[str] = None
    job_city: Optional[str] = None
    job_seniority: Optional[str] = None         # Junior | Mid | Senior | Lead | Director
    job_salary_min: Optional[float] = None      # offered salary range min
    job_salary_max: Optional[float] = None      # offered salary range max
    job_specialities: List[str] = []            # required specialities
    job_art_styles: List[str] = []              # required art styles
    job_platforms: List[str] = []              # required platforms
    job_engines: List[str] = []                # required engines
    job_genres: List[str] = []                 # required genres

    # Candidate — basic
    current_job_title: Optional[str] = None     # JobTitle in OutSystems
    category: Optional[str] = None             # broad discipline: Engineering, Art, Design, etc.
    seniority: Optional[str] = None            # candidate's self-reported seniority level
    experience_years: Optional[float] = None   # TotalYears
    years_in_gaming: Optional[float] = None    # YearsInGaming — gaming-specific experience
    location: Optional[str] = None             # City + Country
    attendance_mode: Optional[str] = None      # "Onsite" | "Online" for the conference
    is_open_to_relocation: Optional[bool] = None
    expected_salary: Optional[str] = None
    linkedin_url: Optional[str] = None

    # Candidate — structured gaming tags (lists)
    candidate_tags: List[str] = []             # Specializations
    skills: List[str] = []                     # Skills
    platforms: List[str] = []                  # PC | Mobile | Console | VR | etc.
    engines: List[str] = []                    # Unity | Unreal | Godot | etc.
    genres: List[str] = []                     # RPG | FPS | Casual | Strategy | etc.
    art_styles: List[str] = []                 # Art styles (relevant for Art & Animation roles)
    employment_types: List[str] = []           # Full-time | Part-time | Contract | Freelance
    work_preferences: List[str] = []           # Remote | Hybrid | Onsite

    # Candidate — free text signals
    game_titles: Optional[str] = None              # GameTitlesOrApps — shipped titles (strong seniority signal)
    tasks_and_responsibilities: Optional[str] = None  # What they did in previous roles
    motivation: Optional[str] = None               # Why looking for a new role
    expectations: Optional[str] = None             # What they want from next role
    dream_job: Optional[str] = None                # DreamJob
    achievements: Optional[str] = None             # Key career achievements


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
    next_steps: List[str]         # Recruiter actions: what to do next
    interview_questions: List[str]  # Questions to ask based on gaps/unknowns
    recruiter_brief: str          # Narrative recommendation for recruiter — why to interview and what to focus on
    recruiter_text: str           # Combined tooltip text: headline + summary + strengths + to explore
    calculated_at: str


class BatchMatchRequest(BaseModel):
    job_id: str
    job_title: str
    job_description: str
    required_skills: List[str] = []
    job_location: Optional[str] = None
    job_country: Optional[str] = None
    job_city: Optional[str] = None
    job_seniority: Optional[str] = None
    job_salary_min: Optional[float] = None
    job_salary_max: Optional[float] = None
    job_art_styles: List[str] = []
    job_platforms: List[str] = []
    job_engines: List[str] = []
    job_genres: List[str] = []
    candidates: List[dict]        # list of candidate dicts — all MatchRequest candidate fields accepted


class BatchMatchResponse(BaseModel):
    job_id: str
    matches: List[MatchResponse]
    total: int
