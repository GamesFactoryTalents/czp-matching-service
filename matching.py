import json
import anthropic
from models import MatchRequest, MatchResponse
from datetime import datetime, timezone


SYSTEM_PROMPT = """You are an expert games industry recruiter with 15+ years of experience.
Your job is to evaluate how well a candidate's profile matches a job opening.
You understand the games industry deeply — the difference between engines, platforms, genres,
and what seniority really means in studios of different sizes.
Always return valid JSON only — no markdown, no explanation outside the JSON."""

MATCH_PROMPT = """Evaluate this candidate for the job opening below.

## JOB
Title: {job_title}
Location: {job_location}
Seniority required: {job_seniority}
Required skills: {required_skills}

Job Description:
{job_description}

## CANDIDATE PROFILE
ID: {candidate_id}
Current title: {current_job_title}
Category/Discipline: {category}
Self-reported seniority: {seniority}
Total years experience: {experience_years}
Years in gaming specifically: {years_in_gaming}
Location: {candidate_location}
Conference attendance: {attendance_mode}
Open to relocation: {is_open_to_relocation}
Expected salary: {expected_salary}
Employment types sought: {employment_types}
Work preferences: {work_preferences}

Specializations: {candidate_tags}
Skills: {skills}
Platforms: {platforms}
Engines: {engines}
Genres: {genres}

Shipped titles / apps: {game_titles}
Achievements: {achievements}
Motivation for change: {motivation}
Expectations from next role: {expectations}
Dream job: {dream_job}

CV:
{cv_text}

## YOUR TASK
Analyze this candidate holistically against the job requirements. Consider:
1. **Depth of relevant experience** — shipped titles and achievements beat years alone
2. **Seniority match** — use years in gaming, current title, and shipped work as signals
3. **Technical fit** — engines, platforms, genres alignment with the role
4. **Location / logistics** — attendance mode, relocation willingness, work preferences
5. **Motivation fit** — does what they want align with what this role offers?
6. **Hidden strengths** — things tag-matching would miss

Return ONLY this JSON (no markdown, no extra text):
{{
  "score": <integer 0-100>,
  "seniority_match": <true|false>,
  "location_match": <true|false>,
  "strengths": [<up to 5 short strings>],
  "gaps": [<up to 5 short strings, empty array if none>],
  "summary": "<2-3 sentence plain English assessment>",
  "recommendation": "<Shortlist|Review|Pass>"
}}

Scoring guide:
- 80-100: Strong match, shortlist immediately
- 60-79: Good match worth reviewing, minor gaps
- 40-59: Partial match, significant gaps
- 0-39: Poor match, pass

Be honest about gaps — a false positive wastes recruiter time."""


def _fmt_list(items: list) -> str:
    return ", ".join(items) if items else "Not specified"


def _fmt(value, fallback: str = "Not specified") -> str:
    if value is None:
        return fallback
    return str(value)


def build_prompt(req: MatchRequest) -> str:
    exp = f"{req.experience_years} years" if req.experience_years is not None else "Not specified"
    gaming_exp = f"{req.years_in_gaming} years" if req.years_in_gaming is not None else "Not specified"
    relocation = {True: "Yes", False: "No"}.get(req.is_open_to_relocation, "Not specified")

    return MATCH_PROMPT.format(
        job_title=req.job_title,
        job_location=_fmt(req.job_location),
        job_seniority=_fmt(req.job_seniority),
        required_skills=_fmt_list(req.required_skills) if req.required_skills else "See job description",
        job_description=req.job_description[:4000],
        candidate_id=req.candidate_id,
        current_job_title=_fmt(req.current_job_title),
        category=_fmt(req.category),
        seniority=_fmt(req.seniority),
        experience_years=exp,
        years_in_gaming=gaming_exp,
        candidate_location=_fmt(req.location),
        attendance_mode=_fmt(req.attendance_mode),
        is_open_to_relocation=relocation,
        expected_salary=_fmt(req.expected_salary),
        employment_types=_fmt_list(req.employment_types),
        work_preferences=_fmt_list(req.work_preferences),
        candidate_tags=_fmt_list(req.candidate_tags),
        skills=_fmt_list(req.skills),
        platforms=_fmt_list(req.platforms),
        engines=_fmt_list(req.engines),
        genres=_fmt_list(req.genres),
        game_titles=_fmt(req.game_titles),
        achievements=_fmt(req.achievements),
        motivation=_fmt(req.motivation),
        expectations=_fmt(req.expectations),
        dream_job=_fmt(req.dream_job),
        cv_text=req.cv_text[:6000],
    )


def match_candidate(client: anthropic.Anthropic, req: MatchRequest) -> MatchResponse:
    prompt = build_prompt(req)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    # Strip markdown code fences if model adds them despite instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    data = json.loads(raw)

    return MatchResponse(
        candidate_id=req.candidate_id,
        job_id=req.job_id,
        score=int(data.get("score", 0)),
        seniority_match=bool(data.get("seniority_match", False)),
        location_match=bool(data.get("location_match", False)),
        strengths=data.get("strengths", []),
        gaps=data.get("gaps", []),
        summary=data.get("summary", ""),
        recommendation=data.get("recommendation", "Review"),
        calculated_at=datetime.now(timezone.utc).isoformat(),
    )
