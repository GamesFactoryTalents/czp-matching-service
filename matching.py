import json
import anthropic
from models import MatchRequest, MatchResponse
from taxonomy import taxonomy_context
from datetime import datetime, timezone


SYSTEM_PROMPT = f"""You are an expert games industry recruiter with 15+ years of experience.
Your job is to evaluate how well a candidate's profile matches a job opening.

You understand the games industry deeply:
- Shipped titles are the strongest proof of seniority — a candidate with 2 shipped mobile games
  outweighs one with 5 years at a studio that never launched
- Engine expertise matters: Unity and Unreal are different pipelines; switching has real cost
- Genre experience matters for design/product roles (Casino ≠ Casual ≠ AAA RPG)
- Seniority labels are unreliable — verify against actual CV evidence
- Salary mismatch is a hard blocker when both sides have data; skip salary scoring when data is missing

{taxonomy_context()}

Always return valid JSON only — no markdown, no explanation outside the JSON."""

MATCH_PROMPT = """Evaluate this candidate for the job opening below.
Read the CV carefully — it is the most important signal.

## JOB
Title: {job_title}
Seniority required: {job_seniority}
Country: {job_country}
City: {job_city}
Location (combined): {job_location}
Salary range: {job_salary}
Required skills: {required_skills}
Required specialities: {job_specialities}
Required platforms: {job_platforms}
Required engines: {job_engines}
Required genres: {job_genres}
Required art styles: {job_art_styles}

Job Description:
{job_description}

## CANDIDATE PROFILE
Current title: {current_job_title}
Category/Discipline: {category}
Self-reported seniority: {seniority}
Total years experience: {experience_years}
Years in gaming specifically: {years_in_gaming}
Country: {candidate_location}
Open to relocation: {is_open_to_relocation}
Conference attendance: {attendance_mode}
Expected salary: {expected_salary}
Employment types sought: {employment_types}
Work preferences: {work_preferences}

Specializations: {candidate_tags}
Skills: {skills}
Platforms: {platforms}
Engines: {engines}
Genres: {genres}
Art styles: {art_styles}

Shipped titles / apps: {game_titles}
Tasks in previous roles: {tasks_and_responsibilities}
Achievements: {achievements}
Motivation for change: {motivation}
Expectations from next role: {expectations}
Dream job: {dream_job}

CV (read this carefully — primary evidence of real experience):
{cv_text}

## SCORING INSTRUCTIONS
Assess holistically. Do NOT just count tag overlaps. Evidence from the CV overrides self-reported fields.

Priority order:
1. CV evidence of relevant experience depth (highest weight)
2. Shipped titles / published work — proof of delivery
3. Seniority match (CV-verified, not just self-reported label)
4. Skills, engines, platforms alignment with job requirements
5. Specialities and category fit
6. Location / work preference compatibility
7. Salary fit (only score this if BOTH candidate expected salary AND job salary range are provided;
   if either is missing, ignore salary entirely — do not penalise)
8. Employment type, motivation, expectations (lowest weight)

Key rules:
- A candidate one level above the required seniority is still a strong match
- Missing fields on either side = skip that dimension, do not penalise
- Be honest about gaps — a false positive wastes recruiter time
- Strengths and gaps should be specific (e.g. "5 years Unity Mobile" not just "has Unity experience")

Return ONLY this JSON (no markdown, no extra text):
{{
  "score": <integer 0-100>,
  "seniority_match": <true|false>,
  "location_match": <true|false>,
  "strengths": [<up to 5 specific strings>],
  "gaps": [<up to 5 specific strings, empty array if none>],
  "summary": "<2-3 sentence plain English assessment, cite CV evidence>",
  "recommendation": "<Shortlist|Review|Pass>"
}}

Scoring guide:
- 80-100: Strong match, shortlist immediately
- 60-79: Good match worth reviewing, minor gaps
- 40-59: Partial match, significant gaps
- 0-39: Poor match, pass"""


def _fmt_list(items: list) -> str:
    return ", ".join(items) if items else "Not specified"


def _fmt(value, fallback: str = "Not specified") -> str:
    if value is None:
        return fallback
    return str(value)


def _fmt_salary(min_val, max_val) -> str:
    if min_val is None and max_val is None:
        return "Not provided"
    if min_val and max_val:
        return f"{min_val:,.0f} – {max_val:,.0f}"
    if min_val:
        return f"from {min_val:,.0f}"
    return f"up to {max_val:,.0f}"


def build_prompt(req: MatchRequest) -> str:
    exp = f"{req.experience_years} years" if req.experience_years is not None else "Not specified"
    gaming_exp = f"{req.years_in_gaming} years" if req.years_in_gaming is not None else "Not specified"
    relocation = {True: "Yes", False: "No"}.get(req.is_open_to_relocation, "Not specified")

    return MATCH_PROMPT.format(
        job_title=req.job_title,
        job_seniority=_fmt(req.job_seniority),
        job_country=_fmt(req.job_country),
        job_city=_fmt(req.job_city),
        job_location=_fmt(req.job_location),
        job_salary=_fmt_salary(req.job_salary_min, req.job_salary_max),
        required_skills=_fmt_list(req.required_skills) if req.required_skills else "See job description",
        job_specialities=_fmt_list(getattr(req, 'job_specialities', []) if hasattr(req, 'job_specialities') else []),
        job_platforms=_fmt_list(req.job_platforms),
        job_engines=_fmt_list(req.job_engines),
        job_genres=_fmt_list(req.job_genres),
        job_art_styles=_fmt_list(req.job_art_styles),
        job_description=req.job_description[:4000],
        current_job_title=_fmt(req.current_job_title),
        category=_fmt(req.category),
        seniority=_fmt(req.seniority),
        experience_years=exp,
        years_in_gaming=gaming_exp,
        candidate_location=_fmt(req.location),
        is_open_to_relocation=relocation,
        attendance_mode=_fmt(req.attendance_mode),
        expected_salary=_fmt(req.expected_salary),
        employment_types=_fmt_list(req.employment_types),
        work_preferences=_fmt_list(req.work_preferences),
        candidate_tags=_fmt_list(req.candidate_tags),
        skills=_fmt_list(req.skills),
        platforms=_fmt_list(req.platforms),
        engines=_fmt_list(req.engines),
        genres=_fmt_list(req.genres),
        art_styles=_fmt_list(req.art_styles),
        game_titles=_fmt(req.game_titles),
        tasks_and_responsibilities=_fmt(req.tasks_and_responsibilities),
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
        max_tokens=2048,
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
