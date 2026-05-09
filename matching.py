import json
import anthropic
from models import MatchRequest, MatchResponse
from datetime import datetime, timezone


SYSTEM_PROMPT = """You are an expert games industry recruiter with 15+ years of experience.
Your job is to evaluate how well a candidate's CV matches a job opening.
You read CVs carefully and assess real experience depth, not just keyword presence.
Always return valid JSON only — no markdown, no explanation outside the JSON."""

MATCH_PROMPT = """Evaluate this candidate for the job opening below.

## JOB
Title: {job_title}
Location: {job_location}
Seniority: {job_seniority}
Required skills: {required_skills}

Job Description:
{job_description}

## CANDIDATE
ID: {candidate_id}
Current title: {current_job_title}
Current employer: {current_employer}
Years of experience: {experience_years}
Highest qualification: {highest_qualification}
Location: {candidate_location}
Conference attendance: {attendance_mode}
Tags/Specialities: {candidate_tags}
Skill set (self-reported): {skill_set}

CV:
{cv_text}

## YOUR TASK
Analyze the candidate's actual experience from their CV against the job requirements.
Focus on:
- Depth of relevant experience (not just keyword presence)
- Seniority level match (use years of experience and current title as signals)
- Location compatibility (consider conference attendance mode if relevant)
- Skill gaps that would block performance in this role
- Hidden strengths the tag matching might miss

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
- 60-79: Good match worth reviewing, some gaps
- 40-59: Partial match, significant gaps
- 0-39: Poor match, pass

Be honest about gaps — a false positive wastes recruiter time."""


def build_prompt(req: MatchRequest) -> str:
    exp = f"{req.experience_years} years" if req.experience_years is not None else "Not specified"
    return MATCH_PROMPT.format(
        job_title=req.job_title,
        job_location=req.job_location or "Not specified",
        job_seniority=req.job_seniority or "Not specified",
        required_skills=", ".join(req.required_skills) if req.required_skills else "See job description",
        job_description=req.job_description[:4000],
        candidate_id=req.candidate_id,
        current_job_title=req.current_job_title or "Not specified",
        current_employer=req.current_employer or "Not specified",
        experience_years=exp,
        highest_qualification=req.highest_qualification or "Not specified",
        candidate_location=req.location or "Not specified",
        attendance_mode=req.attendance_mode or "Not specified",
        candidate_tags=", ".join(req.candidate_tags) if req.candidate_tags else "None provided",
        skill_set=req.skill_set or "Not provided",
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
