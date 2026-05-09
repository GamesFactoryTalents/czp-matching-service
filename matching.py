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
Location: {candidate_location}
Tags/Skills selected: {candidate_tags}

CV:
{cv_text}

## YOUR TASK
Analyze the candidate's actual experience from their CV against the job requirements.
Focus on:
- Depth of relevant experience (not just keyword presence)
- Seniority level match
- Location compatibility
- Skill gaps that would block performance
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
    return MATCH_PROMPT.format(
        job_title=req.job_title,
        job_location=req.job_location or "Not specified",
        job_seniority=req.job_seniority or "Not specified",
        required_skills=", ".join(req.required_skills) if req.required_skills else "See job description",
        job_description=req.job_description[:4000],  # cap to avoid token overflow
        candidate_id=req.candidate_id,
        candidate_location=req.location or "Not specified",
        candidate_tags=", ".join(req.candidate_tags) if req.candidate_tags else "None provided",
        cv_text=req.cv_text[:6000],  # cap CV text
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
