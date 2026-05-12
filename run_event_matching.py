"""
Fetch the 123 most recent Applied-to-event candidates (last Talent Fair),
get their full profiles + CVs, and match against a target job.

Candidates are pre-filtered by category (including configured overlaps) before
the expensive AI matching step, so only relevant candidates are scored.

Usage: python3 run_event_matching.py ZR_640_JOB
"""
import sys
import os
import json
import io
import requests
import anthropic

# Load .env — always override so empty shell vars don't block the script
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
for _line in open(_env_path):
    _line = _line.strip()
    if "=" in _line and not _line.startswith("#"):
        _k, _v = _line.split("=", 1)
        if _v.strip():  # only set if .env has a real value
            os.environ[_k.strip()] = _v.strip()

ZOHO_BASE        = "https://recruit.zoho.eu/recruit/v2"
ZOHO_ACCOUNTS    = "https://accounts.zoho.eu/oauth/v2/token"
EVENT_CAND_COUNT = 123   # candidates in last event

CLIENT_ID     = os.environ["ZOHO_CLIENT_ID"]
CLIENT_SECRET = os.environ["ZOHO_CLIENT_SECRET"]
REFRESH_TOKEN = os.environ["ZOHO_REFRESH_TOKEN"]

# For each job category, list the candidate categories to include.
# The job's own category is always included automatically.
CATEGORY_OVERLAPS = {
    "Game Design":                ["Product & LiveOps", "Writing", "Production"],
    "Product & LiveOps":          ["Game Design", "Monetisation", "Data & Analytics", "Production"],
    "Programming & Engineering":  ["QA & Testing", "Data & Analytics", "Audio & Sound"],
    "Art & Animation":            ["UI & UX Design", "Game Design", "Production"],
    "UI & UX Design":             ["Art & Animation", "Game Design"],
    "Data & Analytics":           ["Product & LiveOps", "Monetisation", "Programming & Engineering"],
    "Monetisation":               ["Product & LiveOps", "Data & Analytics", "UA & Marketing", "Game Design"],
    "UA & Marketing":             ["Monetisation", "Player Support & Community", "Product & LiveOps", "Data & Analytics"],
    "Production":                 ["Business & Management", "Game Design", "Product & LiveOps", "Art & Animation"],
    "Business & Management":      ["Production", "UA & Marketing", "Product & LiveOps"],
    "QA & Testing":               ["Programming & Engineering"],
    "Audio & Sound":              ["Programming & Engineering"],
    "Localisation":               ["Writing", "Player Support & Community"],
    "Writing":                    ["Game Design", "Localisation"],
    "Player Support & Community": ["UA & Marketing", "Localisation"],
}


def _allowed_categories(job_category: str) -> set:
    """Return the set of candidate categories to include for a given job category."""
    norm = job_category.strip()
    allowed = {norm}
    allowed.update(CATEGORY_OVERLAPS.get(norm, []))
    return allowed


def _candidate_category(cand: dict) -> str:
    """Zoho UI label is CATEROGRY but the API field name is Pick_List_5."""
    val = cand.get("Pick_List_5") or cand.get("CATEROGRY") or ""
    if isinstance(val, dict):
        return val.get("name", "").strip()
    return str(val).strip()


def _job_category(job: dict) -> str:
    val = job.get("CATEGORY1") or ""
    if isinstance(val, dict):
        return val.get("name", "").strip()
    return str(val).strip()


def _parse_skills(raw) -> set:
    """Parse comma- or semicolon-separated string (or list) into a normalised lowercase set."""
    if not raw:
        return set()
    if isinstance(raw, list):
        # Each list item may itself be a semicolon-separated string (candidate SKILLS field)
        parts = []
        for v in raw:
            if isinstance(v, dict):
                parts.append(v.get("name", ""))
            else:
                parts.extend(str(v).replace(";", ",").split(","))
        items = parts
    else:
        items = str(raw).replace(";", ",").split(",")
    return {s.lower().strip() for s in items if s.strip()}


def _parse_work_prefs(raw) -> set:
    """Parse Work_Preferences or Work_Preferences1 into a normalised lowercase set."""
    if not raw:
        return set()
    if isinstance(raw, list):
        return {s.lower().strip() for s in raw if s}
    return {s.lower().strip() for s in str(raw).split(",") if s.strip()}


SENIORITY_ORDER = ["student/trainee", "junior", "mid", "senior", "lead", "director"]

SENIORITY_ACCEPT = {
    "student/trainee": {"student/trainee", "junior"},
    "junior":          {"student/trainee", "junior", "mid"},
    "mid":             {"junior", "mid", "senior"},
    "senior":          {"mid", "senior", "lead"},
    "lead":            {"senior", "lead", "director"},
    "director":        {"lead", "director"},
}

SENIORITY_NORMALISE = {
    # student/trainee
    "student":          "student/trainee",
    "trainee":          "student/trainee",
    "student/trainee":  "student/trainee",
    "intern":           "student/trainee",
    "graduate":         "student/trainee",
    # junior
    "junior":           "junior",
    "entry":            "junior",
    "entry-level":      "junior",
    # mid
    "mid":              "mid",
    "middle":           "mid",
    "intermediate":     "mid",
    "mid-level":        "mid",
    # senior
    "senior":           "senior",
    # lead
    "lead":             "lead",
    "principal":        "lead",
    "staff":            "lead",
    # director
    "director":         "director",
    "head":             "director",
    "vp":               "director",
    "c-level":          "director",
    "executive":        "director",
}


HAIKU_PROMPT = """You are a headhunter presenting a candidate to a hiring manager.
Your job is to make the recruiter excited to meet this person.
Be specific — every claim must be backed by something in the CV.
Do not invent. Do not generalise. No "strong communicator" or "team player".

## THE JOB

Description:
{job_description_tb}

Requirements:
{requirements_tb}

Responsibilities:
{responsibilities_tb}

Required skills: {required_skills}
Required specialities: {required_specialities}
Engines / Platforms / Genres: {engines_platforms_genres}

## CANDIDATE CV
{cv_text}

## YOUR TASK
1. Score 1–100 based on CV evidence vs job needs. Be honest — a weak CV should score low.
2. Recommendation: Shortlist (score ≥ 70) | Review (score 40–69)
3. headline: one punchy sentence selling the candidate to the recruiter (name the most impressive thing from the CV)
4. strengths: 2-4 specific strengths backed by CV evidence (cite years, tools, titles, numbers)
5. explore: one area worth exploring in the interview, framed as a positive opportunity — not a gap or criticism
6. summary: 2-3 sentences — open with the strongest alignment, then the most important thing the recruiter should know

Return JSON only — no markdown, no explanation:
{{"score": <int 1-100>, "recommendation": "<Shortlist|Review>", "headline": "<str>", "strengths": [<str>, ...], "explore": "<str>", "summary": "<str>"}}"""


def haiku_quick_score(client: anthropic.Anthropic, job: dict, cv_text: str) -> dict:
    """
    Score candidate CV against job using Haiku.
    Returns dict with score, recommendation, headline, strengths, explore, summary.
    Returns {"score": 0, "error": str} on failure.
    """
    def fmt(val):
        if not val:
            return "Not specified"
        return str(val).strip()

    def fmt_list(field):
        val = job.get(field, "")
        if isinstance(val, list):
            return ", ".join(v for v in val if v) or "Not specified"
        return fmt(val)

    epg_parts = list(filter(None, [
        fmt_list("Engines1"),
        fmt_list("Platforms1"),
        fmt_list("Genres1"),
    ]))

    prompt = HAIKU_PROMPT.format(
        job_description_tb       = fmt(job.get("Job_Description_TB")),
        requirements_tb          = fmt(job.get("Requirements_TB")),
        responsibilities_tb      = fmt(job.get("Responsibilities_TB")),
        required_skills          = fmt_list("SKILLS1"),
        required_specialities    = fmt(job.get("SPECIALITY1")),
        engines_platforms_genres = " | ".join(epg_parts) if epg_parts else "Not specified",
        cv_text                  = cv_text[:6000],
    )

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        data = json.loads(raw)
        return {
            "score":          int(data.get("score", 0)),
            "recommendation": data.get("recommendation", "Review"),
            "headline":       data.get("headline", ""),
            "strengths":      data.get("strengths", []),
            "explore":        data.get("explore", ""),
            "summary":        data.get("summary", ""),
        }
    except Exception as e:
        return {"score": 0, "error": str(e)}


def pre_filter(cand: dict, job: dict) -> tuple:
    """
    Apply rule-based pre-filter layers 2-4.
    Returns (pass: bool, reason: str).
    Layer 1 (category) is already applied at Zoho query time.
    """
    cand_cat = _candidate_category(cand).strip()
    job_cat  = _job_category(job).strip()
    same_category = cand_cat.lower() == job_cat.lower()

    # --- Layer 2: Seniority ---
    job_sen_raw  = (job.get("Work_Experience") or "").strip().lower()
    cand_sen_raw = (cand.get("Single_Line_1") or cand.get("Seniority_Level_2") or "").strip().lower()
    job_sen  = SENIORITY_NORMALISE.get(job_sen_raw, job_sen_raw)
    cand_sen = SENIORITY_NORMALISE.get(cand_sen_raw, cand_sen_raw)
    if job_sen and cand_sen:
        accepted = SENIORITY_ACCEPT.get(job_sen)
        if accepted and cand_sen not in accepted:
            return False, f"seniority mismatch ({cand_sen} vs job={job_sen})"

    # --- Layer 3: Location / Relocation ---
    job_prefs = _parse_work_prefs(job.get("Work_Preferences1"))
    if job_prefs and not ({"remote"} >= job_prefs):  # job is not remote-only
        job_country  = (job.get("Country") or "").strip().lower()
        cand_country = (cand.get("Country") or "").strip().lower()
        if isinstance(job.get("Country"), dict):
            job_country = job["Country"].get("name", "").strip().lower()
        if isinstance(cand.get("Country"), dict):
            cand_country = cand["Country"].get("name", "").strip().lower()
        if job_country and cand_country and job_country != cand_country:
            relocation = cand.get("Relocation")
            if relocation is False:
                return False, f"location mismatch ({cand_country} vs {job_country}) and not open to relocation"

    # --- Layer 4: Skills + Specialities ---
    job_skills  = _parse_skills(job.get("SKILLS1"))
    cand_skills = _parse_skills(cand.get("SKILLS"))

    if job_skills and cand_skills:
        if not job_skills & cand_skills:
            return False, f"no skill overlap (job={list(job_skills)[:3]}, cand={list(cand_skills)[:3]})"

    if same_category:
        job_specs  = _parse_skills(job.get("SPECIALITY1"))
        cand_specs = _parse_skills(cand.get("Specialities_2"))
        if job_specs and cand_specs:
            if not job_specs & cand_specs:
                return False, f"same category but no speciality overlap"

    return True, ""


def get_token():
    r = requests.post(ZOHO_ACCOUNTS, params={
        "refresh_token": REFRESH_TOKEN,
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type":    "refresh_token",
    })
    r.raise_for_status()
    return r.json()["access_token"]


def zoho_get(token, path, params=None):
    r = requests.get(f"{ZOHO_BASE}/{path}",
                     headers={"Authorization": f"Zoho-oauthtoken {token}"},
                     params=params or {}, timeout=30)
    return r


def find_job_by_ref(token, job_ref):
    r = zoho_get(token, "JobOpenings/search",
                 {"criteria": f"(Job_Opening_ID:equals:{job_ref})"})
    data = r.json().get("data", [])
    return data[0] if data else None


def get_candidates_for_job(token, job_id: str, target_status: str) -> list:
    """Use private API to get candidate IDs associated with a specific job opening and status."""
    PAGE_SIZE = 100
    PRIVATE_BASE = "https://recruit.zoho.eu/recruit/private/json/JobOpenings/getAssociatedCandidates"
    ids = []
    from_index = 1
    while True:
        params = {
            "id": job_id,
            "version": "2",
            "fromIndex": from_index,
            "toIndex": from_index + PAGE_SIZE - 1,
        }
        resp = requests.get(PRIVATE_BASE, headers={"Authorization": f"Bearer {token}"}, params=params, timeout=30)
        data = resp.json()

        if data.get("response", {}).get("nodata"):
            break

        rows = data.get("response", {}).get("result", {}).get("JobOpenings", {}).get("row", [])
        if not rows:
            break
        if not isinstance(rows, list):
            rows = [rows]

        for row in rows:
            fields = row.get("FL", [])
            if not isinstance(fields, list):
                fields = [fields]
            status = next((f["content"] for f in fields if f["val"] == "STATUS"), None)
            cand_id = next((f["content"] for f in fields if f["val"] == "CANDIDATEID"), None)
            if status == target_status and cand_id:
                ids.append(cand_id)

        if len(rows) < PAGE_SIZE:
            break
        from_index += PAGE_SIZE

    return ids


def get_event_candidates_for_categories(token, allowed_categories: set, status: str = "Applied-to-event"):
    """Fetch candidates by status filtered by category using Zoho criteria."""
    if not allowed_categories:
        return _fetch_event_candidates(token, criteria=f"(Candidate_Status:equals:{status})")

    all_cands = []
    for cat in allowed_categories:
        criteria = f"(Candidate_Status:equals:{status})AND(Pick_List_5:equals:{cat})"
        cands = _fetch_event_candidates(token, criteria=criteria)
        print(f"  '{cat}': {len(cands)} candidates")
        all_cands.extend(cands)

    # Deduplicate by id (a candidate could theoretically appear in multiple categories)
    seen = set()
    unique = []
    for c in all_cands:
        if c["id"] not in seen:
            seen.add(c["id"])
            unique.append(c)

    # Sort by Created_Time descending to preserve recency order
    unique.sort(key=lambda c: c.get("Created_Time", ""), reverse=True)
    return unique


def _fetch_event_candidates(token, criteria: str) -> list:
    """Fetch all pages for a given search criteria."""
    results = []
    page = 1
    while True:
        r = zoho_get(token, "Candidates/search", {
            "criteria":   criteria,
            "per_page":   200,
            "page":       page,
            "sort_by":    "Created_Time",
            "sort_order": "desc",
            "fields":     "id,First_Name,Last_Name,Created_Time,Pick_List_5",
        })
        if r.status_code == 204:
            break
        data = r.json().get("data", [])
        if not data:
            break
        results.extend(data)
        if not r.json().get("info", {}).get("more_records"):
            break
        page += 1
    return results


def get_candidate_details(token, cand_id):
    r = zoho_get(token, f"Candidates/{cand_id}")
    data = r.json().get("data", [])
    return data[0] if data else {}


def get_cv_text(token, cand_id):
    try:
        from pdfminer.high_level import extract_text as pdf_extract
    except ImportError:
        return ""

    r = zoho_get(token, f"Candidates/{cand_id}/Attachments")
    if r.status_code != 200:
        return ""

    resume = None
    for att in r.json().get("data", []):
        cat = att.get("Category", {})
        cat_name = cat.get("name", "") if isinstance(cat, dict) else cat
        if cat_name == "Resume":
            resume = att
            break

    if not resume:
        return ""

    att_id = resume.get("id")
    dl = requests.get(
        f"{ZOHO_BASE}/Candidates/{cand_id}/Attachments/{att_id}",
        headers={"Authorization": f"Zoho-oauthtoken {token}"},
        timeout=30,
    )
    if dl.status_code != 200:
        return ""

    try:
        text = pdf_extract(io.BytesIO(dl.content))
        return text.strip()[:8000]
    except Exception:
        return ""


def format_recruiter_text(match: dict) -> str:
    """Combine Haiku output into a single text block for the OutSystems tooltip."""
    parts = []
    if match.get("headline"):
        parts.append(match["headline"])
    if match.get("summary"):
        parts.append(match["summary"])
    strengths = match.get("strengths", [])
    if strengths:
        parts.append("Strengths:\n" + "\n".join(f"• {s}" for s in strengths))
    if match.get("explore"):
        parts.append(f"To explore: {match['explore']}")
    return "\n\n".join(parts)


def build_candidate_payload(cand, cv_text):
    def pick(obj):
        return obj.get("name", "") if isinstance(obj, dict) else (obj or "")

    def pick_list(field):
        val = cand.get(field)
        if not val:
            return []
        if isinstance(val, list):
            return [pick(v) for v in val if v]
        if isinstance(val, str):
            return [s.strip() for s in val.split(",") if s.strip()]
        return []

    return {
        "candidate_id":               cand.get("id", ""),
        "cv_text":                    cv_text,
        "current_job_title":          cand.get("Current_Job_Title") or cand.get("Job_Title_Primary_Skill"),
        "category":                   _candidate_category(cand),
        "seniority":                  cand.get("Single_Line_1") or pick(cand.get("Seniority_Level_2")),
        "experience_years":           cand.get("Experience_in_Years"),
        "years_in_gaming":            cand.get("Experience_in_Creative_Industry"),
        "location":                   f"{cand.get('City', '')} {pick(cand.get('Country', ''))}".strip(),
        "is_open_to_relocation":      cand.get("Relocation"),
        "expected_salary":            str(cand.get("Expected_Salary")) if cand.get("Expected_Salary") is not None else None,
        "linkedin_url":               cand.get("LinkedIn") or cand.get("LinkedIn__s"),
        "candidate_tags":             pick_list("Specialities_2") or pick_list("SPECIALITY1"),
        "skills":                     pick_list("SKILLS"),
        "platforms":                  pick_list("Platforms_2"),
        "engines":                    pick_list("Engines_2"),
        "genres":                     pick_list("Genres_2"),
        "art_styles":                 pick_list("Art_Styles"),
        "employment_types":           [],
        "work_preferences":           pick_list("Work_Preferences"),
        "game_titles":                cand.get("Game_Titles_or_Apps"),
        "tasks_and_responsibilities": cand.get("Tasks"),
        "motivation":                 cand.get("Motivation"),
        "expectations":               cand.get("Expectations_and_Dream_Job"),
        "dream_job":                  cand.get("What_are_the_challenges_and_lessons_learned"),
        "achievements":               cand.get("Achievements"),
    }


def build_job_payload(job):
    def pick(obj):
        return obj.get("name", "") if isinstance(obj, dict) else (obj or "")

    def pick_list(field):
        val = job.get(field, [])
        return [pick(v) for v in val if v] if isinstance(val, list) else []

    return {
        "job_id":          job.get("id", ""),
        "job_title":       job.get("Job_Opening_Name", ""),
        "job_description": " ".join(filter(None, [
            job.get("Job_Description"),
            job.get("Job_Description_TB"),
            job.get("Requirements_TB"),
            job.get("Responsibilities_TB"),
            job.get("Benefits_TB"),
        ])) or job.get("Job_Opening_Name", ""),
        "required_skills":   pick_list("SKILLS1"),
        "job_specialities":  [s.strip() for s in (job.get("SPECIALITY1") or "").split(",") if s.strip()],
        "job_seniority":     pick(job.get("Work_Experience")),
        "job_country":       pick(job.get("Country")),
        "job_city":          job.get("City", ""),
        "job_salary_min":    job.get("Salary_Range_Min") or job.get("Annual_Salary_From"),
        "job_salary_max":    job.get("Salary_Range_Max") or job.get("Annual_Salary_To"),
        "job_platforms":     pick_list("Platforms1"),
        "job_engines":       pick_list("Engines1"),
        "job_genres":        pick_list("Genres1"),
        "job_art_styles":    pick_list("Art_Styles"),
    }


def main():
    job_ref       = sys.argv[1] if len(sys.argv) > 1 else "ZR_640_JOB"
    cand_status   = sys.argv[2] if len(sys.argv) > 2 else "Applied-to-event"
    source_job_id = sys.argv[3] if len(sys.argv) > 3 else None  # e.g. 18387000010922002

    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print(f"\nFetching token...")
    token = get_token()

    print(f"Finding job {job_ref}...")
    job = find_job_by_ref(token, job_ref)
    if not job:
        print(f"Job {job_ref} not found")
        return

    # Fetch full job details
    r = zoho_get(token, f"JobOpenings/{job['id']}")
    job = r.json().get("data", [{}])[0]
    job_name = job.get("Job_Opening_Name", job_ref)
    country = job.get("Country", {})
    country_name = country.get("name", "") if isinstance(country, dict) else country
    job_category = _job_category(job)
    print(f"Job: {job_name} ({job.get('City', '')} {country_name})")
    print(f"Job category: {job_category or '(none)'}")

    job_payload = build_job_payload(job)

    # --- Fetch candidates ---
    if source_job_id:
        # Precise fetch: only candidates associated with a specific job opening
        print(f"\nFetching candidates from job {source_job_id} with status '{cand_status}' (private API)...")
        cand_ids = get_candidates_for_job(token, source_job_id, cand_status)
        print(f"Found {len(cand_ids)} candidates exactly matching job+status")
        # Build minimal ref dicts (details fetched later per candidate)
        cand_refs = [{"id": cid, "First_Name": "", "Last_Name": "", "Created_Time": "", "Pick_List_5": ""} for cid in cand_ids]
    elif job_category:
        allowed = _allowed_categories(job_category)
        print(f"\nFetching candidates in categories: {', '.join(sorted(allowed))}")
        cand_refs = get_event_candidates_for_categories(token, allowed, cand_status)
        print(f"Total to score: {len(cand_refs)} candidates\n")
    else:
        print(f"\nNo job category — fetching all candidates with status {cand_status}...")
        cand_refs = _fetch_event_candidates(token, f"(Candidate_Status:equals:{cand_status})")
        cand_refs = cand_refs[:EVENT_CAND_COUNT]
        print(f"Scoring first {len(cand_refs)} candidates\n")

    results = []
    for i, ref in enumerate(cand_refs, 1):
        cand_id = ref["id"]

        cand = get_candidate_details(token, cand_id)
        name = f"{cand.get('First_Name', '') or ref.get('First_Name', '')} {cand.get('Last_Name', '') or ref.get('Last_Name', '')}".strip() or cand_id
        created = (cand.get("Created_Time") or ref.get("Created_Time", ""))[:10]
        cand_cat = _candidate_category(cand) or _candidate_category(ref)
        print(f"[{i:3}/{len(cand_refs)}] {name} (registered {created}, category: {cand_cat or '?'})...")

        passed, reason = pre_filter(cand, job)
        if not passed:
            print(f"         Skipped: {reason}")
            continue

        cv_text = get_cv_text(token, cand_id)
        cv_status = f"{len(cv_text)} chars" if cv_text else "no CV"
        print(f"         CV: {cv_status}")

        if not cv_text:
            print(f"         Skipped: no CV (Stage 1 requires CV)")
            continue

        # Haiku: score CV against job, produce recruiter-ready output
        match = haiku_quick_score(anthropic_client, job, cv_text)
        if "error" in match:
            print(f"         Error: {match['error']}")
            continue

        score = match["score"]
        print(f"         Score: {score}/100 | {match['recommendation']} | {match['summary'][:70]}...")
        if score < 40:
            print(f"         Skipped: below threshold")
            continue

        result = {
            **match,
            "recruiter_text": format_recruiter_text(match),
            "candidate_id":   cand_id,
            "job_id":         job.get("id", ""),
            "_name":          name,
            "_created":       created,
            "_category":      cand_cat,
        }
        results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)

    print("\n" + "=" * 70)
    print(f"RESULTS: {len(results)} candidates vs {job_name}")
    print("=" * 70)

    for rec in ["Shortlist", "Review"]:
        group = [r for r in results if r.get("recommendation") == rec]
        if group:
            print(f"\n--- {rec} ({len(group)}) ---")
            for r in group:
                print(f"  {r['score']:>3}/100  {r['_name']}  [{r.get('_category', '?')}]")
                if r.get("headline"):
                    print(f"         {r['headline']}")
                if r.get("summary"):
                    print(f"         {r['summary'][:120]}")
                for s in r.get("strengths", [])[:2]:
                    print(f"         + {s}")
                if r.get("explore"):
                    print(f"         ? {r['explore']}")

    out = f"results_{job_ref}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out}")


if __name__ == "__main__":
    main()
