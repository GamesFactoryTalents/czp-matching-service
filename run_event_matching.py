"""
Fetch the 123 most recent Applied-to-event candidates (last Talent Fair),
get their full profiles + CVs, and match against a target job.

Usage: python3 run_event_matching.py ZR_640_JOB
"""
import sys
import os
import json
import time
import io
import requests
from dotenv import load_dotenv

load_dotenv()

ZOHO_BASE     = "https://recruit.zoho.eu/recruit/v2"
ZOHO_ACCOUNTS = "https://accounts.zoho.eu/oauth/v2/token"
MATCHING_URL  = "https://czp-matching-service-production.up.railway.app"
MATCHING_KEY  = "careers-zone-matching"
EVENT_CAND_COUNT = 123   # candidates in last event

CLIENT_ID     = os.environ["ZOHO_CLIENT_ID"]
CLIENT_SECRET = os.environ["ZOHO_CLIENT_SECRET"]
REFRESH_TOKEN = os.environ["ZOHO_REFRESH_TOKEN"]


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


def get_last_event_candidates(token, count=123):
    """Get the `count` most recently created Applied-to-event candidates."""
    all_ids = []
    page = 1
    while len(all_ids) < count:
        r = zoho_get(token, "Candidates/search", {
            "criteria":   "(Candidate_Status:equals:Applied-to-event)",
            "per_page":   200,
            "page":       page,
            "sort_by":    "Created_Time",
            "sort_order": "desc",
            "fields":     "id,First_Name,Last_Name,Created_Time",
        })
        if r.status_code == 204:
            break
        data = r.json().get("data", [])
        all_ids.extend(data)
        if not r.json().get("info", {}).get("more_records"):
            break
        page += 1
    return all_ids[:count]


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
        "category":                   pick(cand.get("Category") or cand.get("CATEGORY1")),
        "seniority":                  pick(cand.get("Seniority_Level_2")),
        "experience_years":           cand.get("Experience_in_Years"),
        "years_in_gaming":            cand.get("Experience_in_Creative_Industry"),
        "location":                   f"{cand.get('City', '')} {pick(cand.get('Country', ''))}".strip(),
        "is_open_to_relocation":      cand.get("Relocation"),
        "expected_salary":            str(cand.get("Expected_Salary")) if cand.get("Expected_Salary") is not None else None,
        "linkedin_url":               cand.get("LinkedIn") or cand.get("LinkedIn__s"),
        "tags":                       pick_list("Specialities_2") or pick_list("SPECIALITY1"),
        "skills":                     pick_list("Skill_Set") or pick_list("SKILLS"),
        "platforms":                  pick_list("Platforms_2"),
        "engines":                    pick_list("Engines_2"),
        "genres":                     pick_list("Genres_2"),
        "art_styles":                 pick_list("Art_Styles"),
        "employment_types":           [],
        "work_preferences":           pick_list("Work_Preferences"),
        "game_titles":                cand.get("Game_Titles_or_Apps"),
        "tasks_and_responsibilities": cand.get("What_are_you_roles_and_responsibilities"),
        "motivation":                 cand.get("Motivation"),
        "expectations":               cand.get("Expectations_and_Dream_Job"),
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
        "required_skills": pick_list("SKILLS1") or pick_list("Required_Skills"),
        "job_seniority":   pick(job.get("Work_Experience")),
        "job_country":     pick(job.get("Country")),
        "job_city":        job.get("City", ""),
        "job_salary_min":  job.get("Salary_Range_Min") or job.get("Annual_Salary_From"),
        "job_salary_max":  job.get("Salary_Range_Max") or job.get("Annual_Salary_To"),
        "job_platforms":   pick_list("Platforms1"),
        "job_engines":     pick_list("Engines1"),
        "job_genres":      pick_list("Genres1"),
        "job_art_styles":  pick_list("Art_Styles"),
    }


def main():
    job_ref = sys.argv[1] if len(sys.argv) > 1 else "ZR_640_JOB"

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
    print(f"Job: {job_name} ({job.get('City', '')} {job.get('Country', {}).get('name', '') if isinstance(job.get('Country'), dict) else job.get('Country', '')})")

    job_payload = build_job_payload(job)

    print(f"\nFetching {EVENT_CAND_COUNT} most recent event candidates...")
    cand_refs = get_last_event_candidates(token, EVENT_CAND_COUNT)
    print(f"Got {len(cand_refs)} candidates\n")

    results = []
    for i, ref in enumerate(cand_refs, 1):
        cand_id = ref["id"]
        name = f"{ref.get('First_Name', '')} {ref.get('Last_Name', '')}".strip()
        created = ref.get("Created_Time", "")[:10]
        print(f"[{i:3}/{len(cand_refs)}] {name} (registered {created})...")

        cand = get_candidate_details(token, cand_id)
        cv_text = get_cv_text(token, cand_id)
        cv_status = f"{len(cv_text)} chars" if cv_text else "no CV"
        print(f"         CV: {cv_status}")

        if not cv_text:
            # Build a text summary from profile fields so the model has something to work with
            parts = [
                cand.get("Job_Title_Primary_Skill") or "",
                cand.get("Skill_Set") or "",
                cand.get("What_are_you_roles_and_responsibilities") or "",
                cand.get("Motivation") or "",
                cand.get("Expectations_and_Dream_Job") or "",
                cand.get("Game_Titles_or_Apps") or "",
            ]
            cv_text = " | ".join(p for p in parts if p).strip()

        cand_payload = build_candidate_payload(cand, cv_text)
        payload = {
            **job_payload,
            **cand_payload,
            "job_description": job_payload["job_description"],
        }

        try:
            r = requests.post(
                f"{MATCHING_URL}/match",
                json=payload,
                headers={
                    "Authorization": f"Bearer {MATCHING_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=90,
            )
            r.raise_for_status()
            result = r.json()
            result["_name"] = name
            result["_created"] = created
            results.append(result)
            print(f"         Score: {result['score']}/100 | {result['recommendation']} | {result['summary'][:70]}...")
        except Exception as e:
            print(f"         Error: {e}")

        time.sleep(0.3)

    results.sort(key=lambda x: x["score"], reverse=True)

    print("\n" + "=" * 70)
    print(f"RESULTS: {len(results)} candidates vs {job_name}")
    print("=" * 70)

    for rec in ["Shortlist", "Review", "Pass"]:
        group = [r for r in results if r.get("recommendation") == rec]
        if group:
            print(f"\n--- {rec} ({len(group)}) ---")
            for r in group:
                print(f"  {r['score']:>3}/100  {r['_name']}")
                print(f"         {r['summary'][:90]}")
                if r.get("strengths"):
                    print(f"         + {r['strengths'][0]}")
                if r.get("gaps"):
                    print(f"         - {r['gaps'][0]}")

    out = f"results_{job_ref}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out}")


if __name__ == "__main__":
    main()
