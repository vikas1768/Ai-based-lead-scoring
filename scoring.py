"""Deterministic, explainable lead scoring engine.

No LLM is involved here. Identical input -> identical output, always.
Weights sum to 100, so the final score is simply the sum of points earned.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any

# --------------------------------------------------------------------------
# Configuration (edit these to tune the model for a different product / ICP)
# --------------------------------------------------------------------------
WEIGHTS = {
    "job_title": 22,      # decision-making authority
    "requirement": 24,    # buying intent + clarity of need
    "company_size": 15,   # fit with ideal customer size
    "lead_source": 10,    # historical conversion quality of channel
    "industry": 8,        # fit with target industries
    "location": 6,        # fit with serviceable geography
    "email": 6,           # reachability / credibility of email
    "website": 5,         # company legitimacy
    "phone": 4,           # reachability
}
assert sum(WEIGHTS.values()) == 100

HOT_THRESHOLD = 70
WARM_THRESHOLD = 40

# Share of a field's weight awarded when the data is missing (neutral-ish,
# but always flagged as a negative and always lowers confidence).
MISSING_CREDIT = {"email": 0.0, "phone": 0.0, "website": 0.0}
DEFAULT_MISSING_CREDIT = 0.30

TARGET_INDUSTRIES = {
    "it services", "software", "saas", "fintech", "bfsi", "e-commerce", "ecommerce",
    "real estate", "manufacturing", "healthcare", "education", "consulting",
    "digital marketing", "logistics", "telecom", "retail",
}
INDIA_METROS = {
    "bangalore", "bengaluru", "mumbai", "delhi", "new delhi", "gurgaon", "gurugram",
    "noida", "hyderabad", "chennai", "pune", "kolkata",
}
OTHER_INDIA = {
    "ahmedabad", "jaipur", "kochi", "indore", "chandigarh", "coimbatore", "lucknow",
    "nagpur", "surat", "vadodara", "bhubaneswar", "india",
}
FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "yahoo.in", "hotmail.com", "outlook.com", "rediffmail.com",
    "live.com", "icloud.com", "aol.com", "proton.me", "protonmail.com",
}
PLACEHOLDER_EMAIL_DOMAINS = {"example.com", "example.org", "test.com", "domain.com", "email.com", "mailinator.com"}

SOURCE_SCORES = {
    "referral": 1.00, "demo request": 0.95, "contact form": 0.85, "website": 0.80,
    "partner": 0.75, "event": 0.65, "webinar": 0.65, "linkedin": 0.60,
    "social media": 0.50, "email campaign": 0.45, "advertisement": 0.45, "ads": 0.45,
    "cold call": 0.30, "cold email": 0.30, "purchased list": 0.15, "list": 0.15,
}

TITLE_TIERS = [  # (credit, patterns) — first match wins, checked top-down
    (0.10, ["intern", "student", "trainee", "freelancer"]),
    (1.00, ["founder", "co-founder", "cofounder", "owner", "ceo", "chief", "cto", "coo", "cfo", "cmo",
            "managing director", "president", "proprietor", "partner", "chairman"]),
    (0.80, ["vp", "vice president", "director", "head of", "head -", "general manager"]),
    (0.55, ["manager", "lead", "supervisor", "team lead"]),
    (0.30, ["engineer", "developer", "analyst", "executive", "associate", "assistant", "coordinator",
            "consultant", "specialist", "officer"]),
]
UNKNOWN_TITLE_CREDIT = 0.35

INTENT_STRONG = ["looking for", "need", "require", "want to buy", "want to purchase",
                 "evaluating", "evaluate", "implement", "migrate", "replace", "upgrade", "demo",
                 "pricing", "quote", "quotation", "proposal", "rfp", "purchase", "buy"]
INTENT_WEAK = ["interested in", "interested", "exploring", "curious", "information", "details", "enquiry", "inquiry"]
PRODUCT_TERMS = ["crm", "software", "platform", "erp", "automation", "tool", "solution", "system",
                 "app", "saas", "integration", "dashboard", "pipeline"]
URGENCY_BUDGET = ["urgent", "asap", "immediately", "this month", "this quarter", "next month",
                  "budget", "approved", "deadline", "go live", "timeline", "within"]
LOW_INTENT = ["just exploring", "just browsing", "just curious", "no plan", "no budget", "free",
              "student project", "assignment", "research only", "job", "internship", "not sure", "maybe later"]


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------
@dataclass
class Factor:
    parameter: str
    weight: int
    points: float
    ratio: float
    input_value: str
    reason: str
    missing: bool = False

    @property
    def sentiment(self) -> str:
        if self.missing:
            return "negative"
        if self.ratio >= 0.7:
            return "positive"
        if self.ratio < 0.5:
            return "negative"
        return "neutral"


@dataclass
class ScoreResult:
    lead_name: str
    score: int
    classification: str
    quality: str
    confidence: str
    confidence_pct: int
    positives: list[str] = field(default_factory=list)
    negatives: list[str] = field(default_factory=list)
    explanation: str = ""
    next_action: str = ""
    factors: list[Factor] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for f, obj in zip(d["factors"], self.factors):
            f["sentiment"] = obj.sentiment
        return d


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _clean(v: Any) -> str:
    return str(v).strip() if v is not None and str(v).strip().lower() not in {"", "nan", "none", "null", "n/a", "-"} else ""


def _has(text: str, terms: list[str]) -> list[str]:
    return [t for t in terms if re.search(r"(?<![a-z])" + re.escape(t) + r"(?![a-z])", text)]


def _domain_of(url_or_email: str) -> str:
    s = url_or_email.lower().strip()
    s = s.split("@")[-1]
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^www\.", "", s)
    return s.split("/")[0]


# --------------------------------------------------------------------------
# Individual parameter scorers: each returns (ratio 0..1, reason, missing)
# --------------------------------------------------------------------------
def score_job_title(v: str):
    if not v:
        return DEFAULT_MISSING_CREDIT, "Job title not provided; decision-making authority unknown", True
    t = v.lower()
    for credit, pats in TITLE_TIERS:
        hit = _has(t, pats)
        if hit:
            labels = {1.0: "top decision-maker", 0.8: "senior leader with strong influence",
                      0.55: "mid-level manager with some influence", 0.3: "individual contributor, limited buying power",
                      0.1: "intern/student, unlikely to buy"}
            return credit, f"'{v}' is a {labels[credit]}", False
    return UNKNOWN_TITLE_CREDIT, f"'{v}' is an unrecognised title; authority unclear", False


def score_company_size(v: str):
    if not v:
        return DEFAULT_MISSING_CREDIT, "Company size not provided", True
    nums = [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", v)]
    if not nums:
        return DEFAULT_MISSING_CREDIT, f"Could not read a number from company size '{v}'", True
    n = max(nums)  # upper bound of range
    if "+" in v and len(nums) == 1:
        n = nums[0]
    if n <= 1:
        return 0.30, "Solo operation; very small budget", False
    if n <= 10:
        return 0.45, f"Micro company (~{n} employees); limited budget", False
    if n <= 50:
        return 0.75, f"Small company (~{n} employees); good fit, quick decisions", False
    if n <= 200:
        return 1.00, f"Mid-size company (~{n} employees); ideal size for the product", False
    if n <= 1000:
        return 0.90, f"Large company (~{n} employees); strong budget, longer sales cycle", False
    return 0.70, f"Enterprise (~{n}+ employees); high value but complex, long procurement", False


def score_requirement(v: str):
    if not v:
        return DEFAULT_MISSING_CREDIT, "No requirement stated; buying intent unknown", True
    t = v.lower()
    low = _has(t, LOW_INTENT)
    ratio, notes = 0.0, []
    if _has(t, INTENT_STRONG):
        ratio += 0.40
        notes.append("explicit buying intent")
    elif _has(t, INTENT_WEAK):
        ratio += 0.20
        notes.append("mild interest only, no firm buying intent")
    product = _has(t, PRODUCT_TERMS)
    if product:
        ratio += 0.25
        notes.append("names a product category")
    if _has(t, URGENCY_BUDGET):
        ratio += 0.20
        notes.append("mentions urgency/budget/timeline")
    if len(t.split()) >= 6:
        ratio += 0.15
        notes.append("specific, detailed requirement")
    if low:
        ratio = min(ratio, 0.15)
        notes.append("low-intent wording (" + ", ".join(low) + ")")
    ratio = min(ratio, 1.0)
    if not notes:
        ratio, notes = 0.15, ["no clear buying signal"]
    return ratio, f"'{v}': " + ", ".join(notes), False


def score_source(v: str):
    if not v:
        return DEFAULT_MISSING_CREDIT, "Lead source not provided", True
    t = v.lower()
    for key in sorted(SOURCE_SCORES, key=len, reverse=True):
        if key in t:
            r = SOURCE_SCORES[key]
            q = "high-converting" if r >= 0.8 else "average" if r >= 0.5 else "low-converting"
            return r, f"'{v}' is a {q} channel", False
    return 0.40, f"'{v}' is an unrecognised source; assumed below average", False


def score_industry(v: str):
    if not v:
        return DEFAULT_MISSING_CREDIT, "Industry not provided", True
    t = v.lower().strip()
    if any(ind in t or t in ind for ind in TARGET_INDUSTRIES):
        return 1.0, f"'{v}' is a target industry", False
    return 0.55, f"'{v}' is outside the core target industries", False


def score_location(v: str):
    if not v:
        return DEFAULT_MISSING_CREDIT, "Location not provided", True
    t = v.lower()
    if any(c in t for c in INDIA_METROS):
        return 1.0, f"'{v}' is a primary metro market", False
    if any(c in t for c in OTHER_INDIA):
        return 0.70, f"'{v}' is a serviceable Indian market (non-metro)", False
    return 0.55, f"'{v}' is outside the primary India market", False


def score_email(v: str, website: str):
    if not v:
        return 0.0, "No email provided; cannot reach the lead", True
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v):
        return 0.0, f"Email '{v}' is not a valid format", False
    dom = _domain_of(v)
    if dom in PLACEHOLDER_EMAIL_DOMAINS:
        return 0.35, f"Email domain '{dom}' looks like a placeholder; verify before outreach", False
    if dom in FREE_EMAIL_DOMAINS:
        return 0.55, "Free email provider; less indicative of a real business buyer", False
    if website and _domain_of(website) == dom:
        return 1.0, "Corporate email matching the company website", False
    return 0.85, f"Corporate email domain '{dom}'", False


def score_website(v: str):
    if not v:
        return 0.0, "No website provided; company legitimacy unverified", True
    if re.fullmatch(r"(https?://)?([a-z0-9-]+\.)+[a-z]{2,}(/.*)?", v.lower().strip()):
        return 1.0, "Valid company website provided", False
    return 0.3, f"'{v}' does not look like a valid website", False


def score_phone(v: str):
    if not v:
        return 0.0, "No phone number provided", True
    digits = re.sub(r"\D", "", v)
    if len(digits) == 10 and digits[0] in "6789":
        return 1.0, "Valid Indian mobile number", False
    if len(digits) == 12 and digits.startswith("91") and digits[2] in "6789":
        return 1.0, "Valid Indian mobile number (+91)", False
    if 10 <= len(digits) <= 15:
        return 0.8, "Plausible international phone number", False
    return 0.2, f"Phone '{v}' has an invalid length", False


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
FIELD_ALIASES = {
    "name": "name", "company": "company", "company_name": "company", "website": "website",
    "job_title": "job_title", "title": "job_title", "industry": "industry",
    "location": "location", "company_location": "location", "company_size": "company_size",
    "size": "company_size", "lead_source": "lead_source", "source": "lead_source",
    "requirement": "requirement", "email": "email", "phone": "phone",
}


def normalise(lead: dict[str, Any]) -> dict[str, str]:
    out = {k: "" for k in set(FIELD_ALIASES.values())}
    for k, v in lead.items():
        key = FIELD_ALIASES.get(re.sub(r"[\s\-]+", "_", str(k).strip().lower()))
        if key:
            out[key] = _clean(v)
    return out


def classify(score: int) -> str:
    return "Hot" if score >= HOT_THRESHOLD else "Warm" if score >= WARM_THRESHOLD else "Cold"


def _quality(cls: str, score: int, lead: dict[str, str]) -> str:
    base = {
        "Hot": "High-quality lead: strong fit and clear buying signals; prioritise immediately.",
        "Warm": "Moderate-quality lead: some fit and interest, but needs qualification before committing heavy effort.",
        "Cold": "Low-quality lead: weak fit or little buying intent; keep in long-term nurture.",
    }[cls]
    return f"{base} (Score {score}/100)"


def _confidence(factors: list[Factor], lead: dict[str, str]) -> tuple[str, int]:
    total = sum(f.weight for f in factors)
    present = sum(f.weight for f in factors if not f.missing)
    pct = present / total * 100
    # Penalise data that exists but looks unreliable
    for f in factors:
        if not f.missing and f.parameter in ("email", "website", "phone") and f.ratio < 0.5:
            pct -= f.weight * 0.5
    # Penalise a score sitting right on a class boundary
    pct = max(0, min(100, round(pct)))
    label = "High" if pct >= 85 else "Medium" if pct >= 65 else "Low"
    return label, pct


def _next_action(cls: str, lead: dict[str, str], factors: dict[str, Factor]) -> str:
    has_phone = not factors["phone"].missing and factors["phone"].ratio >= 0.5
    has_email = not factors["email"].missing and factors["email"].ratio >= 0.5
    if cls == "Hot":
        act = ("Call within 24 hours and book a personalised product demo." if has_phone
               else "Send a personalised email within 24 hours proposing a demo call.")
    elif cls == "Warm":
        act = ("Send a tailored intro email with a case study, then follow up with a discovery call within 3 days."
               if has_email else "Attempt a discovery call within 3 days to qualify needs and budget.")
    else:
        act = "Add to a low-touch email nurture sequence and re-score if new information or engagement arrives."
    notes = []
    if factors["email"].ratio < 0.5 and not factors["email"].missing:
        notes.append("verify the email address first")
    if factors["job_title"].ratio < 0.5:
        notes.append("identify the actual decision-maker at the company")
    if factors["requirement"].missing or factors["requirement"].ratio < 0.4:
        notes.append("clarify the requirement and timeline")
    if notes:
        act += " Also: " + "; ".join(notes) + "."
    return act


def _explain(score: int, cls: str, factors: list[Factor]) -> str:
    ranked = sorted(factors, key=lambda f: f.points, reverse=True)
    top = ", ".join(f"{f.parameter.replace('_', ' ')} ({f.points:.1f}/{f.weight})" for f in ranked[:3])
    gaps = sorted(factors, key=lambda f: f.weight - f.points, reverse=True)[:2]
    lost = ", ".join(f"{f.parameter.replace('_', ' ')} (-{f.weight - f.points:.1f})" for f in gaps if f.weight - f.points > 0.5)
    msg = (f"The lead scored {score}/100 and is classified {cls} "
           f"(Hot ≥ {HOT_THRESHOLD}, Warm ≥ {WARM_THRESHOLD}). "
           f"Biggest contributors: {top}.")
    if lost:
        msg += f" Largest point losses: {lost}."
    return msg


def score_lead(raw_lead: dict[str, Any]) -> ScoreResult:
    lead = normalise(raw_lead)
    spec = [
        ("job_title", "Job Title", lead["job_title"], score_job_title(lead["job_title"])),
        ("requirement", "Requirement", lead["requirement"], score_requirement(lead["requirement"])),
        ("company_size", "Company Size", lead["company_size"], score_company_size(lead["company_size"])),
        ("lead_source", "Lead Source", lead["lead_source"], score_source(lead["lead_source"])),
        ("industry", "Industry", lead["industry"], score_industry(lead["industry"])),
        ("location", "Location", lead["location"], score_location(lead["location"])),
        ("email", "Email", lead["email"], score_email(lead["email"], lead["website"])),
        ("website", "Website", lead["website"], score_website(lead["website"])),
        ("phone", "Phone", lead["phone"], score_phone(lead["phone"])),
    ]
    factors: list[Factor] = []
    for key, _label, value, (ratio, reason, missing) in spec:
        if missing and key in MISSING_CREDIT:
            ratio = MISSING_CREDIT[key]
        w = WEIGHTS[key]
        factors.append(Factor(key, w, round(w * ratio, 2), round(ratio, 3), value or "—", reason, missing))

    score = int(round(sum(f.points for f in factors)))
    score = max(0, min(100, score))
    fmap = {f.parameter: f for f in factors}
    capped = False
    if fmap["email"].ratio < 0.5 and fmap["phone"].ratio < 0.5 and score >= HOT_THRESHOLD:
        score, capped = HOT_THRESHOLD - 1, True  # cannot be Hot if the lead is unreachable
    cls = classify(score)
    conf_label, conf_pct = _confidence(factors, lead)

    positives = [f"{f.parameter.replace('_', ' ').title()} (+{f.points:.1f}/{f.weight}): {f.reason}"
                 for f in factors if f.sentiment == "positive"]
    negatives = [f"{f.parameter.replace('_', ' ').title()} ({f.points:.1f}/{f.weight}): {f.reason}"
                 for f in factors if f.sentiment == "negative"]

    return ScoreResult(
        lead_name=lead["name"] or "Unnamed lead",
        score=score, classification=cls, quality=_quality(cls, score, lead),
        confidence=conf_label, confidence_pct=conf_pct,
        positives=positives, negatives=negatives,
        explanation=_explain(score, cls, factors) + (
            f" Score capped at {HOT_THRESHOLD - 1} because neither email nor phone is usable, "
            "so the lead cannot be contacted yet." if capped else ""),
        next_action=_next_action(cls, lead, fmap),
        factors=factors,
    )
