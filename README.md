# AI-Based Lead Scoring System

Explainable lead qualification: a **deterministic weighted rule engine** produces the score; an **optional LLM** only writes the narrative around it. Same input → same output, always.

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py          # web UI (single lead + CSV batch)
python cli.py sample_leads.json   # command line
pytest -q                     # 20 tests
export ANTHROPIC_API_KEY=...  # optional: enables the AI narrative (model via LEAD_LLM_MODEL)
```

## Architecture
| File | Role |
|---|---|
| `scoring.py` | Scoring engine: weights, rules, classification, confidence, next action (no LLM) |
| `llm_explainer.py` | LLM analysis/risks/outreach tip; temperature 0; falls back to rule-based text on any failure |
| `app.py` | Streamlit UI |
| `cli.py`, `sample_leads.json`, `test_scoring.py` | CLI, test cases, tests |

The LLM receives the finished score and factor breakdown and is told not to change them, so it cannot influence the number.

## Scoring methodology (total = 100 points)
Each parameter yields a credit ratio (0–1) × its weight.

| Parameter | Weight | How it is scored |
|---|---|---|
| Job title | 22 | Founder/CEO/Owner 100% · VP/Director 80% · Manager 55% · Engineer/Analyst/Exec 30% · Intern/Student 10% · unknown 35% |
| Requirement | 24 | Strong intent words (looking for, need, buy, demo…) +40% · product term (CRM, software…) +25% · urgency/budget/timeline +20% · detailed (≥6 words) +15%. Mild interest only: +20%. Low-intent phrases ("just exploring", "no budget") cap at 15% |
| Company size | 15 | 51–200 100% · 201–1000 90% · 11–50 75% · 1000+ 70% · 2–10 45% · solo 30% |
| Lead source | 10 | Referral 100% · Demo request 95% · Contact form 85% · Website 80% · Event/Webinar 65% · LinkedIn 60% · Email campaign 45% · Cold call 30% · Purchased list 15% |
| Industry | 8 | Target industry (IT services, SaaS, fintech, retail, …) 100%, other 55% |
| Location | 6 | India metro 100% · other Indian city 70% · elsewhere 55% |
| Email | 6 | Corporate matching website 100% · corporate 85% · free provider 55% · placeholder domain (example.com) 35% · invalid/missing 0% |
| Website | 5 | Valid 100% · malformed 30% · missing 0% |
| Phone | 4 | Valid Indian mobile 100% · plausible international 80% · invalid 20% · missing 0% |

**Classification:** Hot ≥ 70 · Warm 40–69 · Cold < 40.
**Reachability rule:** if neither email nor phone is usable, a lead is capped at 69 (cannot be Hot), and the explanation says so.

### Missing / incomplete data
- Contact fields (email, phone, website) earn 0 when missing; other fields earn a neutral 30% of their weight.
- Every missing field is listed as a negative factor and lowers **confidence**.
- Confidence = share of total weight backed by real data, minus a penalty for present-but-unreliable contact data. High ≥ 85%, Medium ≥ 65%, else Low.
- Key names are flexible (`Job Title`, `job_title`, `title`), and blanks / NaN / "N/A" are treated as missing. Empty input never crashes.

## Sample results
| Test case | Score | Class | Confidence |
|---|---|---|---|
| TC1 Assignment sample (Rahul Sharma) | 86 | Hot | High (97%) |
| TC2 Ideal lead (CEO, referral, urgent CRM) | 100 | Hot | High |
| TC3 Mid-level manager, vague need | 67 | Warm | High |
| TC4 Student from purchased list | 26 | Cold | High (91%) |
| TC5 Almost-empty record | 29 | Cold | Low (6%) |
| TC6 Strong director, invalid email/phone | 69 | Warm (capped) | High (92%) |

Note on TC1: `example.com` is flagged as a placeholder email, so the next action includes "verify the email address first".

## Tuning
All weights, thresholds, keyword lists and target industries are constants at the top of `scoring.py`. The assertion there guarantees weights always sum to 100.

## Demo script (2–3 min)
1. Open the app, click **Load sample lead**, press **Score lead**: show score, class, confidence, factors, breakdown table.
2. Clear Job Title/Email/Phone and rescore: score and confidence drop, negatives appear.
3. Change requirement to "just exploring": show the score fall.
4. Upload a CSV in the **Batch** tab and download the ranked results.
5. Toggle **AI narrative** (with API key) and show the score is unchanged.
