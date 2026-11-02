"""Streamlit UI for the AI-Based Lead Scoring System.   Run: streamlit run app.py"""
import html
import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st

from llm_explainer import llm_available, llm_insights
from scoring import HOT_THRESHOLD, WARM_THRESHOLD, WEIGHTS, score_lead

st.set_page_config(page_title="Lead Scoring AI", page_icon="🎯", layout="wide")

COLORS = {"Hot": "#ef4444", "Warm": "#f59e0b", "Cold": "#3b82f6"}
EMOJI = {"Hot": "🔥", "Warm": "🌤️", "Cold": "❄️"}
FIELDS = [
    ("name", "Name"), ("company", "Company name"), ("website", "Website"), ("job_title", "Job title"),
    ("industry", "Industry"), ("location", "Company location"), ("company_size", "Company size"),
    ("lead_source", "Lead source"), ("requirement", "Requirement"), ("email", "Email"), ("phone", "Phone"),
]
SAMPLE = {
    "name": "Rahul Sharma", "company": "ABC Technologies", "website": "www.abctechnologies.com",
    "job_title": "Founder", "industry": "IT Services", "location": "Bangalore",
    "company_size": "50–100 Employees", "lead_source": "Website",
    "requirement": "Looking for CRM software", "email": "rahul@example.com", "phone": "+91 9876543210",
}
PRESET_LABELS = {
    "TC1_assignment_sample": "Assignment sample: Rahul Sharma",
    "TC2_ideal_lead": "Ideal lead: CEO, referral, urgent need",
    "TC3_mid_manager": "Mid-level manager, vague need",
    "TC4_cold_student": "Student from a purchased list",
    "TC5_missing_data": "Almost-empty record",
    "TC6_invalid_contact": "Strong lead, invalid contact details",
}
FIXES = {
    "job_title": "Find and reach the real decision-maker (founder, director or head of department)",
    "requirement": "Clarify the exact need, timeline and budget on a discovery call",
    "company_size": "Confirm the company size",
    "email": "Verify the email or collect a corporate address",
    "website": "Add a valid company website",
    "phone": "Collect a valid mobile number",
}
esc = lambda t: html.escape(str(t))
LABEL = lambda p: p.replace("_", " ").capitalize()


def load_presets():
    try:
        data = json.loads((Path(__file__).parent / "sample_leads.json").read_text(encoding="utf-8"))
        return {PRESET_LABELS.get(c["id"], c["id"]): c["input"] for c in data}
    except Exception:
        return {"Assignment sample: Rahul Sharma": SAMPLE}


PRESETS = load_presets()

st.markdown("""<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.6rem; max-width: 1250px;}
.hero {background: linear-gradient(135deg,#4f46e5 0%,#7c3aed 55%,#ec4899 100%); color:#fff; border-radius:18px; padding:26px 30px; margin-bottom:18px;}
.hero h1 {margin:0; font-size:30px; font-weight:800; color:#fff; padding:0;}
.hero p {margin:6px 0 12px; opacity:.92; font-size:15px;}
.chip {display:inline-block; background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.3); border-radius:999px; padding:3px 12px; font-size:12px; margin-right:6px;}
.card {background:rgba(128,128,128,.08); border:1px solid rgba(128,128,128,.22); border-radius:14px; padding:16px 18px; margin-bottom:14px;}
.card h4 {margin:0 0 8px; font-size:15px;}
.kpi {text-align:center; padding:14px 8px;}
.kpi .l {font-size:12px; opacity:.65; text-transform:uppercase; letter-spacing:.06em;}
.kpi .v {font-size:26px; font-weight:800; line-height:1.25;}
.pill {display:inline-block; color:#fff; font-weight:700; border-radius:999px; padding:5px 18px; font-size:18px;}
.callout {border-radius:12px; padding:12px 16px; margin-bottom:12px; font-size:14.5px;}
.row {display:grid; grid-template-columns:120px 1fr 62px; gap:10px; align-items:center; padding:7px 0; border-bottom:1px solid rgba(128,128,128,.15);}
.row:last-child {border-bottom:none;}
.track {height:9px; background:rgba(128,128,128,.22); border-radius:5px; overflow:hidden;}
.fill {height:100%; border-radius:5px;}
.why {grid-column:2 / 4; font-size:12.5px; opacity:.72; margin-top:-4px;}
.tag {font-size:13.5px; padding:5px 0;}
.bar {display:flex; align-items:center; gap:10px; margin:6px 0; font-size:14px;}
.bar .b {height:22px; border-radius:6px; min-width:4px;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def gauge(score: int, cls: str) -> str:
    c, L = COLORS[cls], math.pi * 80
    ticks = ""
    for p in (WARM_THRESHOLD, HOT_THRESHOLD):
        th = p / 100 * math.pi
        x1, y1 = 100 - 70 * math.cos(th), 100 - 70 * math.sin(th)
        x2, y2 = 100 - 92 * math.cos(th), 100 - 92 * math.sin(th)
        ticks += f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="currentColor" stroke-opacity=".45" stroke-width="1.5"/>'
    arc = "M20 100 A80 80 0 0 1 180 100"
    return (f'<svg viewBox="0 0 200 122" width="100%" style="max-width:340px;display:block;margin:0 auto">'
            f'<path d="{arc}" fill="none" stroke="rgba(128,128,128,.25)" stroke-width="16" stroke-linecap="round"/>'
            f'<path d="{arc}" fill="none" stroke="{c}" stroke-width="16" stroke-linecap="round" stroke-dasharray="{L * score / 100:.1f} {L:.1f}"/>'
            f'{ticks}<text x="100" y="94" text-anchor="middle" font-size="42" font-weight="800" fill="{c}">{score}</text>'
            f'<text x="100" y="113" text-anchor="middle" font-size="10" fill="currentColor" opacity=".6">LEAD SCORE OUT OF 100</text></svg>')


def bar_color(ratio: float, missing: bool) -> str:
    if missing:
        return "#9ca3af"
    return "#22c55e" if ratio >= 0.7 else "#f59e0b" if ratio >= 0.5 else "#ef4444"


def breakdown_html(result) -> str:
    rows = ""
    for f in sorted(result.factors, key=lambda x: -x.weight):
        pct = f.points / f.weight * 100
        rows += (f'<div class="row"><div><b>{LABEL(f.parameter)}</b></div>'
                 f'<div class="track"><div class="fill" style="width:{pct:.0f}%;background:{bar_color(f.ratio, f.missing)}"></div></div>'
                 f'<div style="text-align:right"><b>{f.points:.1f}</b><span style="opacity:.55">/{f.weight}</span></div>'
                 f'<div class="why">{esc(f.reason)}</div></div>')
    return f'<div class="card"><h4>📊 Score breakdown</h4>{rows}</div>'


def render_result(result, lead, use_llm):
    c = COLORS[result.classification]
    g, k = st.columns([1.15, 1])
    with g:
        st.markdown(f'<div class="card">{gauge(result.score, result.classification)}'
                    f'<div style="text-align:center;margin-top:4px"><span class="pill" style="background:{c}">'
                    f'{EMOJI[result.classification]} {result.classification.upper()} LEAD</span></div></div>',
                    unsafe_allow_html=True)
    with k:
        conf_c = {"High": "#22c55e", "Medium": "#f59e0b", "Low": "#ef4444"}[result.confidence]
        st.markdown(
            f'<div class="card kpi"><div class="l">Confidence</div><div class="v" style="color:{conf_c}">{result.confidence} · {result.confidence_pct}%</div>'
            f'<div class="track" style="margin-top:6px"><div class="fill" style="width:{result.confidence_pct}%;background:{conf_c}"></div></div></div>'
            f'<div class="card kpi"><div class="l">Quality assessment</div><div style="font-size:14px;margin-top:4px">{esc(result.quality)}</div></div>',
            unsafe_allow_html=True)

    st.markdown(f'<div class="callout" style="background:rgba(34,197,94,.14);border:1px solid rgba(34,197,94,.45)">'
                f'🎯 <b>Recommended next action:</b> {esc(result.next_action)}</div>', unsafe_allow_html=True)

    p, n = st.columns(2)
    pos = "".join(f'<div class="tag">✅ {esc(x)}</div>' for x in result.positives) or "<div class='tag'>None</div>"
    neg = "".join(f'<div class="tag">⚠️ {esc(x)}</div>' for x in result.negatives) or "<div class='tag'>None</div>"
    p.markdown(f'<div class="card"><h4>Positive factors</h4>{pos}</div>', unsafe_allow_html=True)
    n.markdown(f'<div class="card"><h4>Negative factors</h4>{neg}</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="card"><h4>🧾 Scoring explanation</h4>{esc(result.explanation)}</div>', unsafe_allow_html=True)

    if use_llm:
        ins = llm_insights(result, lead)
        st.markdown(f'<div class="card" style="border-color:#7c3aed"><h4>🤖 AI analysis <span style="font-weight:400;opacity:.6;font-size:12px">({esc(ins["source"])})</span></h4>'
                    f'<p><b>Analysis:</b> {esc(ins["analysis"])}</p><p><b>Risks:</b> {esc(ins["risks"])}</p>'
                    f'<p style="margin:0"><b>Outreach tip:</b> {esc(ins["outreach_tip"])}</p></div>', unsafe_allow_html=True)

    st.markdown(breakdown_html(result), unsafe_allow_html=True)

    gaps = [f for f in result.factors if f.weight - f.points >= 0.5 and f.parameter in FIXES]
    gaps.sort(key=lambda f: f.points - f.weight)
    if gaps:
        items = "".join(f'<div class="tag">⬆️ <b>+{f.weight - f.points:.1f} pts</b> · {FIXES[f.parameter]}</div>' for f in gaps[:4])
        st.markdown(f'<div class="card"><h4>🚀 Improvement potential</h4>'
                    f'<div style="opacity:.7;font-size:13px;margin-bottom:4px">Points this lead could still gain if the data improves:</div>{items}</div>',
                    unsafe_allow_html=True)

    report = {"lead": lead, "result": result.to_dict()}
    st.download_button("⬇️ Download JSON report", json.dumps(report, indent=2, ensure_ascii=False),
                       f"lead_score_{result.lead_name.replace(' ', '_')}.json", "application/json")


# ------------------------------------------------------------------ state
if "inited" not in st.session_state:
    st.session_state.inited = True
    st.session_state.scored = dict(SAMPLE)
    for k, _ in FIELDS:
        st.session_state[f"in_{k}"] = SAMPLE[k]


def apply_preset():
    choice = st.session_state.preset
    if choice in PRESETS:
        lead = PRESETS[choice]
        for k, _ in FIELDS:
            st.session_state[f"in_{k}"] = lead.get(k, "")
        st.session_state.scored = {k: lead.get(k, "") for k, _ in FIELDS}


# ------------------------------------------------------------------ header + sidebar
st.markdown("""<div class="hero"><h1>🎯 AI-Based Lead Scoring System</h1>
<p>Know which leads to call first, and exactly why. Every score is calculated by transparent weighted rules, and AI is used only to explain it.</p>
<span class="chip">✔ Explainable</span><span class="chip">✔ Consistent</span><span class="chip">✔ Handles missing data</span><span class="chip">✔ 100-point model</span></div>""",
            unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    use_llm = st.toggle("AI narrative (LLM)", value=llm_available(),
                        help="Needs ANTHROPIC_API_KEY. The score itself never depends on the LLM.")
    if use_llm and not llm_available():
        st.warning("ANTHROPIC_API_KEY not set, so rule-based text is used instead.")
    st.markdown(f"**Classification**  \n🔥 Hot ≥ {HOT_THRESHOLD}  \n🌤️ Warm {WARM_THRESHOLD}–{HOT_THRESHOLD - 1}  \n❄️ Cold < {WARM_THRESHOLD}")
    st.caption("Same input always gives the same score.")

tab1, tab2, tab3 = st.tabs(["🧑‍💼 Score a lead", "📂 Batch scoring", "📘 Methodology"])

# ------------------------------------------------------------------ single lead
with tab1:
    left, right = st.columns([4, 6], gap="large")
    with left:
        st.selectbox("Quick-load a demo lead", ["— choose —"] + list(PRESETS), key="preset", on_change=apply_preset)
        with st.form("lead_form"):
            vals = {}
            c1, c2 = st.columns(2)
            for i, (k, label) in enumerate(FIELDS):
                if k == "requirement":
                    vals[k] = st.text_input(label, key=f"in_{k}")
                else:
                    vals[k] = (c1 if i % 2 == 0 else c2).text_input(label, key=f"in_{k}")
            submitted = st.form_submit_button("Score lead", type="primary", use_container_width=True)
        if submitted:
            st.session_state.scored = vals
    with right:
        lead = st.session_state.scored
        render_result(score_lead(lead), lead, use_llm)

# ------------------------------------------------------------------ batch
with tab2:
    st.write("Upload a CSV with columns such as: " + ", ".join(k for k, _ in FIELDS))
    d1, d2 = st.columns([1, 3])
    sample_csv = pd.DataFrame(list(PRESETS.values())).to_csv(index=False)
    d1.download_button("Download sample CSV", sample_csv, "sample_leads.csv", "text/csv")
    up = d2.file_uploader("CSV file", type="csv", label_visibility="collapsed")
    if up:
        df = pd.read_csv(up)
        rows = []
        for rec in df.to_dict("records"):
            r = score_lead(rec)
            rows.append({"Name": r.lead_name, "Score": r.score, "Class": r.classification,
                         "Confidence": f"{r.confidence} ({r.confidence_pct}%)", "Next action": r.next_action})
        out = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
        counts = out["Class"].value_counts()
        cols = st.columns(4)
        cols[0].markdown(f'<div class="card kpi"><div class="l">Total leads</div><div class="v">{len(out)}</div></div>', unsafe_allow_html=True)
        for col, cls in zip(cols[1:], ["Hot", "Warm", "Cold"]):
            col.markdown(f'<div class="card kpi"><div class="l">{EMOJI[cls]} {cls}</div><div class="v" style="color:{COLORS[cls]}">{int(counts.get(cls, 0))}</div></div>', unsafe_allow_html=True)
        mx = max(int(out["Score"].max()), 1)
        bars = "".join(f'<div class="bar"><span style="width:130px">{esc(html_name)}</span><div class="b" style="width:{s / mx * 60:.0f}%;background:{COLORS[c]}"></div><b>{s}</b></div>'
                       for html_name, s, c in zip(out["Name"], out["Score"], out["Class"]))
        st.markdown(f'<div class="card"><h4>Lead ranking</h4>{bars}</div>', unsafe_allow_html=True)
        st.dataframe(out, hide_index=True, use_container_width=True,
                     column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d")})
        st.download_button("⬇️ Download results", out.to_csv(index=False), "scored_leads.csv", "text/csv")

# ------------------------------------------------------------------ methodology
with tab3:
    st.markdown("### How the score is calculated")
    st.write("Each parameter earns a credit between 0% and 100% of its weight. The credits add up to a score out of 100. "
             "No LLM is involved in this step, so results are repeatable and auditable.")
    mx = max(WEIGHTS.values())
    wb = "".join(f'<div class="bar"><span style="width:130px">{LABEL(k)}</span><div class="b" style="width:{v / mx * 55:.0f}%;background:#7c3aed"></div><b>{v}</b></div>'
                 for k, v in sorted(WEIGHTS.items(), key=lambda x: -x[1]))
    st.markdown(f'<div class="card"><h4>Weightage (total 100)</h4>{wb}</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    a.markdown(f"""<div class="card"><h4>Classification</h4>🔥 <b>Hot</b>: {HOT_THRESHOLD}+<br>🌤️ <b>Warm</b>: {WARM_THRESHOLD}–{HOT_THRESHOLD - 1}<br>❄️ <b>Cold</b>: below {WARM_THRESHOLD}</div>""", unsafe_allow_html=True)
    b.markdown("""<div class="card"><h4>Safeguards</h4>Missing fields are flagged and lower confidence.<br>A lead with no usable email or phone can't be Hot.<br>Placeholder emails (example.com) are penalised.</div>""", unsafe_allow_html=True)
    st.markdown("**Where the LLM is used:** it receives the finished score and factor breakdown, then writes the analysis, risks and outreach tip. It cannot change the numbers.")
