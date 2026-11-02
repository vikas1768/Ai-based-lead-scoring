"""Optional LLM layer: writes a richer narrative around a score that is ALREADY fixed.

The LLM never produces or changes the score, classification or confidence.
If no API key is configured (or the call fails), the deterministic explanation
from scoring.py is used, so the app always works.
"""
from __future__ import annotations

import json
import os

from scoring import ScoreResult

MODEL = os.getenv("LEAD_LLM_MODEL", "claude-sonnet-5-5")

PROMPT = """You are a sales-operations analyst. A deterministic rule engine has already scored a lead.
Do NOT change, recompute or question the numbers. Using only the facts below, write:
1. "analysis": 3-4 sentences on why this lead is (or is not) worth pursuing.
2. "risks": 1-2 sentences on the main uncertainties or red flags.
3. "outreach_tip": 1 sentence with a concrete tip for the first contact.

Return ONLY a JSON object with keys analysis, risks, outreach_tip.

LEAD:
{lead}

ENGINE RESULT:
score={score}/100, class={cls}, confidence={conf}
Factor breakdown:
{factors}
"""


def llm_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def llm_insights(result: ScoreResult, lead: dict) -> dict:
    """Return {'analysis','risks','outreach_tip','source'}; never raises."""
    fallback = {
        "analysis": result.explanation,
        "risks": "; ".join(n.split(":")[0] for n in result.negatives[:2]) or "No major red flags detected.",
        "outreach_tip": result.next_action,
        "source": "rule-based",
    }
    if not llm_available():
        return fallback
    try:
        import anthropic

        client = anthropic.Anthropic()
        factors = "\n".join(
            f"- {f.parameter}: {f.points}/{f.weight} ({f.reason})" for f in result.factors
        )
        msg = client.messages.create(
            model=MODEL,
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": PROMPT.format(
                lead=json.dumps(lead, ensure_ascii=False), score=result.score,
                cls=result.classification, conf=result.confidence, factors=factors)}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        text = text[text.find("{"): text.rfind("}") + 1]
        data = json.loads(text)
        return {
            "analysis": str(data.get("analysis", fallback["analysis"])),
            "risks": str(data.get("risks", fallback["risks"])),
            "outreach_tip": str(data.get("outreach_tip", fallback["outreach_tip"])),
            "source": f"LLM ({MODEL})",
        }
    except Exception as exc:  # network, auth, parse errors -> graceful fallback
        fallback["source"] = f"rule-based (LLM unavailable: {type(exc).__name__})"
        return fallback
