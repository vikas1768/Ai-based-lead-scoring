"""CLI: python cli.py leads.json   (a JSON object or list of objects)"""
import json
import sys

from scoring import score_lead

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_leads.json"
    data = json.load(open(path, encoding="utf-8"))
    for lead in data if isinstance(data, list) else [data]:
        r = score_lead(lead.get("input", lead))
        print(f"{r.lead_name:<18} {r.score:>3}/100  {r.classification:<5} confidence={r.confidence} ({r.confidence_pct}%)")
        print(f"   {r.next_action}")
