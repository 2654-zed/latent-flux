"""Surface only the draft questions that were derived from sai_alerts.
These are the freshest signal layer in the SAI loop output."""
import yaml
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "memory" / "questions_draft.yaml"
with open(path, encoding="utf-8") as f:
    data = yaml.safe_load(f)

def ascii_safe(s: str) -> str:
    return s.encode("ascii", errors="replace").decode("ascii")

sai_drafts = [d for d in data["drafts"] if d["derived_from"].startswith("Q-")]
print(f"Drafts derived from sai_alerts: {len(sai_drafts)}\n")

for transformation in ("adversarial_inversion", "temporal_upgrade", "decomposition"):
    matching = [d for d in sai_drafts if d["transformation"] == transformation]
    print(f"=== {transformation} ({len(matching)}) ===")
    for d in matching:
        q = d["question"].strip()
        if len(q) > 220:
            q = q[:217] + "..."
        print(f"  derived_from={d['derived_from']}")
        print(f"    {ascii_safe(q)}")
        print()
