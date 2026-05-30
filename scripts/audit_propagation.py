"""Retired-claim propagation sweep. For each of the 347 audit-migrated
contracts (confirmed->unanalyzed) + the OFC/0x752c5a95 retraction set,
check whether they're still cited as live confirmed traps in any
docs/INDEX/cases/reports markdown. Cross-reference addresses, not just
numbers. Single clean run -> file."""
import re, sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "surveillance" / "data" / "surveillance.db"
OUT = ROOT / "reports" / "_audit_propagation.txt"
c = sqlite3.connect(DB)
L=[]
def p(s=""): L.append(str(s))

# migrated addresses
migrated = [r[0].lower() for r in c.execute(
    "SELECT contract_address FROM contracts WHERE confidence_tier='unanalyzed' "
    "AND confidence_reason LIKE '%Correction #25%'")]
p(f"audit-migrated (confirmed->unanalyzed) contracts: {len(migrated)}")

# build a short-prefix set for matching truncated cites (0xABCD…/0xABCDEF…)
full_set = set(migrated)
pref10 = {a[:10] for a in migrated}  # 0x + 8 hex

# scan markdown corpus
md_files = []
for sub in ("docs", "reports", "surveillance/data/cases", "l3-narrative"):
    d = ROOT / sub
    if d.exists():
        md_files += list(d.rglob("*.md"))
md_files += [ROOT / "CORRECTIONS.md", ROOT / "claude.md", ROOT / "POTENTIAL_ATTACKS_V3.md"]
md_files = [f for f in md_files if f.exists()]
p(f"scanning {len(md_files)} markdown files")

hits = []
for f in md_files:
    try:
        txt = f.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    low = txt.lower()
    # full-address hits
    for a in full_set:
        if a in low:
            # capture the line
            for ln in txt.splitlines():
                if a in ln.lower():
                    hits.append((f.relative_to(ROOT), a, ln.strip()[:140]))
                    break
p(f"\nFULL-ADDRESS citations of migrated contracts in markdown: {len(hits)}")
for rel, a, ln in hits[:60]:
    p(f"  {rel}  {a[:12]}…")
    p(f"      | {ln}")

# Specifically: do any cite it AS a confirmed trap / harvester / drain (not as a retraction)?
p("\nContext flags (lines mentioning a migrated addr near trap/confirmed/harvester/drain WITHOUT retract/correction nearby):")
flagged = 0
for f in md_files:
    try:
        txt = f.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    lines = txt.splitlines()
    for i, ln in enumerate(lines):
        low = ln.lower()
        if not any(a in low for a in full_set):
            continue
        ctx = " ".join(lines[max(0,i-2):i+3]).lower()
        cited_bad = any(w in low for w in ("confirmed", "harvester", "trap", "drained", "predator"))
        has_retract = any(w in ctx for w in ("retract", "correction #2", "correction #1", "fp", "false positive", "legitimate", "migrated", "unanalyzed"))
        if cited_bad and not has_retract:
            flagged += 1
            p(f"  [{f.relative_to(ROOT)}] {ln.strip()[:130]}")
p(f"\n  total potentially-live bad citations: {flagged}")

OUT.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {OUT}")
