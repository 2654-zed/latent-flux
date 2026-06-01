"""Task 0 — structural integrity check of approval_drain_monitor.py.
Writes results to a file (avoids stdout channel-garble)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src_path = ROOT / "surveillance" / "approval_drain_monitor.py"
out = ROOT / "reports" / "_task0_verify.txt"

src = src_path.read_text(encoding="utf-8")
checks = {
    "line_count": src.count("\n") + 1,
    "check_drains_defined": "def check_drains(" in src,
    "method1_transferfrom_present": "'23b872dd'" in src,
    "method1_reverted_filter": "is_reverted = 0" in src,
    "method2_disabled_marker": "Method 2: DISABLED" in src,
    "method2_body_removed": "deployer_drains" not in src,
    "correction24_note": "Correction #24" in src,
    "correction27_note": "Correction #27" in src,
}
import ast
try:
    ast.parse(src)
    checks["ast_parse"] = "OK"
except SyntaxError as e:
    checks["ast_parse"] = f"FAIL: {e}"

lines = ["TASK 0 — approval_drain_monitor.py structural verification", "=" * 55]
for k, v in checks.items():
    lines.append(f"  {k:32s}: {v}")
# verdict
ok = (checks["ast_parse"] == "OK"
      and checks["check_drains_defined"]
      and checks["method1_transferfrom_present"]
      and checks["method1_reverted_filter"]
      and checks["method2_disabled_marker"]
      and checks["method2_body_removed"])
lines.append("")
lines.append(f"VERDICT: {'INTACT — matches Correction #27 prevention fix' if ok else 'PROBLEM — investigate'}")
out.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {out}")
