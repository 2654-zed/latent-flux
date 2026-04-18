"""Diagnose DORMANT_ACTIVATION alert schema — is alerts.address the
deployer or the activated contract?"""
import sqlite3, json

c = sqlite3.connect("/app/surveillance/data/surveillance.db")

print("All DORMANT_ACTIVATION from today 2026-04-18:")
for r in c.execute(
    "SELECT address, payload, timestamp FROM alerts "
    "WHERE alert_type = 'DORMANT_ACTIVATION' AND timestamp >= '2026-04-18' "
    "ORDER BY timestamp"
):
    try:
        p = json.loads(r[1] or "{}")
    except Exception:
        p = {}
    addr = (r[0] or "").lower()
    dep = (p.get("deployer") or "").lower()
    print(f"  ts={r[2][:19]}")
    print(f"    alerts.address = {addr}")
    print(f"    payload.deployer = {dep}")
    print(f"    match? {addr == dep}")
    print(f"    payload keys: {list(p.keys())}")
