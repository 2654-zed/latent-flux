"""Separate today's drains by token and report correct USD figures."""
import sqlite3, json

DB = "/app/surveillance/data/surveillance.db"
USDC_BASE = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"
USDC_ARB  = "0xaf88d065e77c8cc2239327c5edb3a432268e5831"
USDT_ARB  = "0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9"  # Arbitrum USDT (6dec)
STABLECOINS = {USDC_BASE, USDC_ARB, USDT_ARB}

c = sqlite3.connect(DB)

usdc_total = 0.0
unknown = []
rows = c.execute(
    "SELECT payload, timestamp FROM alerts WHERE alert_type='X402_AGENT_DRAIN' "
    "AND timestamp >= '2026-04-18T00:00:00' ORDER BY timestamp"
).fetchall()

for r in rows:
    p = json.loads(r[0])
    tok = p["token"].lower()
    chain = p.get("chain", "?")
    raw = p["amount"]
    norm = p.get("amount_normalized_6dec", 0)
    if tok in STABLECOINS:
        usdc_total += norm
        print(f"  {r[1][:19]}  {chain:9s}  stablecoin     raw={raw:>18,}  ${norm:>10,.2f}")
    else:
        unknown.append((r[1][:19], raw, tok, chain))
        print(f"  {r[1][:19]}  {chain:9s}  NON-stablecoin raw={raw:>22,}  "
              f"token={tok[:18]}...  (display ${norm:,.2f} is WRONG if token decimals != 6)")

print()
print(f"Total STABLECOIN drains: ${usdc_total:,.2f} ({len(rows)-len(unknown)} events)")
print(f"Non-stablecoin drains: {len(unknown)} events (amount-normalization bug inflates display)")
for ts, raw, tok, chain in unknown:
    # If token has 18 decimals, real token count is raw / 1e18
    real_18 = raw / 1e18
    print(f"  {ts}  {chain:9s}  token={tok[:18]}...  raw={raw:,}  IF 18-dec -> {real_18:.6f} tokens (USD depends on token price)")
