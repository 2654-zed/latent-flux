"""Score 10 known observation-heavy contracts + 1 contrast case.

Expected: infrastructure addresses present in Railway's corpus should land
HIGH or CRITICAL with the observation_capability component contributing
materially. Contracts not present in the corpus are reported as missing.
"""
import json
import sqlite3
import sys
from pathlib import Path

DB_CANDIDATES = [
    Path("/app/surveillance/data/surveillance.db"),
    Path("surveillance/data/surveillance.db"),
]
for _p in DB_CANDIDATES:
    if _p.exists():
        DB = _p
        break
else:
    print("NO DB FOUND", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, "/app")
sys.path.insert(0, ".")
from surveillance.risk_scoring import score_contract

# Arbitrum canonical addresses where possible. Chainlink feeds: ETH/USD on
# Arbitrum. Others are well-known.
CASES = [
    ("Chainlink ETH/USD Arb",   "0x639Fe6ab55C921f74e7fac1ee960C0B6293ba612"),
    ("Uniswap V3 SwapRouter",   "0xE592427A0AEce92De3Edee1F18E0157C05861564"),
    ("LayerZero EndpointV2",    "0x1a44076050125825900e736c501f859c50fE728c"),
    ("Wormhole Core Arb",       "0xa5f208e072434bC67592E4C49C1B991BA79BCA46"),
    ("1inch AggregationRouter", "0x1111111254EEB25477B68fb85Ed929f73A960582"),
    ("Gelato Ops Arb",          "0x340759c8346A1E6Ed92035FB8B6ec57cE1D82c2c"),
    ("Stargate Router Arb",     "0x53Bf833A5d6c4ddA888F69c22C88C9f356a41614"),
    ("Aave V3 Pool Arb",        "0x794a61358D6845594F94dc1DB02A252b5b4814aD"),
    ("CCTP MessageTransmitter", "0xC30362313FBBA5cf9202e4726c7D44CADB8A0f7c"),
    # Contrast — first org_candidate from the 324 pending
    ("org_candidate (sample)",  None),  # resolved below
]

conn = sqlite3.connect(str(DB), timeout=30)
conn.row_factory = sqlite3.Row

# Resolve one org_candidate for the contrast row
oc = conn.execute(
    "SELECT deployer_addresses FROM org_candidates ORDER BY cluster_size DESC LIMIT 1"
).fetchone()
if oc:
    try:
        deployer_list = json.loads(oc["deployer_addresses"])
        if deployer_list:
            # Find a contract deployed by this deployer
            row = conn.execute(
                "SELECT contract_address FROM contracts WHERE deployer_address = ? LIMIT 1",
                (deployer_list[0],),
            ).fetchone()
            if row:
                CASES[-1] = ("org_candidate (sample)", row["contract_address"])
    except (json.JSONDecodeError, KeyError):
        pass

print(f"{'case':<32} {'addr':<44} {'tier':<10} {'obs':>4} {'stored':>7} {'risk':>8}")
print("-" * 110)
for label, addr in CASES:
    if addr is None:
        print(f"{label:<32} (no candidate contract resolved)")
        continue
    # Ensure lowercase match
    addr_l = addr.lower()
    exists = conn.execute(
        "SELECT 1 FROM contracts WHERE contract_address = ?", (addr_l,)
    ).fetchone()
    if exists is None:
        print(f"{label:<32} {addr:<44} not-in-corpus")
        continue
    r = score_contract(conn, addr_l)
    obs = r.get("observation_capability_score", 0)
    sp = r.get("stored_potential", 0)
    rs = r.get("risk_score", 0)
    tier = r.get("risk_tier", "?")
    print(f"{label:<32} {addr:<44} {tier:<10} {obs:>4} {sp:>7} {rs:>8.2f}")
    # Component details for high-scoring cases
    if obs >= 8:
        oc_comp = r["components"].get("observation_capability", {})
        print(f"     obs components: {oc_comp}")
conn.close()
