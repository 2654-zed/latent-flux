"""Audit high-value drain victims for pass-through classification."""
import json
import sys
import urllib.request

arb = "https://arb-mainnet.g.alchemy.com/v2/UrKIoObtPAGjfHQgkcwc3"
base_url = "https://base-mainnet.g.alchemy.com/v2/UrKIoObtPAGjfHQgkcwc3"

DRAINER_PREFIXES = ["0xce5e", "0xe717", "0xa7b9", "0xe3b2", "0x881e", "0xbec8", "0xd270", "0xf71c"]
DRAINER_ADDRS = {
    "0xbec87a77b19797bbe9b920ec521f3716c3725d22",
    "0xbec8721e796b0ce7705d317a73f110693d895d22",
    "0x785ce546ed429559b95895cb4a07874bf8ed329c",
    "0x881e7c4c90f2d7f013558caf4feca330c327e476",
    "0xce5ec7336f863931fda2ee3e4b9dad99fcc53c91",
    "0xe7176831c898d585cd999bcee9984a7fa9a6be96",
    "0xa7b9874d15742358fb455dd56f97c6d19ad74f5c",
    "0xe3b205da6d47989538f03553bc394d941677ffd3",
    "0xa3a1d7a54269be09c34accfeb4b08adc21a51738",
}


def is_drainer(addr):
    a = addr.lower()
    if a in DRAINER_ADDRS:
        return True
    for p in DRAINER_PREFIXES:
        if a.startswith(p):
            return True
    return False


def rpc(url, method, params):
    data = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read()).get("result")


victims = [
    ("0xa3a1d7a54269be09c34accfeb4b08adc21a51738", "arbitrum", 769270, "CE5E"),
    ("0x785ce546ed429559b95895cb4a07874bf8ed329c", "base", 256321, "E3B2"),
    ("0x303d5773082a740c3040d5763b3d86f84478980f", "arbitrum", 179999, "E717"),
    ("0x59f13bc19a82e9e67703d865eb96a45692760cd5", "base", 29059, "A7B9"),
]

urls = {"arbitrum": arb, "base": base_url}

print("=== HIGH-VALUE VICTIM DEPOSIT SOURCE AUDIT ===\n")
total_pt = 0
total_real = 0

for victim, chain, amount, drainer_name in victims:
    url = urls[chain]
    result = rpc(url, "alchemy_getAssetTransfers", [{
        "toAddress": victim, "category": ["erc20"], "maxCount": "0x14", "order": "desc"
    }])
    transfers = (result or {}).get("transfers", [])

    drainer_sources = []
    clean_sources = []
    for t in transfers:
        src = t.get("from", "").lower()
        val = t.get("value", 0)
        asset = t.get("asset", "?")
        if is_drainer(src):
            drainer_sources.append((src[:20], val, asset))
        else:
            clean_sources.append((src[:20], val, asset))

    if drainer_sources and not clean_sources:
        classification = "PASS_THROUGH"
        total_pt += amount
    elif drainer_sources and clean_sources:
        classification = "MIXED"
        total_pt += amount // 2
        total_real += amount // 2
    else:
        classification = "REAL_DRAIN"
        total_real += amount

    print(f"{victim[:24]} [{chain}] ${amount:,} drained by {drainer_name} -> {classification}")
    for src, val, asset in drainer_sources[:3]:
        print(f"  DRAINER SOURCE: {src}... {val} {asset}")
    for src, val, asset in clean_sources[:3]:
        safe_asset = ascii(asset) if not asset.isascii() else asset
        print(f"  CLEAN SOURCE: {src}... {val} {safe_asset}")
    print()

print(f"SUMMARY:")
print(f"  Pass-through volume: ${total_pt:,}")
print(f"  Real victim volume: ${total_real:,}")
print(f"  Total audited: ${total_pt + total_real:,}")
