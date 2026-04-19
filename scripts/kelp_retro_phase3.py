"""Kelp retrospective Phase 3 — historical DVN configuration verification.

Calls EndpointV2.getConfig(oapp, lib, remoteEid, configType=2) on Ethereum
mainnet at 5 historical blocks BEFORE the attack to prove the catastrophic
1-of-1 DVN configuration was publicly observable for weeks prior.

Budget: 5 RPC calls (one per sampled block).

Parameters from the DK27ss PoC:
- oapp: 0x85d456b2dff1fd8245387c0bfb64dfb700e98ef3 (Kelp OFT adapter)
- lib: 0xc02ab410f0734efa3f14628780e6e695156024c2 (Ethereum receive library)
- remoteEid: 30320 (Unichain source)
- configType: 2 (ULN config — the DVN set)

Output: for each sampled block, decode and print the DVN config. Expected:
requiredDVNCount=1, optionalDVNCount=0, optionalDVNThreshold=0, stable
across the window.
"""
import json
import os
import sys
import urllib.request

# ETH mainnet RPC — convert WSS to HTTPS
ETH_RPC = os.environ.get("ETH_HTTP_URL")
if not ETH_RPC:
    wss = os.environ.get("ETH_WSS_URL", "")
    ETH_RPC = wss.replace("wss://", "https://") if wss else None

# Addresses (lowercased)
ENDPOINT_V2_ETH = "0x1a44076050125825900e736c501f859c50fe728c"
OAPP = "0x85d456b2dff1fd8245387c0bfb64dfb700e98ef3"
RECV_LIB = "0xc02ab410f0734efa3f14628780e6e695156024c2"
REMOTE_EID = 30320  # Unichain
CONFIG_TYPE = 2  # ULN

ATTACK_BLOCK = 24908285
SAMPLE_BLOCKS = [
    24500000,  # ~4 weeks before attack
    24600000,  # ~3 weeks before
    24700000,  # ~2 weeks before
    24800000,  # ~1 week before
    24900000,  # ~8k blocks before attack (~1 day)
]


# Function selector for getConfig(address oapp, address lib, uint32 eid, uint32 configType)
# keccak256 computed via Crypto.Hash.keccak — 0x2b3197b9
SELECTOR = "0x2b3197b9"


def encode_getConfig(oapp: str, lib: str, eid: int, cfg_type: int) -> str:
    """ABI-encode getConfig(address,address,uint32,uint32). 4 params, all
    32 bytes each after the selector."""
    def addr32(a: str) -> str:
        a = a.lower().removeprefix("0x")
        return "0" * 24 + a  # pad to 32 bytes

    def u32(n: int) -> str:
        return f"{n:064x}"

    return SELECTOR + addr32(oapp) + addr32(lib) + u32(eid) + u32(cfg_type)


def decode_uln_config(hex_result: str) -> dict:
    """Decode the ULN config struct from bytes returned by getConfig.

    Response layout (verified by inspection):
      word[0]: outer bytes offset (0x20)
      word[1]: outer bytes length
      word[2]: inner struct offset (0x20) — ULN struct starts at word[3]
      word[3]: confirmations (uint64)
      word[4]: requiredDVNCount (uint8)
      word[5]: optionalDVNCount (uint8)
      word[6]: optionalDVNThreshold (uint8)
      word[7]: offset to requiredDVNs array (relative to inner struct)
      word[8]: offset to optionalDVNs array
      [arrays follow]
    """
    data = hex_result.removeprefix("0x")
    if len(data) < 576:  # 9 words minimum for the struct header
        return {"err": "response too short", "raw_len": len(data)}

    def word(i: int) -> int:
        return int(data[i * 64:(i + 1) * 64], 16)

    def addr_word(i: int) -> str:
        return "0x" + data[i * 64 + 24:(i + 1) * 64]

    try:
        confirmations = word(3)
        req_count = word(4)
        opt_count = word(5)
        opt_threshold = word(6)
        req_arr_offset = word(7)  # byte offset within inner struct
        opt_arr_offset = word(8)
    except Exception as e:
        return {"err": f"header parse: {e}"}

    # Arrays follow. Inner struct starts at word[2]+1 = word[3] in absolute terms.
    # But the offsets (word[7], word[8]) are relative to the inner struct start,
    # i.e., relative to the byte position of word[3] = offset 96 in `data`.
    # Inner start byte offset = 96
    inner_start = 96
    # Required DVNs array
    req_dvns = []
    req_arr_start = inner_start + req_arr_offset
    req_arr_word_idx = req_arr_start // 32
    try:
        arr_len = word(req_arr_word_idx)
        for i in range(arr_len):
            req_dvns.append(addr_word(req_arr_word_idx + 1 + i))
    except Exception:
        pass

    opt_dvns = []
    opt_arr_start = inner_start + opt_arr_offset
    opt_arr_word_idx = opt_arr_start // 32
    try:
        arr_len = word(opt_arr_word_idx)
        for i in range(arr_len):
            opt_dvns.append(addr_word(opt_arr_word_idx + 1 + i))
    except Exception:
        pass

    return {
        "confirmations": confirmations,
        "requiredDVNCount": req_count,
        "optionalDVNCount": opt_count,
        "optionalDVNThreshold": opt_threshold,
        "requiredDVNs": req_dvns,
        "optionalDVNs": opt_dvns,
    }


def eth_call(to: str, data: str, block: str | int):
    block_hex = hex(block) if isinstance(block, int) else block
    req = urllib.request.Request(
        ETH_RPC,
        method="POST",
        data=json.dumps({
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [{"to": to, "data": data}, block_hex],
            "id": 1,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=12) as r:
        return json.loads(r.read())


def main():
    if not ETH_RPC:
        print("ERROR: ETH_HTTP_URL or ETH_WSS_URL not set")
        return 1

    data = encode_getConfig(OAPP, RECV_LIB, REMOTE_EID, CONFIG_TYPE)

    print(f"=== Kelp Phase 3 — historical DVN config probe ===")
    print(f"endpoint:  {ENDPOINT_V2_ETH}")
    print(f"oapp:      {OAPP}")
    print(f"recv_lib:  {RECV_LIB}")
    print(f"remoteEid: {REMOTE_EID} (Unichain)")
    print(f"configType:{CONFIG_TYPE} (ULN)")
    print(f"attack block: {ATTACK_BLOCK}")
    print()
    print(f"{'block':>10}  {'confirmations':>14}  {'reqDVN':>7}  {'optDVN':>7}  {'optThr':>7}  requiredDVNs")

    results = []
    for blk in SAMPLE_BLOCKS:
        resp = eth_call(ENDPOINT_V2_ETH, data, blk)
        if "error" in resp:
            print(f"  {blk:>10}  ERROR: {resp['error']}")
            continue
        result_hex = resp.get("result", "")
        decoded = decode_uln_config(result_hex)
        if "err" in decoded:
            print(f"  {blk:>10}  decode err: {decoded['err']}")
            continue
        print(f"  {blk:>10}  {decoded['confirmations']:>14}  "
              f"{decoded['requiredDVNCount']:>7}  "
              f"{decoded['optionalDVNCount']:>7}  "
              f"{decoded['optionalDVNThreshold']:>7}  "
              f"{decoded['requiredDVNs']}")
        results.append((blk, decoded))

    # Summary
    print()
    if results:
        all_1of1 = all(
            r[1]["requiredDVNCount"] == 1 and r[1]["optionalDVNCount"] == 0
            and r[1]["optionalDVNThreshold"] == 0
            for r in results
        )
        print(f"1-of-1 config stable across {len(results)} sampled blocks: {all_1of1}")
        first_block = min(r[0] for r in results)
        lead_blocks = ATTACK_BLOCK - first_block
        # ETH block time ~12s average
        lead_days = lead_blocks * 12 / 86400
        print(f"Earliest sampled block: {first_block} (attack block {ATTACK_BLOCK})")
        print(f"Lead time at earliest sample: {lead_blocks:,} blocks ~= {lead_days:.1f} days")


if __name__ == "__main__":
    raise SystemExit(main())
