# Spec — CU Reduction: Block-Level Receipts + Watched-Set Scoping

**Written:** 2026-06-02
**Status:** SPEC for review. No code changed. Hot-path change to a LIVE monitor — implement only after approval, on a branch, with the verification gates below.
**Problem owner:** `surveillance/` (deployment_monitor + sub-monitors). 0 trading-side impact.

---

## 1. Problem statement (measured, from Alchemy dashboard 2026-06-02)

HTTP CU ≈ **366M/day → ~11B/month**, vs the **2.5B/month limit = 4.4× over**.

| Method | Req/day | CU/day | Share | Origin |
|---|---|---|---|---|
| `eth_getTransactionReceipt` | 19.0M | 285M | **78%** | per-tx receipt fetches in sub-monitors |
| `eth_getBlockByNumber` | 3.5M | 56M | 15% | one full-tx block fetch per block × 3 chains |
| `alchemy_getAssetTransfers` | 104K | 16M | 4% | auto_funder_tracer (1/new deployer) |
| `eth_getLogs` | 106K | 8M | 2% | event_monitors |
| `eth_getCode` | 52K | 1M | 0.3% | bytecode (already Tier-1/2 cached) |

(Jun-02 is the post-restart ramp day; structure, not the ramp, is the issue. Our internal `/api/rpc/usage` telemetry under-counted these because the wrap missed the sync HTTP path — trust the Alchemy dashboard, not the internal number, until the telemetry gap is closed. Tracked separately.)

The 2026-05-22 WebSocket newHeads leak (fixed, trading repo `a810799`) was a real but SECONDARY issue. The dominant cost is HTTP receipts, surveillance-side.

---

## 2. Root cause (confirmed in code)

`surveillance/deployment_monitor.py::_handle_block_header` (line ~498) fetches each block once (`get_block(full_transactions=True)`), then fans the SAME block out to four sub-monitors (lines 626-629):

```
await self._selector_monitor.process_block(w3, block, ts)
await self._revert_detector.process_block(w3, block, ts)
await self._event_monitors.process_block(w3, block, ts)
await self._x402_monitor.process_block(w3, block, ts)
```

**Two of them independently re-fetch receipts, per-transaction:**

- **`selector_monitor.py:178`** — for every tx whose `to` ∈ watched set, calls `get_transaction_receipt(tx.hash)` solely to read `receipt["status"]` (the 1-bit revert flag). The watched set = `db.get_watched_addresses()` = **ALL suspected+confirmed contracts ≈ 137,000 addresses** (135K suspected + ~1.5K confirmed). On Base's tx volume, a large fraction of txs hit a watched address → ~19M receipts/day. **This is the 78% slice.**
- **`revert_cluster_detector.py:98`** — for every tx from a "heavy sender" (≥3 tx in the block), fetches the receipt for `status`. Smaller (already gated on heavy senders) but same wasteful pattern.

Both fetch a full 250+-field receipt object to extract one bit. And `selector_monitor`'s watched set is dominated by *suspected*-tier contracts, which our own audit (Corrections #24/#25) showed are heavily false-positive — so most of the 78% is spent monitoring low-value FP contracts.

`deployment_monitor.py:524` also fetches receipts but ONLY for `to is None` (contract-creation) txs — small, leave as-is (it needs `contractAddress` from the receipt).

---

## 3. Fix — two independent, multiplicative changes

### Fix A (architectural, biggest win): fetch receipts ONCE per block via `eth_getBlockReceipts`

Replace per-tx `get_transaction_receipt` in the sub-monitors with a SINGLE `eth_getBlockReceipts(blockNumber)` call in `_handle_block_header`, then pass the resulting `{tx_hash → receipt}` map down to each sub-monitor.

- `eth_getBlockReceipts` returns ALL receipts for a block in one call. Alchemy CU ≈ 80-ish per call (vs N × 15 for N per-tx fetches).
- On Base (~250-700 tx/block), this collapses hundreds of calls/block into **1**. Net: ~285M CU/day → roughly **3-4M CU/day** on receipts (1 call × ~43K blocks/day × 3 chains × ~80 CU ≈ 10M/day worst case, likely less). **~30-90× reduction on the dominant method.**
- Bonus: removes the per-tx `await` round-trips → faster block processing, less WS idle risk.

**Implementation sketch:**

```
# in _handle_block_header, after get_block:
block_receipts = None
try:
    raw = await w3.provider.make_request("eth_getBlockReceipts", [hex(block_number)])
    items = raw.get("result") or []
    block_receipts = { (r["transactionHash"]).lower(): r for r in items }
except Exception as e:
    logger.warning("eth_getBlockReceipts failed for %d: %s; sub-monitors will degrade", block_number, e)
    block_receipts = None   # sub-monitors fall back / skip receipt-dependent logic

# pass to each sub-monitor:
await self._selector_monitor.process_block(w3, block, ts, block_receipts)
await self._revert_detector.process_block(w3, block, ts, block_receipts)
...
```

Sub-monitors change `get_transaction_receipt(tx.hash)` →
```
rcpt = block_receipts.get(tx["hash"].hex().lower()) if block_receipts else None
is_reverted = (rcpt is not None and int(rcpt.get("status", "0x1"), 16) == 0)
```
Note `eth_getBlockReceipts` returns hex `status` ("0x0"/"0x1") — parse accordingly (the per-tx path returned an int; don't copy that assumption).

**Signature change:** `process_block(self, w3, block, ts)` → `process_block(self, w3, block, ts, block_receipts=None)`. Keep the param optional/defaulted so any other caller / test still works (degrade gracefully when None).

**Fallback discipline (per CLAUDE.md "loud failures, no silent wrong output"):** if `eth_getBlockReceipts` fails or returns partial, do NOT silently fall back to 19M per-tx fetches (that re-creates the bug). Either skip the revert-dependent logic for that block (log it) or, if a chain doesn't support `eth_getBlockReceipts`, gate per-chain. Confirm Base/Arbitrum/Optimism all support it on Alchemy (they do as of 2026; verify at implementation).

### Fix B (scope reduction, independent): shrink the watched set

`db.get_watched_addresses()` currently returns suspected+confirmed (~137K). Two sub-options:

- **B1 (conservative):** confirmed-tier only (~1,500). Drops watched-contract receipt volume ~90×. Justified: per Corrections #24/#25 the suspected tier is heavily FP; selector-monitoring 135K mostly-FP contracts is low-value.
- **B2 (middle):** confirmed + suspected-with-real-evidence (e.g. suspected contracts that have ≥1 approval_watchlist row, or bytecode flags set). Keeps genuine suspects, drops the deployer-derivative noise.

Recommend **B1** for the immediate bleed-stop; revisit B2 once the suspected-tier audit (RESUME_TASKS #3) defines the real-suspect subset.

**With Fix A already in place, Fix B is less CU-critical** (A makes it 1 call/block regardless of watched-set size). But B still matters because it reduces per-block *work* (DB writes, vanity checks, trap-confirmation logic) and shrinks the `tx_events`/`org_transfer_events` growth that bloats the DB (the 17GB sync problem). So do A for CU, B for DB-size + signal quality.

---

## 4. Expected outcome

| | Now | After Fix A | After A+B |
|---|---|---|---|
| receipt CU/day | 285M | ~5-10M | ~5-10M |
| total HTTP CU/day | 366M | ~85M | ~80M |
| **monthly projection** | **~11B (4.4× over)** | **~2.5B (~at limit)** | **~2.4B (under)** |

Fix A alone likely brings it to ~at-limit; A+B + headroom for trading. If still tight, next levers: drop `full_transactions=True` on the block fetch where only deployment detection needs it (but selector/event monitors need tx bodies, so this needs care), or sample blocks on the highest-volume chain.

---

## 5. Verification gates (before merge, and after deploy)

**Pre-deploy (local/branch):**
1. Unit: `is_reverted` from `eth_getBlockReceipts` matches `get_transaction_receipt` for a sample of 20 known txs (10 reverted, 10 success) across all 3 chains — proves the hex-status parse and the tx-hash keying.
2. Confirm Base/Arbitrum/Optimism all return `eth_getBlockReceipts` on the Alchemy endpoints in use.
3. Replay one historical block through old vs new path; assert identical `transaction_events` rows written (same is_reverted, same selectors).

**Post-deploy (prod):**
4. Watch Alchemy dashboard `eth_getTransactionReceipt` req/day — must drop from 19M to near-zero; `eth_getBlockReceipts` appears at ~roughly blocks/day.
5. Confirm `transaction_events` still populates with correct `is_reverted` (spot-check 10 fresh rows vs Blockscout).
6. Monitor heartbeats stay fresh (the change should SPEED UP block processing, not stall it).
7. 24h CU re-projection ≤ 2.5B/month.

**Rollback:** single revert of the branch; the per-tx path is the current `master`, so rollback is clean.

---

## 6. Scope / non-goals

- Do NOT touch the trading repo. This is surveillance-only.
- Do NOT change `deployment_monitor.py:524` (deployment-creation receipt) — it's small and needs the receipt's `contractAddress`.
- The `alchemy_getAssetTransfers` (4%) and `eth_getLogs` (2%) paths are fine for now; revisit only if A+B doesn't clear budget.
- Internal `/api/rpc/usage` telemetry undercounts HTTP — closing that gap is a SEPARATE task (the wrap missed the sync-HTTP and possibly the sub-monitor provider path). Until fixed, use the Alchemy dashboard as ground truth for CU.

---

## 7. Files touched (when implemented)

- `surveillance/deployment_monitor.py` — `_handle_block_header`: add `eth_getBlockReceipts` fetch + pass map to the 4 `process_block` calls (lines ~498, 626-629).
- `surveillance/selector_monitor.py` — `process_block` + `_process_interaction`: accept `block_receipts`, replace line-178 fetch with map lookup.
- `surveillance/revert_cluster_detector.py` — `process_block`: accept `block_receipts`, replace line-98 fetch with map lookup.
- `surveillance/db.py` — `get_watched_addresses`: (Fix B) restrict tier filter, OR add a `confirmed_only=True` param and have selector_monitor pass it. **Verified: this function is consumed ONLY by `selector_monitor.py:104`** — so scoping it has zero collateral impact on other code.
- `event_monitors` / `x402_monitor`: **verified — neither fetches receipts.** `event_monitors.py:781` uses `eth_getLogs` (the separate 2% slice, leave as-is); `x402_monitor` makes no receipt/log calls. Thread `block_receipts` through their `process_block` signatures as an unused param only for call-site consistency (optional — or just don't pass it to them).

## 8. Verified facts (checked at spec time, 2026-06-02)

- Receipt-per-tx consumers in the per-block fan-out: ONLY `selector_monitor.py:178` and `revert_cluster_detector.py:98`. (`deployment_monitor.py:524` is creation-only, out of scope.)
- Watched set = suspected+confirmed ≈ 137K (135K suspected, ~1.5K confirmed); consumed only by selector_monitor.
- `event_monitors` = eth_getLogs (2% slice); `x402_monitor` = no RPC receipts.
- Fan-out call site: `deployment_monitor.py:626-629`. Block fetched once at `:509`.
- Dashboard ground truth (Jun-02): eth_getTransactionReceipt 19M req/day = 78% of HTTP CU.
