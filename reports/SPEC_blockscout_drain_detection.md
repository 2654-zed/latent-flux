# Spec — Blockscout-Verified Drain Detection (replace the tx_events-join Method 1)

**Date:** 2026-06-05
**Author:** surveillance maintenance
**Status:** SPEC — approved to build this session.
**Cost:** 0 Alchemy CU (Blockscout free REST only). Does not touch the Alchemy budget.

---

## 1. Problem statement (verified, not assumed)

`check_drains()` Method 1 detects a drain by joining `approval_watchlist` against
`transaction_events`:

```sql
SELECT ... FROM transaction_events te
WHERE te.contract_address = ? AND te.function_selector='23b872dd'
  AND te.is_reverted=0 AND te.timestamp > approve_timestamp
```

This structurally **cannot see most drains**. Verified on prod 2026-06-05:

- 64,213 pending approvals (`drain_detected=0`).
- **0** of them have a matching transferFrom in `transaction_events`.
- Only **3** transferFrom (`23b872dd`) events captured in `transaction_events` in the last 24h.

Root cause: `selector_monitor` only logs txs whose `to` is in the **watched
contract set** (suspected/confirmed contracts). A real approval-drain is a
`transferFrom` call the drainer sends to the **token contract** — which is
frequently NOT in the watched set, and during the 2026-05-27→06-01 dark window
NO txs were captured at all. So the drain txs never enter `transaction_events`,
and the join finds nothing.

This is the same blind spot the Bug #19b reconciliation hit: the real drains
were only visible via **Blockscout**, by checking each victim's on-chain
token-transfer history for an outbound (`from=victim`) leg of the contract token.

**Consequence:** drain detection currently reports 0 even though it runs clean.
It is a near-total false-negative, not a quiet market.

## 2. Proven detection primitive (reuse, already validated)

From the reconciliation work (`scripts/t1_apply.py::victim_has_outbound`,
validated against 5,174 victim/contract pairs, 0 errors):

> A `(victim, contract)` approval row is a REAL drain iff the victim has ≥1
> ERC-20 Transfer of the **contract token** with `from == victim` in their
> Blockscout address token-transfer history.
> Endpoint: `GET {blockscout}/addresses/{victim}/token-transfers?type=ERC-20&token={contract}`
> Token address key: `item.token.address_hash` (NOT `.address`). Direction:
> `item.from.hash` / `item.to.hash`. Paginate via `next_page_params`.

This is the correct, evidence-based test. It already powered the 42-drainer
restoration and the phantom purge.

## 3. Design

### 3.1 New function: `check_drains_blockscout(conn, max_victims=None, sleep=0.12)`

Lives in `surveillance/approval_drain_monitor.py` alongside the existing
`check_drains` (do NOT delete the old one — keep it as a fallback / for the
tx_events-captured fast path). New function:

1. Load OLI-suppressed deployers (reuse `_oli_suppressed_deployers`) — keep the
   existing suppression gate (don't flag Circle/Animoca-class distribution).
2. Select pending `(id, victim_address, contract_address, deployer_address, chain)`
   from `approval_watchlist WHERE drain_detected=0`, joined to `contracts` for
   chain. **Tuple-index rows** (connection has no row_factory — this was the
   2026-06-05 regression; do not use dict access).
3. Cache: reuse the `audit_drain_legs(victim, contract, n_out, n_in, truncated,
   err, checked_at)` table from t1_apply so re-runs are incremental and we never
   re-query a known victim. A victim already cached with `n_out>0` → drain; with
   `n_out=0, err IS NULL` → confirmed not-drained (skip); with `err` → retry.
4. For each uncached pending victim: call `victim_has_outbound(base, victim, contract)`.
   - `n_out > 0` → mark `drain_detected=1`. Use the victim's most-recent outbound
     transfer tx as `drain_tx_hash`/`drain_timestamp`; `drain_caller` = the `to`
     of that leg (the collector). (Extend `victim_has_outbound` to also return
     the latest outbound tx_hash + to-address + timestamp.)
   - `n_out == 0, no err` → leave `drain_detected=0` (real not-drained).
   - fetch error → leave pending, count separately, retry next cycle.
5. Commit in batches (every ~200 updates) with `PRAGMA busy_timeout=30000` so it
   coexists with the live monitors writing the same DB.
6. Return `{"drains_detected": N, "checked": M, "cache_hits": C, "errors": E, "oli_suppressed_skips": S}`.

### 3.2 Budget / rate control

- Default `max_victims=400` per invocation. The heartbeat loop calls it every
  ~30 min; 400 victims × ~0.12s ≈ 48s — well within a heartbeat window, and the
  cache means each victim is queried once ever. 64K backlog clears over ~160
  invocations (~3 days) without ever blocking the monitor. A one-time CLI
  backfill (`--drain-scan-all`) can clear the backlog faster out-of-band.
- Hard cap: never exceed `max_victims` Blockscout calls per invocation.

### 3.3 Wiring

In `deployment_monitor.py` heartbeat loop (currently line ~417, the
`heartbeat_count % 30 == 5` block), **replace** the `check_drains(self.conn)`
call with `check_drains_blockscout(self.conn, max_victims=400)`. Keep
`scan_approvals` as-is (it feeds the watchlist). Keep the broad `except` BUT add
an explicit `logger.error` with the exception type so a future regression is not
silent (the 2026-06-05 bug hid for 9 days behind a silent `except`).

### 3.4 Fix the sibling regression

`scan_approvals` (line ~122-131) uses the SAME dict-style row access
(`a["victim"]`, `a["contract_address"]`, …) on the factory-less connection. It
has not thrown only because... it HAS been throwing too — verify and fix it with
tuple unpacking in the same change. (Check: are new approvals actually being
tracked? prod showed 1,426 approval rows/24h, so scan_approvals may be working —
confirm whether it has a row_factory set somewhere, or whether those approvals
come from a different path. RESOLVE before editing.)

## 4. Correctness gates (before wiring to the live loop)

1. **Unit parity:** run `check_drains_blockscout` over the 42 known-restored
   drainer contracts' victims — must re-confirm their drains (n_out>0), matching
   the reconciliation's verdicts already in `audit_drain_legs`.
2. **Negative control:** the 1 known DISTRIBUTION_MISLABEL (`0xf68425d0`, victims
   IN-only) must NOT be flagged (n_out=0).
3. **No-error run:** a 400-victim invocation completes with errors=0 (or retries
   cleanly).
4. **Coexistence:** runs while monitors are live without `database is locked`
   (busy_timeout + batched commits).
5. **Idempotent:** second run with no new approvals flags 0 additional (all cache hits).

## 5. What NOT to do

- Do NOT delete `check_drains` (tx_events fast path may still catch watched-contract drains cheaply; keep both, blockscout as the authoritative pass).
- Do NOT remove OLI suppression (prevents Circle/Animoca distribution FPs — the Correction #20/#24 lesson).
- Do NOT use Alchemy for any of this (Blockscout only — 0 CU).
- Do NOT hot-patch prod; go through git→deploy (the auto-classifier correctly blocks prod SSH writes).
- Do NOT credit a drain on `n_in>0` alone — inbound-only = distribution/airdrop, not a drain (the FIRE/OFC lesson).

## 6. Deliverables

1. `check_drains_blockscout()` + extended `victim_has_outbound()` (return latest outbound tx detail) in `approval_drain_monitor.py`.
2. `scan_approvals` row-access regression fixed (if confirmed broken).
3. Explicit error logging in the heartbeat wiring.
4. CLI `--drain-scan-all` for out-of-band backlog clearing.
5. Correctness-gate test script `scripts/t_drain_blockscout_parity.py`.
6. Correction-log note (this is a detection-method change worth recording).
