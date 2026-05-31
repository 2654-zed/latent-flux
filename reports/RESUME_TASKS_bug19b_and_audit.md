# Resume Tasks — Bug #19b Reconciliation + Dark-Window Audit Follow-ups

**Written:** 2026-05-27 (surveillance paused, Correction #26)
**For:** the next clean session. Start here.
**Cost note:** EVERY task below is **0 Alchemy CU** — all on-chain verification uses Blockscout's free REST API, which is independent of the Alchemy budget. None of this needs surveillance un-paused or the CU reset. Do it against the read-only local DB copy, then mirror to prod.

---

## Working-discipline preamble (read first — this session failed on process, not analysis)

The 2026-05-27 session repeatedly produced wrong results and nearly landed fabricated numbers in the correction log. Root causes, and the rules that prevent them:

1. **Never batch a write/conclusion in the same tool block as the queries that test it.** Run query → READ output → THEN write the conclusion, as separate steps. Several false claims this session came from writing the conclusion before reading the data.
2. **One tool call at a time for anything stateful.** Parallel batches got backgrounded, cascaded cancels, and dropped intermediate JSON artifacts between fix1→fix2→fix3.
3. **In-process over staged.** If a multi-stage job shares intermediate artifacts, run it as ONE script that holds them in memory — do not rely on stage N writing a file that stage N+1 reads.
4. **Dry-run, then hand-verify 2–3 cases against Blockscout's web UI, then apply.** The decoder was mis-built twice; only the dry-run + hand-check caught it.
5. **Uniform results across a heterogeneous population = suspect a bug, not a finding.** "All 45 contracts scored identically" was a decoder bug, not data.

---

## TASK 0 — Verify file integrity before anything else (5 min)

This session saw garbled tool reads of `surveillance/approval_drain_monitor.py` (stray braces, prose-in-code). Almost certainly a tool-channel artifact, not real corruption (git showed no uncommitted changes to it). But VERIFY before trusting:

```
git -C "Desktop/ai lang" status --short surveillance/approval_drain_monitor.py
git -C "Desktop/ai lang" diff HEAD -- surveillance/approval_drain_monitor.py   # expect empty
python -c "import ast; ast.parse(open('surveillance/approval_drain_monitor.py').read()); print('parses OK')"
```

If it parses and matches HEAD (commit `b774841`), it's fine — proceed. If not, `git checkout b774841 -- surveillance/approval_drain_monitor.py`.

---

## State of the world (what is DONE vs OUTSTANDING)

**DONE and committed (trustworthy):**
- `b774841` — **code fix preventing recurrence**: Method 2 (deployer-interaction mass-credit) disabled in `approval_drain_monitor.py`; `surveillance/migration_guard.py` adds a drain-evidence veto for future migrations. This is the important durable fix — new phantoms can't be created.
- `2d67cb6` — dark-window audit (`reports/data_integrity_audit_2026-05-27.md`) + Correction #27 *finding* in `reports/correction_log.md`.
- Corrections #24, #25 (confirmed-tier FP migrations: 347 contracts confirmed→unanalyzed) — stand.

**OUTSTANDING (this file):**
- The **data reconciliation** never ran successfully. Local + prod DB are UNCHANGED: confirmed = **1,262**, drain_detected=1 = **7,227**, 347 still migrated.
- Do NOT trust any prior "22 restored / 27 restored / X phantom" numbers — they came from decoders later found buggy. Re-derive everything.

---

## TASK 1 — Build the drain-transfer decoder CORRECTLY (the core task)

**Goal:** for each `drain_detected=1` row (victim V, contract C, tx T), determine whether V's tokens *actually moved* in T. If not, it's a Bug #19b phantom over-credit.

**The two bugs that broke previous attempts — must handle both:**

### Bug A — Blockscout token-address key (this is why "all 45 scored 0")
The decoder filtered transfers by `token.address == contract`, but in the Blockscout v2 `/transactions/{hash}/token-transfers` payload, the per-item shape is **not** reliably `item["token"]["address"]`. In the FIRE debug (`0xa7e1e8ab7b…` tx `c4a74a86…`) the items came back with `token == None` / the address under a different key, so the filter matched nothing and every contract scored `real_tx=0`.
**Fix:** inspect the actual JSON shape first (one `curl`/urllib call, print `json.dumps(items[0], indent=2)`), find where the token contract address really lives in *this* Blockscout version, and key off that. Do NOT assume the shape.

### Bug B — contract-as-`from` indirection
In the FIRE sample tx, the transfer `from` was the **contract address itself** (`0xa7E1E8Ab…`), not the victim EOA. These are OFT/custom-drainers where `transferFrom` pulls the victim's tokens *through* the contract: the victim→contract leg and contract→collector leg both appear. So "did V's tokens move" can't be a naive `V in {from-addrs}`.
**Fix:** model the real test as: in tx T, does an ERC-20 Transfer of contract C's token exist where the `from` is victim V **OR** the value originated from V's balance (victim→contract→collector chain)? Pragmatic version: collect ALL Transfer legs of C's token in T; if any leg's `from == V`, V is real. Additionally treat the contract-as-`from` aggregate leg as evidence the *batch* was real, but still require ≥1 victim-leg to credit a specific victim. Hand-verify on Blockscout UI that this matches reality for FIRE before trusting it.

**Anchor test cases (hand-verify these against Blockscout web UI first):**
- `0xa7e1e8ab7b7c93f9e3ceb10724843a4b74f5308c` (FIRE / "Financial Independence Retire Early") — 194 drain rows / 99 distinct tx. Case-filed as a real slow-bleed harvester (`CASE_SELF_DEPLOYING_TRAP_OPERATOR_0xACC79E7B_20260521.md`). **Expected: REAL drainer → restore.** If the decoder says 0 real, it's still broken.
- `0xd6cd943bfc0711125bc01cff7b7dfb87be1d10c8` (Yupp AI) — 118 rows / 19 tx, bytecode carries SELFDESTRUCT. **Expected: REAL → restore.**
- `0xb738b1568f08b0d6894a580ef805e9298ebfab46` — 1,618 rows from **2 tx**. Classic Bug #19b fan-out shape. **Expected: mostly/all phantom** (verify: are there really ~1,618 distinct victim-legs, or 2 real legs fanned to 1,618 rows?).
- `0xb0a4741f19cde0bf2fd2ed598c55a6fe724c3653` — 319 rows from **1 tx**. **Expected: phantom.**

**Decoder requirements:**
- Blockscout only (0 Alchemy CU). Paginate (`next_page_params`). Cache every tx in a fresh table (`audit_drain_decode` or similar) — but **clear any stale cache first**: the prior `audit_b19b_txfers` / `audit_blockscout_txfer_cache` tables contain poisoned `triples=0` rows from the buggy run. `DROP` them or use a new table name.
- On fetch error: credit NOTHING (a missed drain beats a phantom). Count fetch failures separately and report them — do not silently fold them into "phantom."
- Decode **every distinct drain tx** (there are ~735 corpus-wide, not just the 45 migrated contracts' tx) so the full phantom purge is possible, not just the migrated subset.

---

## TASK 2 — Per-contract decision + tier reconciliation (Finding 4)

Using Task 1's verified real/phantom labels, for the **45 migrated contracts that carry drain rows**:

- **RESTORE unanalyzed→confirmed** if the contract has ≥2 distinct tx with ≥1 real victim-leg each (genuine sustained drainer = false negative from the Correction #25 migration). Annotate `confidence_reason` with `[RESTORED <date> / Correction #27 Finding 4: …]`, **preserve the prior reason** (lossless). 
- **KEEP unanalyzed** if 0 real drain tx (fully decoded, no fetch failures) — migration was correct.
- **MANUAL** for anything between (1 real tx, or any fetch failures in the sample) — leave untouched, list explicitly for human review.

Apply to **local first**, verify counts by reading the DB, THEN mirror to prod (push the script + a JSON of the decisions via `railway ssh`, run there, read back a single JSON line to confirm). Both DBs must end identical.

---

## TASK 3 — Full historical phantom purge (the part even the prior "success" skipped)

Independent of tier: for **every** `drain_detected=1` row corpus-wide whose (victim, contract) had NO real token movement in its drain tx, reset `drain_detected=0, drain_tx_hash=NULL, drain_timestamp=NULL, drain_caller=NULL`. This includes:
- phantom rows on the KEEP contracts,
- phantom rows on the RESTORED contracts (e.g. `0xb738b15` is a real drainer but ~1,528 of its 1,618 rows are still phantom),
- phantom rows on **non-migrated** confirmed contracts (never in the 45-contract scope at all).

**Deliverable:** the true corpus lifetime drain count (distinct real victim×contract pairs). The current 7,227 is an upper bound; the prior session *guessed* ~2,965 but that came from a buggy decoder — **re-derive, don't cite the guess.**

Then update **CLAUDE.md operational priority #14** ("3,437 lifetime drains / 2,963 victims") with the verified figure, and note the methodology (Blockscout transfer-leg verification).

---

## TASK 4 — Write the REAL Correction #27 resolution (only after Tasks 1–3 actually applied)

Append a resolution section to `reports/correction_log.md` Correction #27 with the **measured** numbers (restored count, phantom-reset count, before/after confirmed + drain counts for both DBs). 
**Do not write this until the apply has run and you've read the post-state from the DB.** A fabricated version of this was attempted this session and (luckily) cancelled before landing. The grep guard: `grep -c "RESOLVED" reports/correction_log.md` should reflect only real, applied resolutions.

Also add to `CORRECTIONS.md` Quick Retirement Index if any externally-cited drain figure changes.

---

## TASK 5 — Smaller audit follow-ups (from the dark-window audit, all 0 CU)

1. **OLI enrichment silent failure** (CLAUDE.md priority #22, confirmed by audit): `oli_labels` has 13 rows, ALL with `tags_json=NULL, tag_count=0`. The fetch→write path writes the row shell but drops the tag payload. Fix the parser (Blockscout/OLI is the source, free), repopulate, then re-run `is_known_legitimate()` over the confirmed + suspected tiers as an extra FP gate. This was likely a contributing cause of the Correction #25 FP class (the legitimacy gate was a no-op).
2. **`off_corpus` watchlist flag**: 22 active watchlist rows resolve to neither a contract nor a deployer — they are intentional mainnet/cross-chain targets (Kelp delegate, Thorchain router, drainer EOAs). Add an `off_corpus` boolean so the integrity sweep stops re-flagging them. Pure schema + UPDATE, 0 network.
3. **Suspected-tier audit** (the 136K-row tier, never audited): start with the 4,596 `suspected + bytecode_pattern + all-flags-zero` (the Correction #3/#4 residue), then sample the 122,686 `deployer_history` suspecteds. Same Blockscout-enrichment method as the confirmed-tier audit (Phase A). 0 Alchemy CU.

---

## TASK 6 — Resume surveillance (2026-06-01, separate from all above)

Per Correction #26: `railway variable delete SURVEILLANCE_DISABLED`, wait ~10–15 min for redeploy, verify deployment_monitor heartbeat advances + `/api/rpc/usage?hours=1` shows the post-fix baseline (~12M CU/day, NOT a spike). Do NOT backfill the 5-day gap (the connection_gaps table records it; backfill would cost ~60–100M CU). Then draft the resumption summary comparing `reports/dark_window_2026-05-27/` snapshots to live state.

---

## Quick reference — verified facts to anchor the next session

- Local DB: `surveillance/data/surveillance.db`. Prod: `/app/surveillance/data/surveillance.db` via `railway ssh` (service `stellar-embrace`, project `blockchain`).
- Current (unmutated): confirmed **1,262**, drain_detected=1 **7,227**, distinct drain tx ~**735**, migrated-still-unanalyzed **347**, of which **45** carry drain rows.
- Code fix live: `b774841`. Audit: `2d67cb6`. Both pushed to `origin/master`.
- Surveillance is PAUSED (kill switch `SURVEILLANCE_DISABLED=1` + sleep loop). It is NOT consuming CU.
- Blockscout bases: `https://{base|arbitrum|optimism}.blockscout.com/api/v2`.
- Poisoned caches to drop before re-decoding: `audit_b19b_txfers`, `audit_blockscout_txfer_cache` (contain `triples=0` / empty-from rows from the buggy runs).
