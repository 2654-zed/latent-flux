# Task 1 Finding — BOTH classes exist: some contracts are distribution-mislabels, some are real drainers. Per-contract verification required.

**Date:** 2026-05-27 (dark window, 0 Alchemy CU — Blockscout probes only)
**Status:** Investigation result. **No DB mutation performed.**

> **CORRECTION (same day, before any mutation):** An earlier version of this
> doc concluded "25/25 victims IN_only → all distribution mislabels → restore
> none." That was WRONG — written from the FIRE-only probe before reading the
> broader archetype probe. The full 5-contract archetype probe
> (`_t1_archetypes.txt`) shows the opposite for 4 of 5 contracts: they have
> real victim-outbound drain legs. The corrected finding is below. This is a
> read-before-conclude miss, caught at the read step; nothing was applied off
> the wrong version.

## What Task 1 set out to do
Rebuild the drain-transfer decoder to separate real approval-drains from Bug #19b phantom over-credits, then restore wrongly-migrated real drainers (Finding 4) and purge phantoms.

## What the discovery probes actually found

Three Blockscout probes (`scripts/t1_probe_shape.py`, `t1_probe_victim.py`, `t1_probe_archetypes.py`; raw outputs `reports/_t1_*.txt`):

### 1. The two known decoder bugs, root-caused
- **Bug A (token key):** Blockscout v2 puts the token address at `item["token"]["address_hash"]`, NOT `item["token"]["address"]`. The prior decoders read `.address` → got `None` → matched nothing → every contract scored `real_tx=0`. (This is why the buggy dry-run said "45 KEEP / 0 restore".)
- **Bug B (indirection):** in a FIRE drain tx, the `from` is the **contract itself** (`0xa7e1e8ab…` → Uniswap V2), i.e. the contract dumping to a DEX — not a victim→contract leg. The stored `drain_tx_hash` is frequently the contract's dump tx, which contains no victim address at all.

### 2. The decisive finding — TWO classes, split by victim flow direction
Per-victim ERC-20 transfer history (address-level, filtered to the contract's token), 5 victims each across 5 contracts spanning both archetypes (`_t1_archetypes.txt`):

| Contract | rows/tx | OUT (victim sent = real drain) | IN_only (recv = mislabel) | none | class |
|---|---|---|---|---|---|
| FIRE `0xa7e1e8ab7b` | 194/99 | **0** | 5 | 0 | **DISTRIBUTION MISLABEL** |
| Yupp AI `0xd6cd943bfc` (+SELFDESTRUCT) | 118/19 | **5** | 0 | 0 | **REAL DRAINER** |
| `0xb738b15` | 1618/2 | **5** | 0 | 0 | **REAL DRAINER** (rows still Bug#19b-inflated) |
| `0xb0a4741f` | 319/1 | **1** | 0 | 4 | mixed/weak — needs full set |
| `0xaa9c0875` | 399/5 | **5** | 0 | 0 | **REAL DRAINER** |

**Both classes are real:**
- **FIRE** — 5/5 victims have ZERO outbound legs; every leg is `from=contract`/`from=deployer` → `to=victim`. These recipients only RECEIVED the token. Not an approval-drain. The `drain_detected=1` rows are a **distribution/airdrop mislabel** (or honeypot buy-in).
- **Yupp AI, 0xb738, 0xaa9c** — victims have real outbound (`from=victim`) Transfer legs of the token: genuine approval-drains. These are **real drainers**, correctly flagged by the behavioral pipeline. (For `0xb738`, real ≠ uninflated: it still shows 1,618 rows from 2 stored tx, so the rows need Bug#19b dedup even though the contract is a true drainer.)

## What this means

1. **There is no single verdict** — the 45 migrated drain-tainted contracts split into distribution-mislabels (FIRE-type, drain rows bogus) and real drainers (Yupp/b738/aa9c-type, drain rows real but possibly inflated). The reconciliation must be **per-contract**, decided by the victim-outbound-leg test.

2. **The restore question is live again, but on corrected evidence.** Real-drainer contracts that the Correction #25 migration moved to unanalyzed (e.g. Yupp AI, which also carries a SELFDESTRUCT) ARE false negatives and should be restored to confirmed. FIRE-type should stay unanalyzed (or be re-examined as honeypot, separately). The prior "restore ~27" number is still not trustworthy — it came from the Bug-A decoder — but the *category* of restorations is real. Re-derive the count from the corrected decoder.

3. **The phantom test:** a `drain_detected=1` row (victim V, contract C) is real iff V has ≥1 outbound (`from=V`) Transfer of C's token. Reset rows that fail it. This both purges FIRE-type contracts entirely AND dedups inflated rows on real drainers.

## Important caveat — this is a SAMPLE
5 victims × 5 contracts. The split is clear at the contract level but the per-contract verdict must be confirmed on the **full victim set of all 45** (and the OUT-leg test run over every drain row), because a contract can be mixed (`0xb0a4` showed 1 OUT + 4 none in its sample). No mutation until the full run.

## Honeypot vs distribution (FIRE-type only; separate task)
FIRE's `IN_only` pattern is consistent with BOTH legitimate airdrop and honeypot buy-in (victim can't sell). Distinguishing needs a sellability test — separate from the drain fix. For the drain bug it doesn't matter: FIRE-type rows are not approval-drains either way.

## Corrected next steps (supersedes RESUME_TASKS Task 1–2 restore logic)
1. Build the decoder around the **victim-as-`from` outbound-leg** test (token key: `token.address_hash`; query the victim's address-level token-transfer history filtered to the contract token — NOT the stored drain tx, which is often the contract's dump leg). 0 CU.
2. Run it over the **full** victim set of all 45 migrated contracts → per-contract real-victim count.
3. **Restore** (unanalyzed→confirmed) contracts with a real-victim-outbound majority (Yupp/b738/aa9c-type). **Keep unanalyzed** FIRE-type (zero victim-outbound).
4. **Reset** every `drain_detected=1` row failing the per-row OUT-leg test — purges FIRE-type entirely and dedups inflated rows on real drainers.
5. Extend to all ~735 drain tx corpus-wide; re-derive true lifetime drain count.
6. Honeypot-sellability analysis is a SEPARATE future task.

## Artifacts
- `scripts/t1_probe_shape.py` → `reports/_t1_probe_shape.txt` (Blockscout payload shape)
- `scripts/t1_probe_victim.py` → `reports/_t1_probe_victim.txt` (FIRE victim legs, all inbound)
- `scripts/t1_probe_archetypes.py` → `reports/_t1_archetypes.txt` (25/25 IN_only)
