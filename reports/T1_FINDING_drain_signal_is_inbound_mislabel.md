# Task 1 Finding — the drain signal is an INBOUND mislabel, not (only) Bug #19b over-credit

**Date:** 2026-05-27 (dark window, 0 Alchemy CU — Blockscout probes only)
**Status:** Investigation result. **No DB mutation performed.** This finding invalidates the prior plan to "restore ~27 real drainers"; do not act on that plan.

## What Task 1 set out to do
Rebuild the drain-transfer decoder to separate real approval-drains from Bug #19b phantom over-credits, then restore wrongly-migrated real drainers (Finding 4) and purge phantoms.

## What the discovery probes actually found

Three Blockscout probes (`scripts/t1_probe_shape.py`, `t1_probe_victim.py`, `t1_probe_archetypes.py`; raw outputs `reports/_t1_*.txt`):

### 1. The two known decoder bugs, root-caused
- **Bug A (token key):** Blockscout v2 puts the token address at `item["token"]["address_hash"]`, NOT `item["token"]["address"]`. The prior decoders read `.address` → got `None` → matched nothing → every contract scored `real_tx=0`. (This is why the buggy dry-run said "45 KEEP / 0 restore".)
- **Bug B (indirection):** in a FIRE drain tx, the `from` is the **contract itself** (`0xa7e1e8ab…` → Uniswap V2), i.e. the contract dumping to a DEX — not a victim→contract leg. The stored `drain_tx_hash` is frequently the contract's dump tx, which contains no victim address at all.

### 2. The decisive finding — victims only ever RECEIVE the token
Per-victim ERC-20 transfer history (address-level, filtered to the contract's token), 5 victims each across 5 contracts spanning both archetypes:

| Contract | rows/tx | sampled victims | OUT legs | IN-only |
|---|---|---|---|---|
| FIRE `0xa7e1e8ab7b` | 194/99 | 5 | **0** | 5 |
| Yupp AI `0xd6cd943bfc` (+SELFDESTRUCT) | 118/19 | 5 | **0** | 5 |
| `0xb738b15` | 1618/2 | 5 | **0** | 5 |
| `0xb0a4741f` | 319/1 | 5 | **0** | 5 |
| `0xaa9c0875` | 399/5 | 5 | **0** | 5 |

**25/25 sampled victims have zero outbound transfers of the token.** Every leg is `from = contract` or `from = deployer`, `to = victim`. A real approval-drain (`transferFrom(victim, collector, amt)`) emits a Transfer with `from = victim`. None exists. **These "victims" were never drained via approval — they received tokens (distribution/airdrop, or honeypot buy-in).**

## What this means

1. **The `drain_detected=1` signal on these contracts is bogus** — not merely inflated (Bug #19b), but pointed at the wrong direction of flow. The pipeline logged token recipients as drain victims.

2. **The prior "restore 27 real drainers" plan is WRONG and must not be executed.** It came from the Bug-A decoder that never matched correctly. Restoring these would push distribution/honeypot-recipient tokens back to confirmed-tier on false drain evidence.

3. **The Correction #25 migration (these 45 → unanalyzed) looks CORRECT after all**, at least w.r.t. drain evidence. Finding 4's "false negative" alarm was itself an artifact of the buggy decoder. (Whether any of these is a *honeypot* by a different mechanism is a separate question that `drain_detected` was never the right signal for.)

4. **The correct phantom test is simpler than from-matching:** a `drain_detected=1` row is real only if the victim has ≥1 outbound (`from=victim`) Transfer of the contract token in/around the drain tx. By that test, the sampled rows are 100% phantom.

## Important caveat — this is a SAMPLE
25 victims across 5 contracts, all IN_only. Strong, consistent, spans both archetypes — but not the full population. Before any mass reset/restore decision, the decoder must run the OUT-leg test across the **full victim set of all 45 migrated contracts** (and ideally all ~735 drain tx corpus-wide). The result may be unanimous (restore none, purge all as phantom) or may surface a real-drain minority. Either way the threshold is: **victim-as-`from` outbound leg of the token = the only valid drain evidence.**

## Honeypot vs distribution (do not conflate)
`IN_only` is consistent with BOTH (a) legitimate airdrop and (b) honeypot buy-in where the victim can't sell. Distinguishing them needs a sellability test (can a holder transfer out?), which is a separate analysis from the drain-reconciliation. For the drain bug specifically, it doesn't matter: neither is an approval-drain, so the drain rows are wrong in both cases.

## Corrected next steps (supersedes RESUME_TASKS Task 1–2 restore logic)
1. Build the decoder around the **victim-as-`from` outbound-leg** test (key: `token.address_hash`; check the victim's address-level token-transfer history, not just the stored drain tx). 0 CU.
2. Run it over the **full** victim set of all 45 migrated contracts.
3. Expected (per sample): most/all rows phantom → reset `drain_detected=0`; restore **none** unless a contract shows genuine victim-outbound drains.
4. Then purge phantoms corpus-wide (the ~735 drain tx) on the same test.
5. Re-derive the true lifetime drain count. The sample suggests it may be **far below** even the prior 2,965 guess — possibly near zero for this migrated subset.
6. Honeypot-sellability analysis is a SEPARATE future task, not part of the drain fix.

## Artifacts
- `scripts/t1_probe_shape.py` → `reports/_t1_probe_shape.txt` (Blockscout payload shape)
- `scripts/t1_probe_victim.py` → `reports/_t1_probe_victim.txt` (FIRE victim legs, all inbound)
- `scripts/t1_probe_archetypes.py` → `reports/_t1_archetypes.txt` (25/25 IN_only)
