# Data-Integrity Audit — Dark Window 2026-05-27

**Filed:** 2026-05-27 (surveillance paused per Correction #26)
**Scope:** Frozen local snapshot `surveillance/data/surveillance.db` (327,064 contracts, 18.26M tx_events, all four Correction #25 migration batches applied: 347 contracts moved confirmed→unanalyzed).
**Method:** Read-only consolidated queries (`scripts/audit_consolidated.py`, `audit_refine.py`, `audit_verify_19b.py`, `audit_phase0_state.py`). No mutations.
**Discipline note:** Early in this session, parallel/streamed query output was misread and produced two fabricated "findings" (a phantom 0x-hash-format split and a 28.7% orphan rate). Both were **false** and are retracted. The corrected method — one script, write to file, Read the file — produced the results below, which are internally cross-checked (e.g. confirmed=1,262 = 666 deployer_history + 596 bytecode_pattern). **No fix was ever applied off the bad reads.**

---

## Verdict: storage layer is clean; the analytic layer has two real defects, one self-inflicted by the audit.

| # | Finding | Severity | Status |
|---|---|---|---|
| 4 | **Migration heuristics never gated on drain evidence: 45/347 migrated contracts had drain rows; ~27 with multi-tx drain histories are false-negative suspects** | **HIGH** | New — most important; Correction #27 amended |
| 1 | Bug #19b: drain-detector credits ALL pending approvers from a single contract interaction | **HIGH** | New — Correction #27 |
| 3 | OLI enrichment silently broken: 13 rows, all `tags_json=NULL tag_count=0` | MEDIUM | Known (priority #22) — confirmed still broken |
| 2 | 22 watchlist rows resolve to neither a contract nor a deployer | LOW | Expected (off-corpus addresses) — documented |
| — | Referential integrity (orphans, dangling FKs, dup watchlist, Phase-0 residue, hash format) | — | **CLEAN** |

Findings 1 and 4 are entangled: both require decoding on-chain transfer logs for the same set of drain transactions, so they share one RPC pass on resume.

**Process note (important):** twice this session I wrote a conclusion ("hash-format inconsistency"; "propagation CLEAN 1/347") into a file *in the same tool-batch as the queries that would have disproven it* — i.e. before reading the output. Both were false and both are corrected above. Root cause: batching the write of a conclusion together with the reads that test it. Discipline going forward: **read query output, then write the conclusion in a separate step.** No DB mutation was ever made off a bad read, so the damage was contained to draft prose caught in the same session.

---

## CLEAN results (the reassuring part)

- **approval_watchlist → contracts:** 0 orphans / 4,627 distinct contracts.
- **drain_tx_hash → transaction_events:** 0 drain rows reference a tx absent from `transaction_events`. (After Phase 0.)
- **Phase 0 persisted:** drain_detected=1 = **7,227** locally (matches the post-Phase-0 figure; the pre-fix 11,850 did not return). 0 drain rows map to a reverted tx.
- **Hash format:** `transaction_events.tx_hash` is 100.00% consistent (18.26M rows, single bare-64-hex format). The earlier "mixed 0x format" claim was a misread — retracted.
- **contracts.deployer_address → deployers:** 2 orphans / 327,064 (0.0%, negligible).
- **bytecode_cache.source_contract → contracts:** 2 orphans / 32,219 (0.0%).
- **watchlist duplicates:** 0 addresses with >1 active row.
- **SAI alerts on retracted data:** 0 of 57 alerts reference an audit-migrated (now-legitimate) contract. The audit migrations did not strand any live SAI alert.

---

## Finding 1 — Bug #19b (HIGH): drain over-crediting confirmed at the row level

Phase 0 (Correction #24) fixed the *reverted-tx* half of the drain bug. The *from-matching* half remains and is now proven with a worst-case row inspection.

**The mechanism (`surveillance/approval_drain_monitor.py` `check_drains`):** both detection methods match a transaction against the *contract*, then stamp `drain_detected=1` onto **every pending approver of that contract**, rather than the single victim whose tokens actually moved. Method 1 (transferFrom scan) and Method 2 (deployer-interaction scan) share this shape.

**Worst-case proof:**

- Drain tx `cf2fed47…bea6d9` on contract `0xb738b1568f08…`
- `transaction_events` shows **exactly ONE** row for that tx_hash: caller `0xa9f65861…`, selector **`f4e2540c`** (a custom method — *not* a transferFrom `23b872dd`), `is_reverted=0`
- `approval_watchlist` credits that single tx to **1,520 distinct victims**, whose approvals span 2026-05-05 → 2026-05-09
- One on-chain action → 1,520 "victims drained." That is definitionally over-crediting.

**Magnitude:**

| Metric | Value |
|---|---|
| drain_detected=1 rows | 7,227 |
| distinct drain tx_hashes | 735 |
| tx_hashes credited to >1 victim | 473 / 735 |
| drain rows attributable to multi-victim tx | **6,965 / 7,227 (96.4%)** |
| drains in ratio≥30 contracts | 3,814 (52.8%) |
| distinct victims among all drains | 6,361 |

**Important caveat (why this is "over-crediting" not "all fake"):** a legitimate batched/multicall drain *can* sweep many victims in one tx. So 6,965 is an **upper bound** on inflation, not a count of fake rows. But the worst case (1,520 victims from a single non-transferFrom call) shows the detector is not validating that each credited victim's tokens actually moved. The true drain count is somewhere between 735 (one-victim-per-tx floor) and 7,227 (current), and cannot be pinned without decoding each tx's transfer logs.

**Correct fix (Phase E, deferred — needs RPC, which is paused):** decode the actual `Transfer`/`transferFrom` logs of each drain tx and credit only the addresses whose balance actually moved. Until then, **any headline "N victims drained" or "N drain events" figure must be quoted as an upper bound** with this caveat.

---

## Finding 2 — Watchlist: 22 unresolved rows (LOW, expected)

22 of 110 active watchlist rows point to addresses not in `contracts` or `deployers`. Inspection shows these are **intentional off-corpus watch targets**, not errors:

- **Mainnet / cross-chain addresses** we watch but don't ingest: `0x1f7a03b7…` (Kelp OApp delegate), `0xa707034429c8…` (mass-dormant-drain hub), `0xd37bbe57…` (Thorchain router off-ramp). These are Ethereum-mainnet; we monitor Base/Arb/OP, so they correctly have no corpus row.
- **Drainer/funder EOAs** flagged from extraction-event investigations (`drainer_funder_*`, `single_purpose_funder_*`, `x402_drain_endpoint_*`) that never themselves deployed a corpus contract.
- **Aurellion / private-key-drain attacker addresses** from case files — external attacker wallets.

**Action:** none required. These are working-as-intended. Recommend adding an `off_corpus` boolean to the watchlist schema on resume so this distinction is explicit rather than inferred, and so this audit doesn't re-flag them every pass.

---

## Finding 3 — OLI enrichment silently broken (MEDIUM, confirms priority #22)

`oli_labels` has 13 rows, **every one** with `tags_json=NULL`, `tag_count=0`, `primary_entity=NULL`, `primary_tag_name=NULL` — only `severity` and `fetched_at` populated, all stamped `2026-05-11 05:12:25`. The fetch path writes the row shell but never persists the actual tag payload from the Blockscout/OLI response.

**Consequence:** `surveillance.oli_enrichment.is_known_legitimate()` — the Correction #20 detection-rule gate that's supposed to stop institutional addresses (Circle, Animoca, Bybit, etc.) from being mislabeled as adversarial — runs on empty data. It cannot return True for anyone because there are no tags to match. This is very likely a contributing cause of the Correction #25 false-positive class (verified-legitimate tokens reaching confirmed tier): the OLI gate that should have caught some of them was a no-op.

The 13 addresses ARE the right ones (they include `0x80b12bd0` Animoca, `0x3304e22d` / `0x39591e7c` / `0x4e3ae00e` CEX hot wallets, `0xd37bbe57` Thorchain) — so the *address selection* works; only the *tag persistence* is broken.

**Action (Phase E, needs RPC):** fix the OLI fetch→write path so `tags_json` + `primary_entity` persist. Then re-run `is_known_legitimate` over the confirmed + suspected tiers as an additional FP filter. This should be sequenced WITH the suspected-tier audit (below) since OLI is one of its key signals.

---

## Correction #3/#4 recurrence check — contained, not regenerating

The suspected-tier mislabel signature (suspected + `detection_method=bytecode_pattern` + bytecode_cache all-flags-zero) stands at **4,596 contracts**. This is the *residue* of the original Correction #3 population (the velocity-escalation pipeline fix in Correction #4 stopped new ones; the historical ones were moved by the migration script but new suspecteds via that path can still appear if the pipeline regressed).

- suspected total: 136,631
- suspected via `deployer_history`: 122,686 (89.8%)
- suspected via `bytecode_pattern`: 13,945 (10.2%)
- of the bytecode_pattern ones, all-flags-zero: 4,596

**This is the single largest unaudited FP-risk pool in the corpus** and motivates the suspected-tier audit (deferred — see below). It is not a *new* regression; it's the known Correction #3/#4 tail that was never fully swept.

---

## Confirmed-tier thin-evidence residue (post-migration)

After all 347 audit migrations, **218 / 1,262 confirmed contracts (17.3%)** still have the triple-negative: no bytecode_cache row AND no approval activity AND deployer has only this one confirmed contract. These are the weakest-evidence survivors of the confirmed tier. They are not necessarily FPs (a real one-off honeypot with a behavioral confirmation can look like this), but they are the natural Phase-C-deep candidates if/when the audit resumes. Logged, not actioned.

---

## What was NOT auditable in the dark window

- **Suspected-tier FP characterization (136K contracts):** the audit-plan Phase A method needs Blockscout enrichment (RPC). Deferred to resume. The 4,596 all-flags-zero + 122,686 deployer-derivative populations are the priority strata.
- **Bug #19b precise correction:** needs transfer-log decoding (RPC). Deferred.
- **OLI re-population:** needs the OLI/Blockscout API (RPC). Deferred.

All three are Phase E / post-resume work, correctly blocked by the intentional RPC pause — not by capability.

---

## Recommended actions on resume (2026-06-01), in order

1. **Fix OLI fetch→write** (priority #22) and backfill `tags_json` for the 13 + expand coverage. Cheap, high-leverage, unblocks the FP gate.
2. **Phase E drain-log decoding** for the 473 multi-victim drain tx_hashes — convert the 7,227 upper-bound into a verified victim count. Re-publish drain headline.
3. **Suspected-tier audit** (the 4,596 all-flags-zero first, then a sample of the 122,686 deployer-derivative) using the same A/B/C method as the confirmed-tier audit.
4. **Add `off_corpus` flag** to watchlist schema so the 22 cross-chain rows stop re-flagging.

---

## Retired-claim propagation sweep — NOT CLEAN. Finding 4 (HIGH).

Checked all 347 audit-migrated addresses against 108 markdown files. **35 of 347** migrated addresses are cited in markdown. The breakdown matters:

- **28 of 35** are cited only in `reports/confirmed_tier_audit_phase_c_sample_review_2026-05-22.md` — that's the audit's own working file listing them as FP candidates. Benign (that's what it's for).
- **7 of 35 are cited elsewhere as live confirmed traps, several with dedicated case files.** These are the real problem and they cut BOTH ways (see Finding 4).

### Finding 4 — Audit/doc contradiction: 7 contracts are BOTH "migrated to unanalyzed (legitimate)" AND "documented as confirmed traps with case files"

| Address | Doc claim | Where |
|---|---|---|
| `0x12577cf0d8a0…` | "Fee-Skimming Token" + dedicated case file | INDEX.md + `CASE_0x12577cf0_base_20260322_040329.md` |
| `0xaeac0e69f6d2…` | "Laser Eagle — confirmed predatory bytecode signature" | INDEX.md + `CASE_HONEYPOT_TOKEN_OPERATOR_0x8ca70232.md` |
| `0xd6cd943bfc07…` | "113 victims drained, Arbitrum, confirmed tier" | INDEX.md |
| `0x44a2ee1369c3…` | "drainer drained 37 victims through tier=confirmed" | INDEX.md |
| `0x955b2c75efff…` | (adjacent to the 37-victim drainer narrative) | INDEX.md |
| `0xa7e1e8ab7b7c…` | "Operator contract #3 (slow-bleed harvester), confirmed-tier" | `CASE_SELF_DEPLOYING_TRAP_OPERATOR_0xACC79E7B_20260521.md` |
| `0x752c5a95d202…` | OFC — already correctly retraction-tagged | INDEX.md + correction_log + case file (all annotated) |

**This is the inverse-error risk and it's the most important finding of the audit.** Each of these 7 was moved confirmed→unanalyzed by the Phase A/B/C automated migrations (verified-source / OZ-framework / high-activity heuristics), yet each has prior hand-written documentation asserting it is a real trap — and three of them (`0xd6cd943bfc`, `0x44a2ee13`, `0xa7e1e8ab7b`) cite **specific drain victim counts**, which is concrete adversarial evidence the automated heuristics are not supposed to override.

**Two possibilities, opposite implications:**
1. **The docs are stale and the audit was right** — these were FPs all along, and the case files / INDEX narratives are themselves part of the FP corpus that Corrections #24/#25 exist to clean up. (Plausible for `0x12577cf0` "fee-skimming token" and `0xaeac0e69f6` "Laser Eagle" — fee-on-transfer tokens are a known FP class.)
2. **The audit over-corrected** — the automated heuristics wrongly migrated genuinely-adversarial contracts that had real drain evidence, introducing a FALSE-NEGATIVE. (Concerning for the three with explicit victim counts.)

**This cannot be resolved without RPC** (need to verify whether `0xd6cd943bfc` actually drained 113 victims on-chain). It is the single most important resume-action: **reconcile the 7 audit-migrated-but-documented-as-trap contracts before trusting either the migration or the case files.** Provisionally, the three with specific victim counts should be treated as **migration-suspect** (possible false negatives from the audit) until verified.

### The real defect (after computing exact numbers): no migration heuristic ever checked the drain table

I twice guessed wrong at the mechanism before computing it (`scripts/audit_migration_drain_gate.py` → `_audit_migration_drain_gate.txt` is the ground truth). The actual finding:

**45 of the 347 migrated contracts (13.0%) had `drain_detected=1` rows in `approval_watchlist` that NO migration heuristic ever looked at.** A contract was downgraded confirmed→unanalyzed for being a verified token / OZ-framework source / high-activity — while simultaneously carrying recorded drain evidence. The heuristics checked Blockscout legitimacy and activity shape but **never gated on "does this contract have drains."**

Distribution of the 45 by migration batch:
- **Phase A (holders/verified): 35** ← the dominant source, NOT FROM_ACTIVITY as I wrongly wrote twice
- Phase C FROM_ACTIVITY: 5
- Phase C FROM_SOURCE: 2
- Phase C sample: 1
- (Phase B, FROM_CLUSTER: 0)

Worst examples (drains / distinct drain-tx):
- `0xa7e1e8ab7b` (FIRE token): 194 drains / 99 tx — case file calls it a slow-bleed harvester
- `0xd6cd943bfc` (Yupp AI): 118 drains / 19 tx — **and bytecode shows SELFDESTRUCT** (deferred_threat_score 3), INDEX says 113 victims
- `0xf768d7d152` (FROM_ACTIVITY): 127 drains / 28 tx
- `0x955b2c75` (Futureverse): 43 drains / 19 tx
- ...41 more

**Critical nuance — these drain counts are themselves Bug #19b-inflated, so the 45 are SUSPECTS not confirmed-errors.** Example: `0xb738b1568f08` shows 1,618 drains from only 2 tx — that's the Bug #19b over-credit pattern, and that contract may genuinely be a legit token whose 2 reverted/odd txs got fanned out. The two bugs interact:
- If a contract's drains are **real** (many distinct tx, each draining few victims) → the migration was a **false negative**, restore to confirmed.
- If a contract's drains are **Bug #19b artifacts** (1-2 tx fanned to hundreds of victims) → the migration was **correct**, the "drain evidence" was never real.

So the 45 split by drain-shape:
- **High distinct-tx (≥10 separate drain transactions): ~27 contracts** — these are the strong false-negative suspects. A legit token does not have 99 separate addresses each running transferFrom-drains against it. `0xa7e1e8ab7b` (99 tx), `0xf768d7d152` (28 tx), `0xa68079da` (25 tx) lead this list.
- **Low distinct-tx (1-2 drain tx, high victim count): ~11 contracts** — these are likely Bug #19b artifacts; migration probably correct. `0xb738b15` (2 tx / 1,618 victims), `0xb0a4741f` (1 tx / 319).

**This is the most important finding.** The confirmed-tier audit (Corrections #24/#25) cleaned a real FP problem but introduced a false-negative problem because its heuristics never had a drain-evidence veto. ~27 contracts with multi-transaction drain histories may have been wrongly exonerated.

**Resume-action #0 (ahead of everything):** for each of the 45 (prioritizing the ~27 high-distinct-tx ones), decode the actual on-chain transfer logs (this also resolves Bug #19b for them simultaneously — same RPC work). Restore confirmed-tier for any whose drains are real. Migrations are lossless (original `confidence_reason` preserved in the annotation), so restoration is a tier flip.

**Pipeline fix:** add a drain-evidence veto to the migration logic — never auto-migrate a contract with ≥N distinct drain transactions, regardless of Blockscout legitimacy. (N≥3 distinct tx, not raw drain rows, to avoid Bug #19b false vetoes.)

**Heuristic exoneration:** `LIKELY_FP_FROM_SOURCE` (130, only 2 with drains) and `LIKELY_FP_FROM_CLUSTER` (7, zero with drains) are largely clean. `LIKELY_FP_FROM_ACTIVITY` (25, 5 with drains = 20%) and **Phase A holders/verified (116, 35 with drains = 30%)** are the implicated batches. Phase A — the cheapest/earliest heuristic — is the worst offender, not the Phase C ones.

## Files

- `scripts/audit_consolidated.py` — sections A–E
- `scripts/audit_refine.py` — Correction #3/#4 + Bug #19b refinement
- `scripts/audit_verify_19b.py` — worst-case row proof
- `scripts/audit_phase0_state.py` — Phase 0 persistence check
- `scripts/audit_hash_formats.py` — hash-format check (retracted the false finding; script kept for the corrected 100%-consistent result)
- Raw outputs: `reports/_audit_*.txt`
