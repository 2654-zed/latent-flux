# Layer 3 — What We Learned

**Status:** ARCHIVED. Service stopped 2026-06-08. Ran 2026-03-17 → 2026-06-08 (**~83 days / ~2.7 months — not "4 months," not continuous**: there was an intentional 5-day dark window, 2026-05-27 → 06-01, with no backfill — see §7). Corpus retained read-only.
**Cost:** **~$5K Alchemy compute units (operator estimate, not a billed figure)** on a shared Alchemy app co-hosted with a separate trading experiment, plus ~3 months of operator + analysis/correction time (a real, uncounted cost).
**This document:** the honest record of what the project tested, found, and is worth keeping. It was itself adversarially reviewed before archiving (3 independent lenses; this is the revised version) — because a retrospective about overclaiming that flatters its own recovery would be the final instance of the bug. All load-bearing numbers below were re-verified against the database and the correction log; the framing was corrected where it had drifted into self-exoneration.

---

## 1. The question

> Are people *blatantly deploying smart contracts that steal from users*, on-chain, at rampant scale — and can we detect them by how they look and behave on-chain?

A real-time behavioral surveillance system on Base, Arbitrum, and Optimism, built to answer yes and map the operators.

## 2. The answer

**No.**

- The flagged population is **overwhelmingly legitimate.** This is not one clean "99% PPV" measurement — it's the consistent direction of every audit: the drain method was ~98–99% legitimate DEX sales (#29), the confirmed tier had ≥7% verified-legit downgrades (#25), and the **suspected tier — the corpus's largest population (~115K–136K contracts) — measured ~0.02% observable-harm and was largely never audited** (#9). The "~282K contracts monitored" headline is mostly an unscreened lead pool, not signal.
- The **real adversarial signal is small and specific**: **266 tx-initiator-verified approval-drains** across **259 victims, 59 contracts, 104 drainers** — from an **82% scan** (50,351 of 61,445 approvers; the 18% tail stalled on rate limits and was never completed). "~325 at full scan" is an *extrapolation* of the 0.53% rate onto the unscanned tail, assuming it drains at the same rate — an assumption, not a count.
- Detection-by-on-chain-shape is **structurally broken**: at ~0.8% base rate, legitimate and malicious activity are shape-identical at our resolution, so false positives buried the signal.

The fear wasn't baseless — drainers are real — but the **mechanism differs from the model**: theft is staged *off-chain* (phishing a victim into approving), then settled on-chain. An on-chain *deployment* monitor was watching the wrong layer.

## 3. Real vs. artifact (numbers re-verified against the DB)

| Claimed | Actual | What it was |
|---|---|---|
| 3,437 lifetime drains → briefly **44,540** | **266 confirmed** (259 victims / 104 drainers / 59 contracts) | TWO defects, not one: (a) Bug #19b credited *every* approver of a contract on a single interaction, so every historical count was an upper bound (#27, pipeline fix deferred); (b) the replacement `n_out>0` method counted DEX *sales* as drains, ~98% false (#29) |
| org_001 = **$285M** criminal org | **~$257K observed (itself unverified)**; ~1 real contract | A `SUM ... LIKE 'EXTRACTION_00%'` swept in a DPRK Solana hack ($285M), a NEAR exploit ($18.4M), and Wasabi ($5M) — none org_001's (#30) |
| 5 orgs + "criminal infrastructure" | **1 drained contract, 0 draining deployers** across the whole layer | Separately-claimed inventories that *each* collapsed to zero: 53 role-wallets (0 drains), 7 "rogue facilitators" (0; only D270 ~$5K real), 437 x402 facilitators (legit Coinbase infra), 22 "known_attacker"/trap entities (0), and the `bytecode_families` table is **empty** — "Coffee fleet"/"T2 family" are doc-only (#31) |
| 1,650 "confirmed" adversarial | ≥7% verified-legit; core includes **Chainlink oracles** | Shape-based classifier FPs (#25) |
| **$3.9M x402 drain operation / 1,955 victims** | ~$2.3M, then effectively dissolved | The "single worst-hit victim" (`0x785ce546`) was a *controlled intermediary distributing $9.8M for the operators* — an operator mislabeled as a victim (#11–#16) |
| Camouflage / Pattern-D predator signatures | direction **reversed** | Inferences from shape that legit activity shares (#21, #22) |

The biggest "extraction" in the database — **$285M, Drift Protocol** — was a **Solana governance takeover attributed to DPRK**, on a chain we never monitored, imported from a public post-mortem. Never ours to claim.

## 4. Why it failed (one epistemic root cause + a cluster of plumbing bugs)

**The epistemic root cause:** the system inferred *intent* from *shape*, but adversarial-ness is a property of *control*, not appearance. The one detector that worked stopped looking at shape and tested a control fact: was the outbound transfer initiated by someone *other than the owner* (`tx.from ≠ victim`)? That held; everything shape-based did not.

**But it wasn't only epistemics — several independent plumbing/accounting bugs inflated the numbers:** a SQL `SUM`/`LIKE` aggregation that summed unrelated incidents (#30), a token-decimals normalization bug (the "$3.1 quadrillion" figure, off by 10¹²), reverted `transferFrom`s credited as drains (#19/#24), and silent process death with no respawn (#23). These are separately-actionable, not all "the same error."

There is a bitter vindication: the project's own thesis — *harm emerges from correctly-executing components* — is **right about the nature of harm and wrong about its detectability.** Drift proves it: every component executed as designed; the harm was illegitimate *control* obtained off-chain. If components execute correctly, there is no adversarial shape to flag in advance.

## 5. What real adversarial behavior looks like

A spectrum, ordered by *where control is subverted* — and on-chain visibility runs **inversely** to damage:

1. **Approval drains** (small, most legible): off-chain phishing → approve → third-party `transferFrom`. We caught the diffuse tail; the industrial drainer-kits run off our surface.
2. **Oracle / governance manipulation** (large): seed a fake price, phish a quorum. Looks like legitimate trading + admin actions.
3. **Key / admin compromise** (largest, least legible): contracts are perfect; a signer was compromised off-chain. On-chain it's one normal transaction.

Common thread: **the adversarial act is illegitimate acquisition or exercise of control, and control is granted or stolen mostly off-chain.**

## 6. On-chain vs. off-chain — the hard ceiling

The chain is the **crime scene, not the crime.** It records deductively: **effects** (what moved, what drained) and **control events** (who *signed*). It does **not** record: **consent/intent** (was the approval knowing?), the **compromise vector** (phishing/malware/insider), or **identity/attribution** (Drift's DPRK call came from off-chain intelligence, not the chain). **On-chain data is necessary but radically insufficient for adversarial intelligence** — real attribution requires off-chain fusion this system never had. "Continue and the sophisticated threats will appear" was the trap: unfalsifiable, which is exactly what made it dangerous.

## 7. Known limitations — what we did NOT close

A retrospective that only lists wins is the same overclaiming in reverse. Open items at archive time:

- **Dark window (#26):** the service was *intentionally paused* 2026-05-27 → 06-01 (5 days, all chains + detectors + stats dark, no backfill) because the shared Alchemy app neared its CU cap and budget was diverted to the co-hosted trading experiment. "Continuous" is false; uptime and resourcing were shared.
- **The institutional-FP safeguard never worked.** After #20 (Binance/Circle/Coinbase/LI.FI-class addresses labeled adversarial), the fix was an OLI public-label gate. `oli_labels` is empty (13 rows, all `tag_count=0`) — so suppression of institutional false positives is *coincidental, not enforced*.
- **An advertised detector was vaporware:** the 1inch routing-anomaly pathway produced **zero** operational signals corpus-wide; the API key was never provisioned, and the silence went undetected for 22 days after its last heartbeat (#23).
- **The longest detection gap:** a live trap (`T1-2081a9d32218`) sat at "suspected" for 21 days accumulating **928 victims** / 2,147+ txns with zero alerts.
- **The drain ground-truth is 82% complete** (18% tail never scanned); the per-victim verification only became possible at the very end.
- **The pipeline was paused, not repaired.** The FP-generating promotion path (the Phase-E Blockscout-verified + OLI + holders gates) was *never built*; the prod drain detector was left **paused**, not fixed. Anyone reviving the corpus inherits the unfixed classifier.

## 8. What the period actually produced

Not threat intelligence — **a negative result and, eventually, a verification discipline.** Honest framing: the rigor was *intermittent, operator-prompted, and late* — the largest corrections (#28–#31) all landed in the final week, and several were triggered by the operator demanding ground-truth audits, not by the system flagging itself.

On "did we avoid shipping fakes?" — **partially, and that's the honest answer.** The two biggest fabrications (the 44,540 drains, the $285M org) were caught before reaching customers — though the 44,540 figure *was committed to `main`* first and required a follow-on retraction (#28→#29), so the catch was post-commit, not pre-commit. **Earlier retired claims did ship into external materials:** the 14.2× trust-amplification figure and the camouflage predator framing reached pitch/narrative decks and are still flagged "cleanup pending" in the CORRECTIONS propagation watch-list. The discipline arrived mid-flight, not at the start.

**Retained (real, Tier-A):** the 266-drain / 104-drainer forensic set (on-chain-attributed); the raw captured deployments/events (observations sound, labels suspect); `CORRECTIONS.md` + `correction_log.md` (#11–#31); the tx-initiator drain detector (0-CU).
**Archived/paused, not repaired:** the behavioral classifier and risk scoring (shape-based verdicts); "stored potential" as *prediction* (it is a capability surface, never measured intent); org mapping and "$X extracted" headlines.

## 9. Transferable lessons

1. **Shape ≠ intent.** No "adversarial/drain/predator/org" claim without a *deductive discriminator* — a ground-truth control fact.
2. **Validate against the realistic confound, not the trivial negative.** Our drain test passed a no-activity control and missed every seller.
3. **Respect the base rate.** At 0.8% true positives a 3% FP rate buries the signal 4:1; a classifier with no published PPV is a lead generator, not a verdict engine.
4. **No harm number without ground truth + an attribution chain.** Never a `SUM` over an incident catalog; never an external incident on an unmonitored chain.
5. **Make silent failure loud.** Two of our worst gaps — a detector that never functioned (22 days unnoticed) and a live trap that drained 928 victims over 21 days at "suspected" — were *absences of an alert*, not wrong alerts.
6. **The correction log is the asset.** A system that can prove what it got wrong, and when, is worth more than one that only claims what it got right — and it only counts if the corrections actually propagate to the decks, not just the log.

---

*The instruments overclaimed for months — fake drains, a fake $285M org, a "victim" who was an operator, Chainlink oracles labeled as traps. What eventually told the truth was the correction process, forced by ground-truth audits, against the system's own output. That process is the part worth keeping. It arrived late, and that lateness is part of the lesson.*
