# Adversarial Audit — Where Layer 3 Is a House of Cards

**Date:** 2026-06-06
**Author:** surveillance maintenance (adversarial review, operator-requested)
**Posture:** assume the system is fragile and try to knock it down. Find the load-bearing flaw, not a list of bugs.
**Status:** Tier B (analytical judgment), grounded in the correction log + the 2026-06-06 drain-detector failure.

---

## 0. One-paragraph verdict

Layer 3 has a single load-bearing flaw, not a hundred edge cases: **it infers intent (adversarial / drain / predator) from behavioral *shape*, but at the resolution it observes, legitimate and malicious activity are frequently shape-identical.** Every major retraction in `CORRECTIONS.md` / `correction_log.md` is the same error wearing a different mask. The "infinite edge cases" are not infinite-and-random — they cluster around this one missing thing: a **deductive discriminator** (a Tier-A, on-chain fact that separates the two, e.g. *who initiated the transaction*). The system is a house of cards **specifically in its inferential layer** (classifier tiers, risk scores, "stored potential," predator typologies, headline aggregates). Its **deductive layer** (specific captured txs, addresses, fund flows) is sound. The strategic fix is not to enumerate more edge cases; it is to **refuse to ship any "adversarial" claim without a deductive discriminator validated against the realistic legitimate confound.** That move collapses the infinite edge-case space to a finite, checkable one — proven today: the drain detector went from infinite seller edge cases (`n_out>0`) to ~zero (`tx.from ≠ victim`).

---

## 1. The confession is already written

The strongest evidence against Layer 3 is its own correction history. Read as a hostile auditor:

| Correction | Claim made | Reality | Direction |
|---|---|---|---|
| #20 | OLI-tagged institutions = adversarial operators | legitimate entities | **over-claim (FP)** |
| #21 | Pattern D: long mainnet vintage = predator | signal reversed (predators are *recent*) | wrong-direction |
| #22 | "Camouflage Equilibrium": predators revert *less* | predators revert *more* than baseline | wrong-direction |
| #23 | Routing monitor producing signals | 0 signals ever, dead 22 days | **silent failure** |
| #24 | `0x752c5a95` "Pre-Drain Harvester / 4,587 victims" | verified Animoca token; failed txs | **over-claim (FP)** |
| #25 | 1,650 confirmed adversarial contracts | ≥116 were legit tokens (7.2%) | **over-claim (FP)** |
| #27 | N drain *events* | upper bound; multi-victim over-credit | **over-claim (inflation)** |
| #28 | tx_events join "blind"; 40,144 recovered drains | mostly real story, wrong on regression + count | mixed |
| #29 (today) | 44,540 verified drains | ~98% legitimate DEX sales | **over-claim (FP)** |

Plus the standing retirement list in `claude.md`: 14.2×, org_001's 899 deployers, camouflage 68%, Pattern D 54%, $3.9M, the $3.1 *quadrillion* OP "drain." **The overwhelming majority of retired claims failed in one direction: Layer 3 said adversarial/drain/predator; reality said legitimate.** That is a systematic bias, not bad luck.

---

## 2. The discriminator test (how to read the inventory)

For every claim or surface, ask one question:

> **Is there a deductive (Tier-A, on-chain replicable) signal that separates adversarial from legitimate — or does the claim rest on behavioral/bytecode *shape* that legitimate activity also produces?**

- **DEDUCTIVE** — a ground-truth fact carries the claim (who signed the tx; did value actually move; is the source verified). Survives scrutiny.
- **SHAPE-ONLY** — the claim is an inference from a pattern that legitimate actors also exhibit. This is a *card*. It will leak false positives forever, because the space of legitimate behavior is unbounded.
- **BROKEN** — the surface silently fails (no output, empty data, dead process), so it is neither.

The drain detector is the worked example: `n_out>0` (tokens left the wallet) is **shape** — every seller matches. `tx.from ≠ victim` (a third party moved them) is **deductive** — only a drain matches. Same target, opposite epistemic class.

---

## 3. Inventory of inferential surfaces

| Surface | What it asserts | Class | Evidence | Verdict |
|---|---|---|---|---|
| **`confirmed` tier** (1,556) | "adversarial contract" | SHAPE-ONLY + self-confirming | #25 (≥7.2% FP); promotions via self-loop/behavioral confirmation grade their own homework | **DEMOTE to leads; hand-verify a small core** |
| **`suspected` tier** (71,843) | "likely trap" | SHAPE-ONLY | 136K never audited (op #9/#10); 0.8% base rate ⇒ FP-dominated by construction | **LEAD-only; never a verdict** |
| **Risk score / "stored potential"** | `(stored_potential×volatility)/realized_value`; "absence of harm = danger" | SHAPE-ONLY + **unfalsifiable** | identical to the description of any powerful legit protocol (USDC, a bridge); a claim that can't be wrong can't be right | **RETIRE as a metric, or reframe explicitly Tier-B "capability surface," never "risk"** |
| **Approval-drain counts** | "N victims drained" | was SHAPE-ONLY → **now DEDUCTIVE** | #27 (inflated), #29 (n_out=sales); fixed by tx-initiator gate | **KEEP (small, Tier-A, post-fix)** |
| **Camouflage ratio** | predator revert signature | SHAPE-ONLY | #22 reversed; baseline-vs-predator shape collision | **RETIRED** |
| **Pattern D** | cross-chain reputation import = predator | SHAPE-ONLY | #21 reversed | **RETIRED** |
| **OLI suppression gate** | "deployer is a known institution" | **BROKEN** | `oli_labels` 13 rows, all `tag_count=0`/entity NULL (op #22); suppresses on severity with no tags; coincidentally hides `org_004` | **FIX or disable; do not rely on** |
| **Routing monitor** | 1inch routing anomalies | **BROKEN** | 0 signals corpus-wide, dead 22 days (#23) | **RETIRE or re-provision + watchdog** |
| **Bot candidates** (5,628) | "bot / self-deploying operator" | MIXED | reverts ARE on-chain facts (deductive), but "bot"/"operator intent" is inference | **KEEP fact, tier the label** |
| **Organizations / fund-flow mapping** | org_001…004 boundaries | MIXED | fund flows deductive; org *boundaries* + deployer counts method-dependent (16/26/308/324) | **KEEP flows; always state method on counts** |
| **USD harm attribution** | "$X drained" | **GAP / not built** | `loss_estimate_usd` 0/2,159 populated (op #15); the $3.1q decimals bug | **build deductively or do not quote $** |
| **Headline corpus aggregates** | drain totals, %s, ratios | SHAPE-ONLY denominators | computed over FP-laden tiers, published, then retired — the 14.2× treadmill | **recompute on deductive subsets only; caveat** |

---

## 4. The recurring failure modes (cross-cutting)

1. **Shape = intent.** The root. §2. Every FP correction is an instance.
2. **Self-confirmation.** A "confirmed trap" is confirmed by Layer 3's own heuristics, never by observed harm. No labeled ground-truth set ⇒ the system grades its own homework ⇒ every downstream stat is circular.
3. **Silent failure as default.** Swallowed `except` blocks (the 2026-05-27→06-05 drain bug hid behind one), no per-component health surface (op #17). Routing monitor + OLI pipeline ran "dead" for weeks. The system can be *running while half its organs are dead*.
4. **Aggregation on contaminated denominators.** Headlines built on FP-laden tiers, cited externally, then walked back. Structural, not incidental (the 14.2× principle exists *because* of this).
5. **Base-rate neglect.** At ~0.8% true-adversarial base rate, even a 3% FP rate buries true positives ~4:1. The `suspected` tier cannot be a verdict at this base rate, period.
6. **Unfalsifiable framing.** "Absence of realized value is the danger signal" makes "stored potential" immune to disproof — and therefore worthless as a claim under scrutiny.

---

## 5. What is genuinely solid (the audit must be fair)

- **The deductive layer.** "Address X signed tx Y that moved Z at block N" — replicable by any third party. The drain *forensics* (drainer + tx, post-fix) are Tier-A.
- **The raw corpus.** 376K captured deployments, 21.5M tx events, fund-flow edges — the *observations* are real; only the *labels* on them are suspect.
- **The correction culture.** The single strongest asset. The corrections aren't the failure — they're the immune system. Today's drain FP was caught *because* the verification discipline exists. The failure mode to fear is shipping the inferential layer to someone who **can't** run the correction.

---

## 6. Is it a lost game of infinite edge cases?

**As currently played, yes.** A comprehensive behavioral classifier that carves "malicious" out of unbounded legitimate behavior using features legitimate behavior shares is unwinnable; each correction is one tile on an infinite treadmill.

**Re-scoped, no.** Two coherent products survive scrutiny:

- **(A) Forensics engine.** Stop predicting intent; trace *realized* harm precisely after it occurs (Tier-A). Keep a *small, hand-verified* confirmed set, not 1,556 heuristic ones. A few ground-truth detectors (the tx-initiator drain test). Publish PPV per tier with honest FP bounds (op #1, never done).
- **(B) Leads, not verdicts.** Keep the classifier, but every flag is a *lead* requiring deductive confirmation before it becomes a claim. "71,843 suspected" → "71,843 leads, N confirmed by ground truth."

**Not viable:** continuing to ship the inferential layer as if it were intelligence.

---

## 7. The standard that collapses the edge cases

A single rule, applied everywhere, retires most of §3 at once:

> **No "adversarial / drain / predator / confirmed" claim ships without a deductive discriminator, validated against the realistic legitimate confound — not a trivial negative control.**

Corollaries:
- Every surface carries its tier; **Tier-B = lead, not verdict.**
- Every "negative control" must include the *hard* confound (a seller, an institution, a verified token) — not just the easy negative (an empty address). The drain parity test passing on an inbound-only control while missing every seller is the cautionary tale.
- Per-component health endpoint (op #17) so BROKEN surfaces can't masquerade as quiet.
- Recompute every headline on the deductive subset; retire the rest.

---

## 8. Triage map (actionable)

- **KEEP (deductive):** drain forensics (post-fix), fund-flow edges, captured tx/event facts, revert observations.
- **DEMOTE to lead:** confirmed tier, suspected tier, bot/org *labels*.
- **FIX (broken):** OLI pipeline, routing monitor, per-component health, USD attribution.
- **RETIRE (shape-only as verdict):** "stored potential" as *risk*, camouflage ratio, Pattern D, any headline aggregate over a contaminated denominator.

The drain detector is the template for the whole system: it was a card; today it became a (small) deductive brick. Apply the §7 standard and Layer 3 stops being a house of cards — by becoming a much smaller, much harder-to-knock-down house.
