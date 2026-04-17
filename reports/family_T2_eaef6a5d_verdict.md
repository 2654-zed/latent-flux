# Family T2-eaef6a5d7678 — Verdict
## Classification Audit of the Largest Bytecode Family

**Family ID:** `T2-eaef6a5d7678`
**Family name:** `Tier2-fee=0|asym=0|crev=0`
**Clustering key:** `fee=0|asym=0|crev=0` (hash verified)
**Size:** 21,936 contracts, 8,240 deployers
**Share of corpus:** 21,936 / 73,106 = 30.0%

---

## Verdict: **MIXED / UNKNOWN — this is the NULL bucket**

This is NOT a family in the template sense. It is the catch-all for contracts where the Tier 1 clustering produced no `bytecode_pattern_notes` AND all three flag detectors (`has_unusual_fee_structure`, `has_asymmetric_transfer`, `has_conditional_revert`) returned zero. Contracts are grouped by the absence of signal, not by any shared pattern.

---

## Evidence

### Rules out LEGITIMATE_INFRASTRUCTURE (single template)

| Check | Result |
|-------|--------|
| Unique code_hashes in family | **21,312** across 21,312 cache entries |
| Largest single-hash cluster | **1 contract** |
| Unique deployers | **8,240** |

A real factory template (Uniswap V2 Pair, SafeProxy, standard launchpad) would show one or a few code_hashes repeated thousands of times. Here every contract has a distinct bytecode hash. This is not a template.

### Rules out MALICIOUS_TEMPLATE (single operator)

8,240 distinct deployers with the top one (`0x2e20b26172...` — the Dragon) contributing 2,077 contracts (9.5%). No single operator's template could span 8,240 deployers. The Dragon's fleet is a subset, not the whole family.

### Rules out DORMANT (as a pure label)

21,033 of 21,936 contracts have zero callers (95.9% dormant). But **903 are active**, including **63 behaviorally confirmed traps** with 277 trap events and 237 coffee-fleet victims. Some of these are definitely weaponized. Calling the whole family dormant would bury real threats.

### The signature that proves it's a NULL bucket

Bytecode cache confirms the classifier ran successfully on each contract and found nothing:

```
Bytecode analyzed (2223B) — checked [asymmetric_transfer, blacklist_check,
  tx_origin_conditional, callback_trap, hidden_fee, selfdestruct,
  delegatecall_in_token, timestamp_activation, origin_eoa_gate, obfuscated_fee]
signals: {has_asymmetric_transfer: false, has_conditional_revert: false,
  has_unusual_fee_structure: false, pattern_notes: null}
```

Every contract in this family was examined by all 10 pattern detectors and none fired. The "family" is a reporting artifact — it captures what the classifier couldn't label.

### Only 4 exceptions

Exactly 4 of 21,936 have non-null `bytecode_pattern_notes`. Three of them contain `CALLER -> EQ/SLOAD -> JUMPI -> REVERT` patterns (blacklist / conditional revert) and one has `SELFDESTRUCT + DELEGATECALL`. These are clustering edge cases — their detectors fired but didn't set the three tracked flag columns, so they ended up in the `0|0|0` bucket instead of Tier 1.

---

## Composition Breakdown

| Segment | Count | Share |
|---------|-------|-------|
| Dormant (0 callers) | 21,033 | 95.9% |
| Active, unconfirmed | 840 | 3.8% |
| Confirmed traps | 63 | 0.29% |
| Chain = base | 17,406 | 79.4% |
| Chain = arbitrum | 4,530 | 20.6% |
| Confidence = unknown | 13,025 | 59.4% |
| Confidence = suspected | 8,848 | 40.4% |
| Confidence = confirmed | 63 | 0.29% |
| Method = bytecode_pattern | 15,170 | 69.2% |
| Method = deployer_history | 6,766 | 30.8% |

The confirmed traps inside the family (63) attract mostly the coffee fleet — 237 of 277 trap events (85.6%) are coffee-fleet victims. So a real malicious subset exists, but it's <0.3% of the family.

---

## Top Deployers in the Family

| Deployer | Contracts | Notes |
|----------|-----------|-------|
| `0x2e20b26172...` | 2,077 | **The Dragon** — documented pre-staged liquidity-trap inventory |
| `0xa18478d133...` | 518 | |
| `0xbdb6985c0e...` | 296 | |
| `0x1f39f9a601...` | 245 | |
| `0x19393c44ab...` | 208 | |
| `0xb84959cb6a...` | 189 | |
| `0xeca762ff39...` | 153 | velocity-flagged deployer (seen in samples) |
| `0x8b503c5604...` | 131 | |
| `0xd660fa35cd...` | 123 | also appears in dormant-activation alerts |
| `0xe2f215bbd5...` | 114 | |

The Dragon alone contributes ~10% of the family. These 10 deployers account for 4,074 contracts (18.6%).

---

## Implications for the "46% suspected" Statistic

The suspected tier includes **8,848 contracts from this family**, all of which have:
- Zero bytecode trap signatures (`at=0, cr=0, uf=0`)
- No pattern notes

Of those 8,848:
- **6,729** are `suspected + deployer_history` — flagged because the deployer also deployed a confirmed trap, NOT because the contract itself has signals
- **2,119** are `suspected + bytecode_pattern` but with no detected patterns — these appear misclassified; the classifier produced no signal yet they got a suspected label

**Correction:** If the suspected tier is meant to signal "bytecode indicates trap-like structure," this family should not be contributing 8,848 entries. The 6,729 deployer_history contracts are derivative flags — their suspicion is inherited, not earned by the bytecode. The 2,119 bytecode_pattern contracts with no signals look like a labeling bug and warrant a separate audit.

A corrected suspected count that requires at least one bytecode signature or pattern note would drop by roughly these 2,119 contracts and would make the "46%" headline tighter and more defensible.

---

## What Additional Data Would Resolve Remaining Uncertainty

I could not do deeper bytecode inspection (opcode walks, selector extraction) because raw bytecode is not stored — only a 64-char `code_hash` and the classifier's signal output. To fully classify the 903 active non-confirmed contracts in this family, we would need:

1. **Raw bytecode via RPC** (`eth_getCode`) for the 903 active contracts — ~903 RPC calls, feasible as a one-time batch
2. **Selector extraction** from those bytecodes to identify interface (ERC-20 standard vs custom vs proxy)
3. **Top 7 unknown high-volume selectors** decoded (`1cff79cd`, `4fd11b2f`, `d2154138`, `a47bd76d`, `f2005da3`, `a2e62045`, `30cf6ea2`) — these account for ~14,000 calls across a handful of contracts; identifying them would reclassify those contracts

Without RPC, we cannot distinguish within the 903 active contracts between legitimate custom protocols and weaponized contracts that simply use mechanics none of our 10 detectors cover.

---

## Operational Recommendation

Treat this family as a reporting artifact, not a signal. Two concrete changes:

1. **Rename / flag** the family in `bytecode_families` as `Tier2-NULL-bucket` with a note explaining it contains contracts where no pattern fired. This prevents the family size (21,936) from being cited as a single threat cluster.

2. **Audit the 2,119** contracts tagged `suspected + bytecode_pattern` with `fee=0|asym=0|crev=0` and no notes. These should be `unknown` tier unless we can show what evidence upgraded them. If the upgrade rule can't be explained, downgrade them.

The 63 confirmed traps inside this family are real and already properly labeled. They don't need reclassification — their confirmation came from behavioral evidence (coffee-fleet revert events), not bytecode patterns.
