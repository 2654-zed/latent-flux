# Investigation: Address `0x80b12bd0` and the Animoca Brands Attribution

**Date:** 2026-05-15
**Analyst:** Layer 3 (Claude session 2026-05-15-recent-drains)
**Tracker:** [UNK-031](../memory/UNKNOWNS.md#unk-031--is-0x80b12bd0-actually-animoca-or-is-the-oli-tag-a-false-positive)
**Status:** OPEN — recommend external verification + notification to Animoca Brands / REVV Motorsport
**Epistemic tier:** A for the observed behavior; B for the prevailing key-compromise interpretation; A for the OLI tag's existence

---

## TL;DR

Layer 3's drain-wave analysis surfaced address `0x80b12bd0f1793bf6cea767fa83eb2068eaa17dc8` as the operator behind a **4,587-victim drain event on Base chain at 2026-05-09 11:28–11:58 UTC** — the largest single-day drain in corpus history. The address carries a Blockscout OLI tag attributing it to Animoca Brands (specifically REVV Motorsport), and on-chain holdings corroborate the attribution: it currently holds REVV, OneFootball Club, ANIMOCA, and MATIC tokens.

**Three explanations are consistent with the data:**

1. **Key compromise (most likely; v3 Attack 11a at brand scale).** The address was a legitimate Animoca/REVV deployer wallet whose private key was compromised. The attacker is exploiting the institutional cover identity (7-year mainnet vintage, Animoca-portfolio token holdings) to harvest victim approvals on Base under a trust signal that would not exist for a fresh address.

2. **Rogue former-employee / detached operator.** Someone with legitimate historical access to the wallet (former Animoca developer, ex-REVV contractor) retained the key, separated from the company, and now runs unauthorized operations.

3. **Stale or incorrect OLI tag.** The Blockscout `animoca-deployer` tag may have been added in 2019–2020 when this address deployed a small REVV-related contract, but the address has since been repurposed for scam operations by an unrelated owner. The institutional signal would then be artifactual.

The first two scenarios are operationally identical from Layer 3's perspective: an address that the public-facing OLI registry says is Animoca-controlled is **actively running adversarial operations**. Notification to Animoca Brands is warranted regardless of which of the three is correct.

This document records the evidence chain so that any reader (Animoca security, an independent investigator, Layer 3 in a future session) can re-derive the conclusion or correct it.

---

## 1. Evidence: Blockscout OLI attribution

### 1.1 The OLI tag itself

Source: Blockscout Open Labels Initiative metadata service at `https://metadata.services.blockscout.com/api/v1/metadata`, queried 2026-05-15.

```json
{
  "addresses": {
    "0x80b12BD0F1793BF6CEa767Fa83Eb2068eaa17DC8": {
      "tags": [
        {
          "slug": "animoca-deployer",
          "name": "Animoca: Deployer",
          "tagType": "name",
          "ordinal": 10,
          "meta": "{\"main_entity\":\"Animoca\",\"tooltipDescription\":\"Deployer address for Animoca.\",\"tooltipUrl\":\"https://www.revvmotorsport.com/\"}"
        },
        {
          "slug": "contract-deployer",
          "name": "Contract Deployer",
          "tagType": "generic",
          "ordinal": 0,
          "meta": "{}"
        },
        {
          "slug": "animoca",
          "name": "Animoca",
          "tagType": "protocol",
          "ordinal": 0,
          "meta": "{}"
        }
      ]
    }
  }
}
```

The tag is **curated, not heuristic**. Key markers:

- `tagType: "name"` (a specific name-class tag, the highest-quality OLI tag type)
- `ordinal: 10` (positional weighting within the registry — name-class tags with ordinal ≥ 10 are typically curator-prioritized)
- `tooltipUrl: "https://www.revvmotorsport.com/"` (an explicit link to a known Animoca Brands subsidiary — REVV Motorsport was acquired by Animoca in 2018)
- `main_entity: "Animoca"` in the meta

### 1.2 Blockscout's main address API does NOT corroborate

Cross-check at `https://eth.blockscout.com/api/v2/addresses/0x80b12bd0...`:

```json
{
  "name": null,
  "public_tags": [],
  "private_tags": [],
  "metadata": null,
  "watchlist_names": [],
  "reputation": "ok",
  "is_scam": false,
  "is_contract": false,
  "ens_domain_name": null
}
```

Blockscout's own curated tags (the `public_tags` field on the main address page) are **empty**. The Animoca attribution exists ONLY in the OLI metadata service product, which aggregates third-party-submitted labels from sources like Etherscan tag database, community submissions, and partner registries.

This is a **single-source attribution**. It's not corroborated by Blockscout's own native curation. That fact alone doesn't disprove the tag — many real entity attributions live in OLI before being adopted into the main page — but it means the attribution rests on a single registry's accuracy.

### 1.3 Token-holding corroboration

Query: `https://eth.blockscout.com/api/v2/addresses/0x80b12bd0.../token-transfers`, 50 most-recent transfers.

**Token-symbol distribution among the address's ERC-20 transfers:**

| Symbol | Count | Name | Animoca-portfolio? |
|---|---|---|---|
| **REVV** | 36 | REVV | ✅ — Animoca Brands' racing-game token |
| **OFC** | 6 | OneFootball Club | ✅ — Animoca Brands strategic partner |
| **ANIMOCA** | 2 | Animoca Brands | ✅ — Animoca Brands' own token |
| MATIC | 2 | Polygon | neutral |
| HEX | 1 | HEX | neutral (likely scam-airdrop dust) |
| MALLY | 1 | Malamute.Finance | neutral (meme/scam) |
| WETH | 1 | WETH | neutral |
| "Airdrop ethapy.io" | 1 | (clear scam-airdrop) | neutral |

**Three different Animoca-portfolio tokens** (REVV + OFC + ANIMOCA) accumulated in the wallet. A random scam operator would not naturally accumulate REVV-specific, OFC-specific, AND ANIMOCA-specific tokens — these are illiquid game / brand assets, not the meme-token noise that drifts into adversarial wallets via spam airdrops. **This corroborates the institutional attribution** to at least the level of "the address has been used historically as an Animoca-portfolio wallet."

---

## 2. Evidence: contradictory adversarial behavior

### 2.1 The Base-chain drain operation

A separate file [scripts/investigate_drain_operator.py](../scripts/investigate_drain_operator.py) characterizes the drain. Headlines:

| Fact | Value |
|---|---|
| Bait contract | `0x752c5a95d202972e124390f30a50154409d3c858` on Base, confirmed-tier |
| Deployed by | `0x80b12bd0` on **2026-03-26 03:12:53 UTC** |
| Approvals harvested | **8,007 total** (first 2026-04-11, last 2026-05-15) |
| Drain discharge | **2026-05-09 11:28:23–11:58:01 UTC** (30-minute window) |
| Victims drained | **4,587** |
| Drain throughput | ~150 victims/min = ~0.4s per tx (heavily automated) |
| Drain executor cells | Two — `0x1d81aff2` (3,228 victims) + `0x0e222468` (1,359 victims) |
| Cell victim overlap | **Zero** — pre-partitioned victim sets, indicating coordinated execution sharding |
| Post-discharge state | Approvals continue arriving (400 on May-10, 192 on May-11, ~26/day through May-15); bait remains live |

**This is a textbook Pattern A (Permission Harvesting + Routing Parasite) discharge from POTENTIAL_ATTACKS_V3.md.** The operational pattern matches:
- Storing potential: 6 weeks of approval accumulation under an institutionally-trusted cover identity
- Seeding: 8,007 victims granted ERC-20 approvals to the contract
- Discharge: 30-minute parallelized drain via two coordinated execution cells

### 2.2 Layer 3's pre-drain assessment was correct

The operator was already on the watchlist as **HIGH priority "pristine-reputation solo operator"** before the discharge:

```
2026-04-24 investigator review: deployer of 0x752c5a95 (confirmed pre-drain,
1898 approvals, 0 drains). 2019-05-23 mainnet vintage provides laundering
cover. Two-contract Base footprint defeats fleet detectors.
```

So Layer 3's behavioral classifier correctly identified the discharge risk on 2026-04-24, **15 days before** the actual drain. This is one of the few cases in the corpus where the framework anticipated a major event with significant lead time. The OLI tag was applied 15 days later (2026-05-09) — same day as the discharge.

### 2.3 The cross-chain choreography

The operator's behavior at the moment of trap-deploy is highly informative.

| Time (UTC) | Chain | Action |
|---|---|---|
| 2026-03-26 03:12:53 | Base | Deploy bait contract `0x752c5a95` (the trap) |
| 2026-03-26 03:14:27 | Base | Deploy second contract `0xda42fe39` (currently unanalyzed) |
| 2026-03-26 03:56:59 | Ethereum | Receive 1 OFC token from `0x9eFCf2B5F9c0fe09A2Eef9b969900071621E6fd8` |
| 2026-03-26 04:02:11 | Ethereum | LayerZero `send()` of 1 OFC to Base — destination EID 30184, destination address `0x80b12bd0` (self), refund `0x80b12bd0` (self). Fee 147,691,982,060,816 wei (~0.0001 ETH). |
| 2026-03-26 04:06:11 | Ethereum | Receive 1 OFC from `0xeA614F1d42af662199b0aaC8E70ddAbb7863b793` |

The operator deployed the trap on Base first, then **bridged 1 OneFootball Club token from Ethereum to the same address on Base** ~50 minutes later. The choreography suggests:
- The Ethereum-side institutional identity is being used as **cross-chain cover** for the Base-side adversarial deployment.
- The "self-bridge of 1 OFC" is the kind of low-value test transaction someone would run to validate that the Base-side address can receive tokens — a precursor to running larger operations.

For a key-compromise interpretation, this fits: the attacker confirms cross-chain control of the compromised address before committing to the larger operation. For a "real Animoca operation" interpretation, this is harder to explain — Animoca wouldn't deploy an approval-bait masquerading as a REVV / OneFootball contract on Base.

### 2.4 The Apr-27 mainnet deploy burst

Between 2026-04-27 06:37:35 and 2026-04-27 06:48:35 UTC (~11 minutes), the address sent **50 outbound transactions to the Deterministic Deployment Proxy** (`0x4e59b44847b379578588920cA78FbF26c0B4956C`, the public "Foundry CREATE2 deployer" used for vanity-address generation).

| Cluster start | 2026-04-27 06:37:35 UTC |
| Cluster end | 2026-04-27 06:48:35 UTC |
| Count | 50 transactions |
| Pacing | ~12 seconds between consecutive txs (automated) |
| Destination | `0x4e59b44...` (public CREATE2 deployer) |
| Value | 0 ETH each |
| Method | `0x00000000` (raw bytecode in input data) |

This is the signature of a **scripted vanity-deployment burst**. Legitimate Animoca-corporate deployments would route through company factories or multisigs — not the open CREATE2 deployer. The 50-contract burst is more characteristic of:
- An attacker prepositioning 50 vanity-prefix contracts for future operations
- An automated trap-template factory run
- (Possible benign explanation) A REVV NFT factory using CREATE2 for predictable token IDs — but this would emit specific event signatures and produce verified contracts, which is checkable

The 50 deployed contract addresses themselves can be enumerated from the receipts and checked for verified-source status. That follow-up is not done in this report (estimated effort: ~1 hour for receipt enumeration + bytecode classification on a sample).

---

## 3. Synthesis: which scenario fits the data?

### Hypothesis 1 — Key compromise (Animoca's key stolen)

Predicts:
- Address holds institutional fingerprint (✅ REVV, OFC, ANIMOCA holdings)
- Address has dormant-then-active timeline (✅ 1,201 mainnet txs over 7 years, but only ~75 token transfers — very low activity for a corporate deployer; consistent with dormant primary use)
- Recent activity is qualitatively different from historical (✅ Mar-26 Base trap deploy + Apr-27 mainnet CREATE2 burst + May-9 4,587-victim drain all in the last 7 weeks)
- Cross-chain coordination (✅ Ethereum-to-Base self-bridge before Base trap goes live)
- Drain pattern matches v3 Attack 11a (key compromise → admin authority → drain pooled value) — partially fit (the trap is an approval-bait, not a UUPS-upgradable adapter, so it's Attack 1 not 11a in mechanism, but the institutional-identity reuse is 11a-flavored)

**Status:** consistent with all observations.

### Hypothesis 2 — Rogue former employee

Predicts the same observations as Hypothesis 1 from outside, but with different causality. Distinguishing evidence would require non-on-chain information (Animoca's HR records, the address's known historical signers).

**Status:** consistent with all observations; not distinguishable from key compromise on-chain.

### Hypothesis 3 — OLI tag is stale / incorrect

Predicts:
- The OLI tag could be wrong despite specificity (~unlikely given tooltipUrl points specifically at revvmotorsport.com)
- The token holdings could be coincidence — but accumulating REVV + OFC + ANIMOCA across three separate Animoca-portfolio brands without being Animoca-affiliated is implausible
- The 7-year mainnet vintage and the very-low corporate-deployer-style activity profile is just coincidence — possible but the Bayesian weight is low

**Status:** weakly supported. The OLI tag could be wrong, but multi-portfolio-token corroboration is hard to explain away.

### Conclusion

**The Animoca-attribution OLI tag is most likely correct.** The address is institutionally connected to Animoca Brands / REVV Motorsport. **And** the address is currently running adversarial operations. Hypotheses 1 and 2 are operationally indistinguishable and operationally identical from Layer 3's perspective: the address should be treated as compromised pending external verification from Animoca Brands.

---

## 4. Implication for Layer 3 methodology

### 4.1 OLI guardrail calibration

The OLI guardrail (Layer 3 invariant INV-007) only redirects classifications when the OLI severity is **HIGH**. This address's OLI tag was assigned **LOW** severity by the audit script's heuristic (`scripts/blockscout_tag_audit.py`), because "animoca-deployer" is not in the institutional-brand watch-list the script uses. As a result, the guardrail did **not** redirect 0x80b12bd0's classification, and the address remained on the watchlist at HIGH priority. **This was correct** — Layer 3's behavioral classifier identified the discharge risk 15 days before the drain, and the LOW-severity OLI tag did not erase that finding.

**This case validates INV-007's design.** A HIGH-severity guardrail would have removed an adversarial classification that turned out to be correct. The LOW-severity tagging properly recorded the institutional signal without forcing a re-classification.

### 4.2 Audit-script heuristic gap

The audit script's `severity()` function (in `scripts/blockscout_tag_audit.py:104`) maintains a whitelist of institutional-brand slugs that auto-promote to HIGH severity:

```python
if any(k in blob for k in [
    "circle", "coinbase", "binance", "kraken", "okx", "bybit", "gate", "kucoin",
    "uniswap", "aave", "compound", "lido", "maker", "curve", "balancer",
    ...
    "ens", "eas", "safe", "argent", "rabby",
    "centre", "tether", "paypal", "pyusd",
    "infrastructure", "official", "cex", "exchange",
]):
    return "HIGH"
```

"animoca" is not in this list. If it were, this address would have been auto-promoted to HIGH severity, and INV-007 would have redirected it to `COMMERCIAL/institutional_oli_tagged`. That would have been **wrong** in this case — the address is misbehaving regardless of its institutional connection.

**Lesson:** the audit script's institutional whitelist is not the right gate for "the OLI guardrail should apply." A better gate is the OLI tagType (`name` tags are higher-confidence) plus tag-specificity heuristics (tooltipUrl pointing at a known organization's domain). The Animoca tag has `tagType: name` and `tooltipUrl: revvmotorsport.com` — by tag-quality criteria it's a high-confidence institutional attribution, even though "animoca" isn't on the brand whitelist.

The current architecture handled this case correctly by accident: severity-LOW kept the classification, which was right. But the underlying logic of the audit script's heuristic is fragile and could fail in the inverse direction (e.g., assigning HIGH to a low-confidence tag for a brand on the whitelist).

### 4.3 Drain-cell coverage gap (still real)

The two drain cells `0x1d81aff2` and `0x0e222468` that executed the discharge are NOT on the watchlist (UNK-032). The watchlist currently flags **operators** (deployers) well but misses **execution cells** (drain_callers). The Animoca case illustrates the gap: Layer 3 flagged the operator 15 days early, but the discharge was executed by addresses Layer 3 had never seen.

---

## 5. Action items

| # | Item | Priority | Owner |
|---|---|---|---|
| 1 | **Notify Animoca Brands security team** that an OLI-attributed Animoca deployer address (`0x80b12bd0...`) is implicated in a 4,587-victim drain. Provide this report + on-chain evidence. | HIGH | external (Animoca / public channel) |
| 2 | Verify whether `0x80b12bd0` is on Animoca Brands' published / authoritative deployer list. If not, retract the OLI tag via the Open Labels Initiative correction mechanism. | HIGH | Layer 3 / independent investigator |
| 3 | Enumerate the 50 contracts deployed by 0x80b12bd0 in the Apr-27 06:37–06:48 CREATE2 burst. Classify them via the Layer 3 bytecode classifier. Surface any that match trap-template signatures. | MEDIUM | Layer 3 (next-session work) |
| 4 | Investigate whether the second Base contract deployed by 0x80b12bd0 (`0xda42fe39`, currently unanalyzed-tier) is also an approval bait staged for future discharge. | MEDIUM | Layer 3 |
| 5 | Add the two drain cells `0x1d81aff2` and `0x0e222468` to the Layer 3 watchlist as HIGH-priority `drain_executor_*` entries. Add the other 32 May-9..15 drain-caller addresses surfaced in this session's broader analysis (UNK-032). | HIGH | Layer 3 |
| 6 | Audit-script enhancement: extend the institutional-brand whitelist OR replace it with a tag-quality heuristic (tagType + tooltipUrl pattern). Animoca / REVV / Yuga / Sandbox / OpenSea / Magic Eden are notable omissions. | MEDIUM | Layer 3 |

---

## 6. Open questions tracked

- **UNK-031** (this report) — Is `0x80b12bd0` actually Animoca, or is the OLI tag a false positive?
  - This report updates the analysis: **OLI tag is most likely correct; the address is most likely compromised or detached from Animoca control**. Final resolution depends on external verification.
- **UNK-032** — Should 34 May-9..15 drain-caller addresses be added to watchlist en masse? (related; this report supports YES for the 2 cells executing this drain)
- **UNK-034** — Why is 0x752c5a95 still receiving approvals after its May-9 drain discharge? (related; potential follow-up discharge)
- **Not yet tracked** — What did the 50 Apr-27 mainnet CREATE2 deploys produce? (worth a dedicated UNK if this report is followed up)

---

## 7. Source materials

- Local DB query scripts:
  - [scripts/investigate_drain_operator.py](../scripts/investigate_drain_operator.py) — full operator footprint
  - [scripts/investigate_may9_drainers.py](../scripts/investigate_may9_drainers.py) — drain cell analysis
  - [scripts/db_recent_drains_analysis.py](../scripts/db_recent_drains_analysis.py) — broad recent-activity survey
- External APIs queried (2026-05-15):
  - `https://metadata.services.blockscout.com/api/v1/metadata?addresses=0x80b12bd0...&chainId=1` (OLI tags)
  - `https://eth.blockscout.com/api/v2/addresses/0x80b12bd0...` (native address page)
  - `https://eth.blockscout.com/api/v2/addresses/0x80b12bd0.../token-transfers` (holdings)
  - `https://eth.blockscout.com/api/v2/addresses/0x80b12bd0.../transactions?filter=from` (outbound)
  - `https://eth.blockscout.com/api/v2/transactions/0xb74b6953...` (specific LayerZero send tx)
- Layer 3 internal:
  - [memory/JOURNAL.md](../memory/JOURNAL.md) `2026-05-15` entries (sync v2, Phase A, recent drains)
  - [POTENTIAL_ATTACKS_V3.md](../POTENTIAL_ATTACKS_V3.md) Attack 1 + Attack 11a + Pattern D
  - [reports/correction_log.md](correction_log.md) Correction #20 (OLI mass mislabel sweep)
  - [reports/blockscout_tag_audit_2026-05-09.csv](blockscout_tag_audit_2026-05-09.csv) line 68

---

*This report should be cross-linked from UNK-031 in `memory/UNKNOWNS.md` on commit. If the external verification (action item #2) concludes that the OLI tag is incorrect, this report should be retired and the Correction #20 OLI-tagging methodology should be revisited.*
