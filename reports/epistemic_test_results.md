# Layer 3 -- Epistemic Integrity Test Results
Generated: 2026-04-03T02:12:14Z

---
## A1: 'GoPlus detects 0 of the top 50'

Tables with 'goplus': ['goplus_results']
Columns with 'goplus': ['goplus_results.goplus_status']

**CORRECTION: CANNOT be verified from database.** No GoPlus data stored. The live API had SSL errors. We used honeypot.is instead (21/50 had data, 3 honeypot, 4 clean). The '0 of 50' claim is not evidence-based.

**Confidence: LOW.** Remove or reframe this claim.
---
## A2: 'org_001 has 899 deployers and 2,042 contracts'

| Method | Deployers | Contracts |
|--------|-----------|-----------|
| entity_classification | 16 | 462 |
| deployer_profiles | 26 | 818 |
| funding_trail | 308 | 1856 |
| **UNION** | **324** | **1875** |

**CORRECTION: '899 deployers, 2,042 contracts' is wrong.** Actual range: 16-324 deployers, 462-1875 contracts depending on attribution method. The numbers disagree by an order of magnitude.
**Confidence: MEDIUM.** True size is uncertain. Conservative: 16 deployers/462 contracts. Aggressive: 324/1875.

---
## A3: 'Camouflage ratio is 68%, stable'

Overall: 221/279 = **79.2%**

| Chain | Total | Camo | Ratio |
|-------|-------|------|-------|
| arbitrum | 71 | 50 | 70.4% |
| base | 207 | 170 | 82.1% |
| optimism | 1 | 1 | 100.0% |

Week 1: 2/5 = 40.0%  |  Week 2: 219/274 = 79.9%

**CORRECTION: Ratio is 79.2%, not 68%.** Off by 11.2pp.
Chain spread: 11.7pp. Week spread: 39.9pp.
'Stable' = questionable.

---
## A4: '832 wallet rotations, 302 high-confidence'

  >= 0.7: 4879
  >= 0.75: 2415
  >= 0.8: 1236
  >= 0.85: 348
  >= 0.9: 27

The '832' and '302' don't map to clean thresholds. These numbers were from a report computation, not direct DB counts.

Top 3 pairs:
  0x1b8f12da0a7385.. <-> 0x8d15e3064dd676.. score=0.999
  0x1b8f12da0a7385.. <-> 0xc30d8163ab9bd7.. score=0.998
  0x6cbf91ceb2a13c.. <-> 0xb4bf8b460215a1.. score=0.997

**Confidence: MEDIUM.** Similarity scores exist but 'rotation' implies temporal succession, which the table doesn't encode.

---
## A5: '49 victim-to-predator conversions'

Reverted callers who are also deployers: **84**
Bot candidates flagged as deployers: 35

  0x45314aa58fb061b2... reverted_first=2026-03-30 deployed_first=2026-03-30 traps_hit=12 deployed=29 genuine_conversion=NO (deployer first)
  0xd806e5453a74a6af... reverted_first=2026-04-02 deployed_first=2026-03-30 traps_hit=5 deployed=9 genuine_conversion=NO (deployer first)
  0xe33bd2238b46f2bb... reverted_first=2026-04-02 deployed_first=2026-04-02 traps_hit=5 deployed=8 genuine_conversion=NO (deployer first)

**Confidence: LOW.** A revert doesn't mean financial loss. Many are bot operators probing traps, not victims.

---
## B1: False Positive Rate of Suspected Tier

Total suspected: 36,915
Suspected with tx data: 1377
Look-legitimate (0 sigs, <5% revert, 10+ callers, 20+ tx): **10**
Estimated FP rate (of those with data): **0.7%**


---
## B4: Biggest Data Gaps

Contracts with zero tx events: **71,677** of 73,106 (98.0%)
Connection gaps: 245, ~22,661 missed blocks

---
## C1: Top 5 Contracts -- Real Victims or Self-Test?

  0x98189ee5d702ae30... base confirmed callers=1462 tx=1,471 deployer_tx=0(0.0%) REAL
  0x62865eed19f5dcc6... base suspected callers=858 tx=960 deployer_tx=0(0.0%) REAL
  0x485c27783a814904... base suspected callers=466 tx=472 deployer_tx=1(0.2%) REAL
  0xaa9c087543f791df... base confirmed callers=364 tx=698 deployer_tx=34(4.9%) REAL
  0x16cdc3ac11df029c... base suspected callers=303 tx=304 deployer_tx=1(0.3%) REAL

---
## C2: 'Trust amplification 14.2x'

Trust tables: ['trust_amplification']
**CORRECTION: No trust amplification data in DB. Cannot verify 14.2x.**

---
## C3: Behavioral Confirmations -- Self-Test Check

Trap events Apr 1: 19
Self-test (bot==deployer): 0
Genuine external victims: 19

---
## C4: Approve-to-Drain Timing

Valid timing pairs: 712
544 was claimed. Actual: 712. Different -- data has grown since that analysis.
The 37-hour ceiling COULD be an observation window artifact. 16 days of data is short.
**Confidence: MEDIUM.** Median is reliable but the ceiling needs more observation time.

---
## D1: False CRITICAL Exposure

Confirmed contracts with <5% revert and 20+ callers: **4**
These are our highest false-CRITICAL risk. Likely cause: deployer_history detection + single revert from non-deployer triggers confirmation.
**Most likely failure mode:** Contract from a known-bad deployer that is actually benign or deactivated.

---
## D2: Missed Trap Blind Spots

Unknown-tier contracts: 35,998
Unknown with reverts: 0
These are blind spots -- customer gets 'UNKNOWN' but contract may be harmful.

---
## SUMMARY OF ALL CORRECTIONS

1. A1: 'GoPlus detects 0/50' -- UNVERIFIABLE from DB. API was unreachable.
2. A2: org_001 claimed 899/2042. Actual: 16-324 deployers, 462-1875 contracts.
3. A3: Camouflage claimed 68%, actual 79.2%.
4. A4: '832/302' rotations -- numbers don't match clean threshold cuts from deployer_similarity.
5. A5: Claimed 49 victim-to-predator, actual 84.
6. C2: 'Trust amplification 14.2x' -- no data in DB to verify.