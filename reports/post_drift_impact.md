# Post-Drift Impact Analysis
## Behavioral Changes on L2 Chains After the April 1, 2026 Drift Protocol Exploit ($285M)

Generated: 2026-04-04T12:25:48Z
Baseline period: March 25-31 (7 days pre-Drift)
Post-Drift period: April 1-3 (3 days, data collection ongoing)

**Limitation:** Only 3 days of post-Drift data. Some behavioral shifts take weeks to materialize. This analysis should be re-run at day 14 and day 30 post-Drift.

---
## 1. Contract Deployment Rate

| Day | Contracts | Deployers |
|-----|-----------|-----------|
| 2026-03-25 | 3,684 | 1,941 |
| 2026-03-26 | 4,515 | 2,274 |
| 2026-03-27 | 587 | 367 |
| 2026-03-28 | 3,993 | 2,042 |
| 2026-03-29 | 4,001 | 2,067 |
| 2026-03-30 | 8,008 | 2,295 |
| 2026-03-31 | 8,912 | 2,072 |
| 2026-04-01 ** | 7,153 | 1,873 |
| 2026-04-02 | 6,899 | 1,882 |

**Baseline avg:** 4,814 contracts/day
**Post-Drift avg:** 7,026 contracts/day
**Change:** +45.9%
**Interpretation:** Deployment rate increased post-Drift. **SIGNIFICANT**
Plausible Drift connection: A major exploit can trigger copycat activity or cause operators to accelerate deployments before potential crackdowns.

---
## 2. Approval Events

| Day | Approvals |
|-----|-----------|
| 2026-03-29 | 24 |
| 2026-03-30 | 870 |
| 2026-03-31 | 1,457 |
| 2026-04-01 | 1,060 |
| 2026-04-02 | 1,253 |

**Baseline avg:** 784/day | **Post-Drift avg:** 1,156/day | **Change:** +47.6%
**Interpretation:** Users approving more post-Drift. **SIGNIFICANT**

---
## 3. Liquidity Events

| Day | Liquidity Events | Critical |
|-----|-----------------|----------|
| 2026-03-29 | 64 | 0 |
| 2026-03-30 | 670 | 19 |
| 2026-03-31 | 730 | 21 |
| 2026-04-01 | 569 | 32 |
| 2026-04-02 | 602 | 14 |

**Baseline avg:** 488/day | **Post-Drift avg:** 586/day | **Change:** +20.0%

---
## 4. Bot Activity

| Day | Bot Events | Unique Bots |
|-----|-----------|-------------|
| 2026-03-29 | 471 | 256 |
| 2026-03-30 | 3,463 | 426 |
| 2026-03-31 | 5,842 | 230 |
| 2026-04-01 | 6,051 | 266 |
| 2026-04-02 | 10,349 | 306 |

**Baseline avg:** 3,259 events/day | **Post-Drift avg:** 8,200/day | **Change:** +151.6%
**Interpretation:** **SIGNIFICANT**

---
## 5. Transaction Events (overall interaction volume)

| Day | TX Events | Reverts | Revert % | Callers |
|-----|-----------|---------|----------|---------|
| 2026-03-27 | 66 | 6 | 9.1% | 13 |
| 2026-03-28 | 2,357 | 191 | 8.1% | 172 |
| 2026-03-29 | 121 | 0 | 0.0% | 75 |
| 2026-03-30 | 7,476 | 686 | 9.2% | 1,507 |
| 2026-03-31 | 15,553 | 4,571 | 29.4% | 1,415 |
| 2026-04-01 | 59,846 | 16,327 | 27.3% | 1,135 |
| 2026-04-02 | 115,177 | 23,655 | 20.5% | 2,689 |

**Baseline avg:** 5,115 tx/day | **Post-Drift avg:** 87,512/day | **Change:** +1611.0%
 **SIGNIFICANT**

---
## 6. Org Activity (org_001 via funding_trail)

| Day | org_001 Contracts | Deployers |
|-----|-------------------|-----------|
| 2026-03-25 | 113 | 44 |
| 2026-03-26 | 156 | 46 |
| 2026-03-27 | 26 | 13 |
| 2026-03-28 | 106 | 59 |
| 2026-03-29 | 127 | 40 |
| 2026-03-30 | 114 | 51 |
| 2026-03-31 | 116 | 70 |
| 2026-04-01 | 103 | 38 |
| 2026-04-02 | 384 | 39 |

**Baseline avg:** 108.3 contracts/day | **Post-Drift avg:** 243.5/day | **Change:** +124.9%
 **SIGNIFICANT**

**org_002:** Baseline 50.7/day -> Post-Drift 66.0/day (+30.1%)

---
## 7. New Deployer Creation Rate (wallet rotation proxy)

| Day | New Deployers |
|-----|---------------|
| 2026-03-25 | 983 |
| 2026-03-26 | 1,487 |
| 2026-03-27 | 272 |
| 2026-03-28 | 1,491 |
| 2026-03-29 | 1,499 |
| 2026-03-30 | 2,310 |
| 2026-03-31 | 1,811 |
| 2026-04-01 | 1,608 |
| 2026-04-02 | 1,494 |

**Baseline avg:** 1,408/day | **Post-Drift avg:** 1,551/day | **Change:** +10.2%


---
## 8. Alert Volume

| Day | Total Alerts | TRAP_CONFIRMED | HIGH_VELOCITY |
|-----|-------------|----------------|---------------|
| 2026-03-25 | 35 | 35 | 0 |
| 2026-03-26 | 26 | 26 | 0 |
| 2026-03-27 | 3 | 3 | 0 |
| 2026-03-28 | 2 | 2 | 0 |
| 2026-03-29 | 1 | 1 | 0 |
| 2026-03-30 | 7 | 7 | 0 |
| 2026-03-31 | 6 | 6 | 0 |
| 2026-04-01 | 266 | 19 | 247 |
| 2026-04-02 | 233 | 12 | 221 |

---
## 9. Trap Confirmation Rate

| Day | Trap Fires | Contracts | Victims |
|-----|-----------|-----------|---------|
| 2026-03-25 | 360 | 45 | 244 |
| 2026-03-26 | 291 | 52 | 184 |
| 2026-03-27 | 29 | 19 | 27 |
| 2026-03-28 | 15 | 8 | 15 |
| 2026-03-29 | 1 | 1 | 1 |
| 2026-03-30 | 12 | 7 | 12 |
| 2026-03-31 | 10 | 8 | 10 |
| 2026-04-01 | 19 | 19 | 18 |
| 2026-04-02 | 12 | 12 | 12 |

**Baseline avg:** 102.6 traps/day | **Post-Drift avg:** 15.5/day | **Change:** -84.9%
Note: Behavioral confirmation module went live April 1. Pre-Drift numbers are from backfill, post-Drift are real-time detections. Not directly comparable.

---
## 10. Drain Activity

| Day | Wallets Drained |
|-----|-----------------|
| 2026-03-25 | 94 |
| 2026-03-26 | 7 |
| 2026-03-30 | 24 |
| 2026-03-31 | 8 |
| 2026-04-01 | 541 |
| 2026-04-02 | 19 |

**Baseline avg:** 33/day | **Post-Drift avg:** 280/day | **Change:** +742.1%
 **SIGNIFICANT**

---
## 11. Dormant Fleet Activation

Dormant deployers (20+ contracts, zero pre-April activity) that activated post-Drift: **10**
  0x5f7476ee17eccbc57de4... activated=28 contracts, first=2026-04-01T10:18:38
  0x662cb998fdcfed53cf1f... activated=27 contracts, first=2026-04-01T18:07:23
  0x694834fea7b9cf607b9a... activated=26 contracts, first=2026-04-01T14:55:30
  0x0984ad6ffe9464db49a0... activated=18 contracts, first=2026-04-02T08:38:49
  0xe5c80000df78a7f4f773... activated=14 contracts, first=2026-04-02T12:01:21
  0x045f4aa85dca9e00b901... activated=8 contracts, first=2026-04-02T13:09:59
  0x9905e56cdc20199ad06c... activated=8 contracts, first=2026-04-01T15:51:16
  0x4988ef88e932c1ab767d... activated=4 contracts, first=2026-04-02T02:18:39
  0x19cbeb465cb8b3b988e4... activated=2 contracts, first=2026-04-01T22:04:57
  0xa41d5faf7ba8b82e2761... activated=1 contracts, first=2026-04-01T16:04:33

---
## Summary: Significant Changes (>20%)

| Metric | Baseline (Mar 25-31) | Post-Drift (Apr 1-3) | Change | Signal? |
|--------|---------------------|---------------------|--------|---------|
| Contract deployments/day | 4,814 | 7,026 | +45.9% | YES |
| TX events/day | 5,115 | 87,512 | +1611.0% | YES |
| New deployers/day | 1,408 | 1,551 | +10.2% | no |
| Bot events/day | 3,259 | 8,200 | +151.6% | YES |
| Approval events/day | 784 | 1,156 | +47.6% | YES |
| Drains/day | 33 | 280 | +742.1% | YES |
| Liquidity events/day | 488 | 586 | +20.0% | no |

## What to Re-Measure

- **Day 14 post-Drift (April 15):** Check if deployment rate changes are sustained or transient
- **Day 30 post-Drift (May 1):** Measure wallet rotation rate change, new org emergence, dormant fleet activation completion
- **Approval volume:** If Drift causes lasting user caution, approval rates should decline for 2-4 weeks
- **New bot strategies:** Major exploits often trigger new bot development. Check for new selectors and strategies after 2-3 weeks (development time)
- **Cross-chain migration:** If Drift increases Solana risk perception, operators may move to L2s. Watch for new deployer influx with no prior L2 history