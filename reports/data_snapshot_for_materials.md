# Layer 3 Data Snapshot for External Materials
Generated: 2026-04-03T04:31:25Z

## 1. Corpus Overview

Total contracts: 73106
Unique deployers: 20433
Total deployer records: 20575
Total tx events: 200596
Unique callers (interacting addresses): 5987
Collection start: 2026-03-17T03:11:38+00:00
Collection end: 2026-04-02T18:00:57+00:00
Duration: ~16 days

By chain:
  base: 61,800
  arbitrum: 11,012
  optimism: 294

## 2. Detection Tiers

  suspected: 36,915
  unknown: 35,998
  confirmed: 193

Confirmed contracts: 193
Trap events total: 1638
Unique trap contracts: 198
Unique victim bots: 1125

Suspected by signature count:
  0 signatures: 31,998
  1 signatures: 4,325
  2 signatures: 495
  3 signatures: 97

Detection methods:
  bytecode_pattern: 56,504
  deployer_history: 16,602

## 3. Organization Intelligence

### Via entity_classification
  org_001: 16 deployers, 462 contracts (entity_classification)
  org_003: 6 deployers, 6 contracts (entity_classification)
  org_002: 2 deployers, 0 contracts (entity_classification)

### Via deployer_profiles.org_link
  org_001: 26 deployers (deployer_profiles)

### Via funding_trail
  org_001: 308 deployers, 1856 contracts (funding_trail)
  org_002: 529 deployers, 529 contracts (funding_trail)
  org_003: 0 deployers, 0 contracts (funding_trail)

### org_001 funding paths
  Whale trader (0xf70da97812cb96ac...): 799 deployers
  Gas station (0x8c826f795466e39a...): 310 deployers
  Whale share: 259.4%
  Gas station share: 100.6%

## 4. Camouflage Ratio

Qualifying contracts (10+ tx): 279
Camouflaged (<10% revert): 221
Overt (>=10% revert): 58
Ratio: 79.2%

By chain:
  arbitrum: 50/71 = 70.4%
  base: 170/207 = 82.1%
  optimism: 1/1 = 100.0%

Victim counts:
  camouflaged: 221 contracts, 5845 victims
  overt: 58 contracts, 233 victims

## 5. Wallet Rotation

Similarity score distribution:
  >= 0.7: 4879
  >= 0.75: 2415
  >= 0.8: 1236
  >= 0.85: 348
  >= 0.9: 27
  >= 0.95: 12

With temporal succession (similarity >= 0.85 AND b.first_seen > a.last_seen):
  Temporal rotations: 274

## 6. Behavioral Confirmation Stats

Confirmations by date:
  2026-03-21: 36
  2026-03-22: 23
  2026-03-23: 15
  2026-03-24: 15
  2026-03-25: 35
  2026-03-26: 26
  2026-03-27: 3
  2026-03-28: 2
  2026-03-29: 1
  2026-04-01: 19
  2026-04-02: 12

Total confirmed: 193

## 7. Alert Engine Stats

Alerts by type:
  LAUNDRY_PIPELINE: 37,641
  CASHOUT_MOVEMENT: 19,273
  HIGH_VELOCITY_DEPLOYER: 468
  address_activity: 216
  TRAP_CONFIRMED: 200
  OPERATOR_ACTIVE: 6
  LIVE_EXTRACTION_OBSERVED: 2
  BOT_DEATHWATCH: 1
Total alerts: 57807

Alerts per day:
  2026-03-18: 57136
  2026-03-20: 2
  2026-03-21: 36
  2026-03-22: 23
  2026-03-23: 16
  2026-03-24: 15
  2026-03-25: 35
  2026-03-26: 26
  2026-03-27: 3
  2026-03-28: 2
  2026-03-29: 1
  2026-03-30: 7
  2026-03-31: 6
  2026-04-01: 266
  2026-04-02: 233

## 8. Approval Drain Intelligence

Total approval_watchlist entries: 6067
Drained: 712
Pending (not drained): 5355

Drains by date:
  2026-03-22: 3
  2026-03-23: 2
  2026-03-24: 14
  2026-03-25: 94
  2026-03-26: 7
  2026-03-30: 24
  2026-03-31: 8
  2026-04-01: 541
  2026-04-02: 19

Top drain contracts:
  0xcf92c4caff3c8bafeb4b...: 282 victims
  0xaa9c087543f791dfda8f...: 233 victims
  0x01bba1aa150125301919...: 53 victims
  0xb154a4548f8bcb03f55f...: 41 victims
  0x20c52ae57cb5c32e849c...: 40 victims

## 9. Bot Ecosystem

Total bot candidates: 1058
Bot candidates who are deployers: 35

Bot strategies:
  UNKNOWN: 193
  FALLBACK_PROBER: 183
  L2_OPTIMIZED_SCANNER: 93
  BLIND_SCANNER: 92
  PROPRIETARY_FRAMEWORK: 60
  SINGLE_FUNCTION_GRINDER: 5
  SPRAY_PROBER: 4
  CALLBACK_ARBITRAGE: 3
  ROUTER_USER: 1

Strategy lifecycle:
  CALLBACK_ARBITRAGE: bots=3 traps=4 saturation=1.333 stage=WEAPONIZED
  ROUTER_USER: bots=1 traps=1 saturation=1.0 stage=WEAPONIZED
  L2_OPTIMIZED_SCANNER: bots=93 traps=23 saturation=0.247 stage=ARMS_RACE
  BLIND_SCANNER: bots=92 traps=1 saturation=0.011 stage=EARLY
  FALLBACK_PROBER: bots=183 traps=0 saturation=0.0 stage=EARLY
  PROPRIETARY_FRAMEWORK: bots=60 traps=0 saturation=0.0 stage=EARLY
  SINGLE_FUNCTION_GRINDER: bots=5 traps=0 saturation=0.0 stage=EARLY
  SPRAY_PROBER: bots=4 traps=0 saturation=0.0 stage=EARLY

Bot sophistication distribution:
  REACTIVE: 94
  HYBRID: 67
  SIMULATOR: 9

## 10. Benchmark Data

  honeypot.is | our_tier=confirmed | result=found | count=1
  honeypot.is | our_tier=confirmed | result=not_found | count=24
  honeypot.is | our_tier=suspected | result=found | count=20
  honeypot.is | our_tier=suspected | result=not_found | count=5

  honeypot.is definitive results: HONEYPOT=4, CLEAN=17, UNKNOWN=0
  Not in honeypot.is: 29

## 11. Dormant Threats

Dormant deployers (20+ contracts, <5% active): 391
Total staged contracts: 19218

Dragon (0x2e20b2...): 2077 contracts, 0 active

## 12. Strategy Lifecycle

  CALLBACK_ARBITRAGE: bots=3 traps=4 sat=1.333 stage=WEAPONIZED
  ROUTER_USER: bots=1 traps=1 sat=1.0 stage=WEAPONIZED
  L2_OPTIMIZED_SCANNER: bots=93 traps=23 sat=0.247 stage=ARMS_RACE
  BLIND_SCANNER: bots=92 traps=1 sat=0.011 stage=EARLY
  FALLBACK_PROBER: bots=183 traps=0 sat=0.0 stage=EARLY
  PROPRIETARY_FRAMEWORK: bots=60 traps=0 sat=0.0 stage=EARLY
  SINGLE_FUNCTION_GRINDER: bots=5 traps=0 sat=0.0 stage=EARLY
  SPRAY_PROBER: bots=4 traps=0 sat=0.0 stage=EARLY

## 13. Connection Gaps / Uptime

Total gaps: 245
Resolved: 115
Unresolved: 130
Estimated missed blocks: 22661

## 14. Funder Tracing Coverage

Total deployers: 20575
With funding trail: 13788
Coverage: 67.0%
Org links in funding_trail: 837
Gas stations identified: 307

## 15. Velocity / Growth

Contracts per day:
  2026-03-17: 417 contracts, 195 deployers
  2026-03-18: 1,428 contracts, 572 deployers
  2026-03-19: 4,787 contracts, 1,407 deployers
  2026-03-20: 3,928 contracts, 2,083 deployers
  2026-03-21: 4,418 contracts, 1,875 deployers
  2026-03-22: 4,419 contracts, 2,160 deployers
  2026-03-23: 2,359 contracts, 1,367 deployers
  2026-03-24: 3,598 contracts, 1,940 deployers
  2026-03-25: 3,684 contracts, 1,941 deployers
  2026-03-26: 4,515 contracts, 2,274 deployers
  2026-03-27: 587 contracts, 367 deployers
  2026-03-28: 3,993 contracts, 2,042 deployers
  2026-03-29: 4,001 contracts, 2,067 deployers
  2026-03-30: 8,008 contracts, 2,295 deployers
  2026-03-31: 8,912 contracts, 2,072 deployers
  2026-04-01: 7,153 contracts, 1,873 deployers
  2026-04-02: 6,899 contracts, 1,882 deployers

Average contracts/day (last 4 days): 6995.0

## 16. Additional Tables Row Counts

  bait_profiles: 51
  behavioral_anomalies: 121
  bytecode_families: 406
  bytecode_family_members: 23984
  camouflage_metrics: 9
  daily_metrics: 9
  deployer_profiles: 1873
  deployer_similarity: 4879
  self_test_traps: 257
  vanity_tags: 142
  watchlist: 18
  watchlist_hits: 3385
  cex_deposit_candidates: 980414
  org_transfer_events: 133115
  liquidity_events: 2635
  approval_events: 4664
  pair_creation_events: 14