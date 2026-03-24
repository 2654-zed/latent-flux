# ENTITY CLASSIFICATION REPORT
**Generated:** 2026-03-24 18:05 UTC
**Total classified entities:** 1080

---

## Category Breakdown

| Category | Count | % |
|---|---|---|
| BOT | 618 | 57.2% |
| COMMERCIAL | 345 | 31.9% |
| INFRASTRUCTURE | 63 | 5.8% |
| CRIMINAL | 44 | 4.1% |
| REFERENCE | 7 | 0.6% |
| INDIVIDUAL | 2 | 0.2% |
| INSTITUTIONAL | 1 | 0.1% |

---

## Subtype Detail

| Subtype | Count | % |
|---|---|---|
| unclassified_bot | 617 | 57.1% |
| private_infrastructure | 334 | 30.9% |
| gas_station | 23 | 2.1% |
| dex_router | 19 | 1.8% |
| known_attacker | 16 | 1.5% |
| cex_hot_wallet | 10 | 0.9% |
| org_deployer | 8 | 0.7% |
| ground_truth | 7 | 0.6% |
| token_factory | 7 | 0.6% |
| trap_contract | 6 | 0.6% |
| dex_pool | 5 | 0.5% |
| org_laundry | 5 | 0.5% |
| bridge | 4 | 0.4% |
| org_treasury | 4 | 0.4% |
| lp_manager | 3 | 0.3% |
| bot_operator | 2 | 0.2% |
| mixer | 2 | 0.2% |
| infrastructure_parasite | 1 | 0.1% |
| mev_factory | 1 | 0.1% |
| mev_vault | 1 | 0.1% |
| org_cashout | 1 | 0.1% |
| org_exit_ramp | 1 | 0.1% |
| rd_bot | 1 | 0.1% |
| trap_deployer | 1 | 0.1% |
| trap_inventory | 1 | 0.1% |

---

## Confidence Distribution

| Confidence | Count | % |
|---|---|---|
| CONFIRMED | 46 | 4.3% |
| HIGH | 68 | 6.3% |
| MEDIUM | 626 | 58.0% |
| LOW | 340 | 31.5% |

---

## Classification Sources

| Source | Count |
|---|---|
| bot_candidates_import | 617 |
| self_operating_heuristic | 334 |
| deployers_import | 67 |
| fund_tracer_registry | 40 |
| investigation_hardcoded | 9 |
| deployment_pattern_heuristic | 7 |
| confirmed_trap_import | 6 |

---

## Organization Attribution

| Org ID | Entities |
|---|---|
| org_001 | 16 |
| org_003 | 6 |
| org_002 | 2 |

---

## CRIMINAL Entities Detail

| Address | Subtype | Confidence | Org | Notes |
|---|---|---|---|---|
| 0xc0ffeefeed8b9d27... | trap_deployer | MEDIUM | - | revert_rate=80.3%, victims=69 |
| 0xe8e0c4883d7196a7... | infrastructure_parasite | HIGH | - | deployers.entity_type=infrastructure_parasite |
| 0x56190cac88b8d4b5... | known_attacker | HIGH | - | deployers.entity_type=known_attacker |
| 0x43aa42d2f11afe42... | known_attacker | HIGH | - | deployers.entity_type=known_attacker |
| 0xf5e753d3da60db21... | known_attacker | HIGH | - | deployers.entity_type=known_attacker |
| 0x5fb0b8584b34e56e... | known_attacker | HIGH | - | deployers.entity_type=known_attacker |
| 0x32dbfce225300249... | known_attacker | HIGH | - | deployers.entity_type=known_attacker |
| 0x59537353248d0b12... | known_attacker | HIGH | - | deployers.entity_type=known_attacker |
| 0x091101b0f31833c0... | known_attacker | HIGH | - | deployers.entity_type=known_attacker |
| 0xe5f8fe69b38613a8... | known_attacker | HIGH | - | deployers.entity_type=known_attacker |
| 0xaa06fde501a82ce1... | known_attacker | HIGH | - | deployers.entity_type=known_attacker |
| 0x2fad746cfaaf68aa... | known_attacker | HIGH | - | deployers.entity_type=known_attacker |
| 0xb13b2ab202cb902b... | known_attacker | HIGH | - | deployers.entity_type=known_attacker |
| 0xd356c82e0c85e156... | known_attacker | HIGH | - | deployers.entity_type=known_attacker |
| 0xd9ff21caeeea4329... | known_attacker | HIGH | - | deployers.entity_type=known_attacker |
| 0xbf6ec059f519b668... | known_attacker | HIGH | - | deployers.entity_type=known_attacker |
| 0x348df930e825da25... | known_attacker | HIGH | - | deployers.entity_type=known_attacker |
| 0xe1e6aa5332deaf0f... | known_attacker | HIGH | - | deployers.entity_type=known_attacker |
| 0xc6962004f452be92... | org_cashout | HIGH | org_001 | deployers.entity_type=cashout |
| 0xe93d64f3fbc35213... | org_deployer | HIGH | org_001 | deployers.entity_type=operator |
| 0xfd51e33d44b376ef... | org_deployer | HIGH | org_001 | deployers.entity_type=operator |
| 0x9f5db1b0436815de... | org_deployer | HIGH | org_003 | deployers.entity_type=org_003_ghost_deployer |
| 0x888a491ed0a09c93... | org_deployer | HIGH | org_003 | deployers.entity_type=org_003_ghost_deployer |
| 0x8f007f3e4f83a57c... | org_deployer | HIGH | org_003 | deployers.entity_type=org_003_ghost_deployer |
| 0xadb085d8279bf7af... | org_deployer | HIGH | org_003 | deployers.entity_type=org_003_ghost_deployer |
| 0x392c564a28d6d87d... | org_deployer | HIGH | org_003 | deployers.entity_type=org_003_ghost_deployer |
| 0x571ba99571f588d8... | org_deployer | HIGH | org_003 | deployers.entity_type=org_003_ghost_deployer |
| 0x01989c93890aed05... | org_exit_ramp | HIGH | org_001 | deployers.entity_type=cex_deposit |
| 0xfdaf1f1714810f8d... | org_laundry | HIGH | org_001 | deployers.entity_type=laundry |
| 0x27920e8039d2b6e9... | org_laundry | HIGH | org_001 | deployers.entity_type=lp_companion |
| 0xc6f780497a95e246... | org_laundry | HIGH | - | deployers.entity_type=laundry_candidate |
| 0xcda53b1f66614552... | org_laundry | HIGH | - | deployers.entity_type=laundry_candidate |
| 0x96daa0b8a5499ea9... | org_laundry | HIGH | org_001 | deployers.entity_type=lp_staging |
| 0xf186cb00e49e1849... | org_treasury | HIGH | org_001 | deployers.entity_type=treasury |
| 0x360e68faccca8ca4... | org_treasury | HIGH | org_001 | deployers.entity_type=treasury_branch |
| 0xde8eb937cb5475ee... | org_treasury | HIGH | org_002 | deployers.entity_type=org_002_treasury_junior |
| 0x238d7170f309a55b... | org_treasury | HIGH | org_002 | deployers.entity_type=org_002_treasury_senior |
| 0x2e20b26172a8625c... | trap_inventory | HIGH | - | deployers.entity_type=trap_inventory_operator |
| 0x3b8b8e5509975418... | trap_contract | CONFIRMED | org_001 | deployer=0xe93d64f3fbc35213..., reason=CONFIRMED T |
| 0x79a2f71187dc9fd9... | trap_contract | CONFIRMED | org_001 | deployer=0xe93d64f3fbc35213..., reason=CONFIRMED T |
| 0xc8e6a328d094609a... | trap_contract | CONFIRMED | org_001 | deployer=0xe93d64f3fbc35213..., reason=CONFIRMED T |
| 0x74b9a8351bd725ca... | trap_contract | CONFIRMED | org_001 | deployer=0xe93d64f3fbc35213..., reason=CONFIRMED T |
| 0xc8f28b043feb244c... | trap_contract | CONFIRMED | org_001 | deployer=0xe93d64f3fbc35213..., reason=CONFIRMED T |
| 0x3e6800980a97038c... | trap_contract | CONFIRMED | org_001 | deployer=0xe93d64f3fbc35213..., reason=CONFIRMED T |
