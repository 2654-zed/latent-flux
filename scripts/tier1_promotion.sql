-- Tier-1 cluster promotions (generated 2026-04-23)
-- Review each INSERT before executing. Each INSERT is idempotent via
-- INSERT OR IGNORE against the (address, chain) primary key of org_wallets.

BEGIN IMMEDIATE;

-- === orgcand_a8f337083daf -> org_005 ===
-- chain=arbitrum  size=8  funder=0x9b64203878f24eb0cdf55c8c6fa7d08ba0cf77e5
-- reason: Serial single-honeypot operator. Lead deployer 0xda977393363d produced 4 trap_events on 2026-03-23 against 4 victims in ...

-- treasury / funder
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0x9b64203878f24eb0cdf55c8c6fa7d08ba0cf77e5', 'arbitrum', 'org_005', 'treasury', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'Serial single-honeypot operator. Lead deployer 0xda977393363d produced 4 trap_events on 2026-03-23 against 4 victims in an 8h window, all on contract 0x248d0105ec63. Fleet of 62 contracts with 1 confirmed acts as decoy around one active trap. Chain=arbitrum. Promoted 2026-04-23 from Tier-1 investigator review.');

INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0xda977393363d166e9e0441bc2e2d02bc1b53f38f', 'arbitrum', 'org_005', 'operator', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_a8f337083daf; traps=4, confirmed=1, total_contracts=62');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0x7018b6b597b8a1c341682d4418492ed2e56e0266', 'arbitrum', 'org_005', 'operator_2', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_a8f337083daf; traps=1, confirmed=1, total_contracts=2');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0x94c51179798aa618776f6dbd0f12305b73e60459', 'arbitrum', 'org_005', 'operator_3', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_a8f337083daf; traps=0, confirmed=0, total_contracts=36');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0x01a63d166348b8ba4b3f34a4cb1d2392fca25535', 'arbitrum', 'org_005', 'operator_4', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_a8f337083daf; traps=0, confirmed=0, total_contracts=22');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0xd6d85829708a17d360a12c921b19f4c4d5f6da88', 'arbitrum', 'org_005', 'operator_5', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_a8f337083daf; traps=0, confirmed=0, total_contracts=11');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0x1feb20495e7e2f0ffdb1ad79dcbc7648619f16fa', 'arbitrum', 'org_005', 'operator_6', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_a8f337083daf; traps=0, confirmed=0, total_contracts=3');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0xdd10b07ed94a654285c3a812654a7a8534714b51', 'arbitrum', 'org_005', 'operator_7', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_a8f337083daf; traps=0, confirmed=0, total_contracts=2');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0x1f11e57eec9c503630c6db6e391c11bcf650b88f', 'arbitrum', 'org_005', 'operator_8', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_a8f337083daf; traps=0, confirmed=0, total_contracts=1');

-- mark candidate promoted
UPDATE org_candidates SET status = 'promoted', notes = 'Promoted to org_005 on 2026-04-23 via tier1_review' WHERE candidate_id = 'orgcand_a8f337083daf';


-- === orgcand_5564c29a9070 -> org_006 ===
-- chain=base  size=15  funder=0xd34ea7278e6bd48defe656bbe263aef11101469c
-- reason: Persistent repeat-victim operator. Lead deployer 0x982ff6be4aa1 has a 3+ week history of trapping the same bot 0x456a3e0...

-- treasury / funder
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0xd34ea7278e6bd48defe656bbe263aef11101469c', 'base', 'org_006', 'treasury', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'Persistent repeat-victim operator. Lead deployer 0x982ff6be4aa1 has a 3+ week history of trapping the same bot 0x456a3e06c64d across 3 different contracts (2026-03-22 -> 2026-04-14). Bot operator appears not to update blacklist. Chain=base. Promoted 2026-04-23 from Tier-1 investigator review.');

INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0x982ff6be4aa1d8e79d516840cdb1be47a1b9a3e5', 'base', 'org_006', 'operator', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_5564c29a9070; traps=3, confirmed=3, total_contracts=10');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0x02d7bbbe2d1e241a089b3279358a625f87d1b994', 'base', 'org_006', 'operator_2', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_5564c29a9070; traps=0, confirmed=1, total_contracts=56');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0xe111763e9c7d008f18e72ade0d4acdba32137991', 'base', 'org_006', 'operator_3', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_5564c29a9070; traps=0, confirmed=0, total_contracts=9');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0xffea3be2d088dfabcaf64837450919b53e40fbe5', 'base', 'org_006', 'operator_4', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_5564c29a9070; traps=0, confirmed=0, total_contracts=9');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0x4253aaca78419dd5d82a74708c29c6bdfe325565', 'base', 'org_006', 'operator_5', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_5564c29a9070; traps=0, confirmed=0, total_contracts=8');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0xa008286ebaeb69e19be91e12fdcd277e00aad9e1', 'base', 'org_006', 'operator_6', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_5564c29a9070; traps=0, confirmed=0, total_contracts=4');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0xb58b742fffbcfa4f42d58026babd73e38069de8f', 'base', 'org_006', 'operator_7', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_5564c29a9070; traps=0, confirmed=0, total_contracts=4');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0x64bba6b701b87c2035607bc2b6fce38bb08c75f5', 'base', 'org_006', 'operator_8', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_5564c29a9070; traps=0, confirmed=0, total_contracts=2');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0xf57e8952e2ec5f82376ff8abf65f01c2401ee294', 'base', 'org_006', 'operator_9', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_5564c29a9070; traps=0, confirmed=0, total_contracts=2');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0x1bb09aa949c64879dff1fd38b12c7ebc3d7d94c0', 'base', 'org_006', 'operator_10', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_5564c29a9070; traps=0, confirmed=0, total_contracts=1');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0x3c319a93f64cd474dae11c042a231df09c601597', 'base', 'org_006', 'operator_11', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_5564c29a9070; traps=0, confirmed=0, total_contracts=1');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0x4e7ceb11f0101895caff8521e78f7dd9f0e0380b', 'base', 'org_006', 'operator_12', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_5564c29a9070; traps=0, confirmed=0, total_contracts=1');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0xa2e2008e542ac85de8986175efe7a3afc101a2e1', 'base', 'org_006', 'operator_13', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_5564c29a9070; traps=0, confirmed=0, total_contracts=1');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0xda676679c524e415323239564ae4ceb8a7e40e24', 'base', 'org_006', 'operator_14', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_5564c29a9070; traps=0, confirmed=0, total_contracts=1');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0xf010243df372d2a14fd836395084e0347c0a9009', 'base', 'org_006', 'operator_15', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_5564c29a9070; traps=0, confirmed=0, total_contracts=1');

-- mark candidate promoted
UPDATE org_candidates SET status = 'promoted', notes = 'Promoted to org_006 on 2026-04-23 via tier1_review' WHERE candidate_id = 'orgcand_5564c29a9070';


-- === orgcand_899651790f70 -> org_007 ===
-- chain=arbitrum,base  size=14  funder=0xb38e8c17e38363af6ebdcb3dae12e0243582891d
-- reason: Cross-chain operator (Arbitrum + Base), 14 deployers. 3 trap_events across 3 distinct deployers/contracts/bots. Prep-to-...

-- treasury / funder
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0xb38e8c17e38363af6ebdcb3dae12e0243582891d', 'arbitrum', 'org_007', 'treasury', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'Cross-chain operator (Arbitrum + Base), 14 deployers. 3 trap_events across 3 distinct deployers/contracts/bots. Prep-to-discharge latency 2d 11h — cleanest observed case of deploy+fire cycle. Mainnet cohort dispersed (not prepared-pool signature). Promoted 2026-04-23 from Tier-1 investigator review.');

INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0x5c89e16cfedb529a0a4632a8e2440a600f6af274', 'arbitrum', 'org_007', 'operator', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_899651790f70; traps=1, confirmed=1, total_contracts=22');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0x596cb11f0cc64cb24ab11ac5eab5ee7e32de74b1', 'arbitrum', 'org_007', 'operator_2', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_899651790f70; traps=1, confirmed=1, total_contracts=10');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0xe5aa0582c701f32fe256a2ed3944faa0def8c048', 'arbitrum', 'org_007', 'operator_3', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_899651790f70; traps=1, confirmed=1, total_contracts=2');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0x83450e0d5937a4bebaf49159074e382a72723e19', 'arbitrum', 'org_007', 'operator_4', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_899651790f70; traps=0, confirmed=0, total_contracts=21');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0x21581cb82d9a66126fbe7639f4af55ddfea48e26', 'arbitrum', 'org_007', 'operator_5', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_899651790f70; traps=0, confirmed=0, total_contracts=18');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0x278012102fa52734f0c6e5eeb22e8f7743997b66', 'arbitrum', 'org_007', 'operator_6', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_899651790f70; traps=0, confirmed=0, total_contracts=17');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0xbe682fed068cc8deb7339255c7f6d4e7ce5a482e', 'arbitrum', 'org_007', 'operator_7', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_899651790f70; traps=0, confirmed=0, total_contracts=12');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0x02458dad86e3a014db36e5ea63589e11030bf24d', 'arbitrum', 'org_007', 'operator_8', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_899651790f70; traps=0, confirmed=0, total_contracts=8');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0x4c8518880e2c15771ab69b99b07349a2e2132cc2', 'base', 'org_007', 'operator_9', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_899651790f70; traps=0, confirmed=0, total_contracts=5');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0xdc4e07fcd1dac2f4cddb839a2e826a851df54035', 'arbitrum', 'org_007', 'operator_10', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_899651790f70; traps=0, confirmed=0, total_contracts=3');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0x11cbb1467fd171d342cd993af2b0687db3c83b4c', 'arbitrum', 'org_007', 'operator_11', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_899651790f70; traps=0, confirmed=0, total_contracts=2');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0x27ddd73edc3b5d0d627b753cd6841150ab38f8ea', 'arbitrum', 'org_007', 'operator_12', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_899651790f70; traps=0, confirmed=0, total_contracts=2');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0xf34aeb83f021b31e1a01e7bc5e7a6f37a7ff190a', 'arbitrum', 'org_007', 'operator_13', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_899651790f70; traps=0, confirmed=0, total_contracts=2');
INSERT OR IGNORE INTO org_wallets (address, chain, org_id, role, added_at, added_by, reason) VALUES ('0x755c3444bb26d1f26c92c6073d3e717282b0878c', 'arbitrum', 'org_007', 'operator_14', '2026-04-23T14:56:02.721486+00:00', 'tier1_review_2026_04_23', 'deployer from orgcand_899651790f70; traps=0, confirmed=0, total_contracts=1');

-- mark candidate promoted
UPDATE org_candidates SET status = 'promoted', notes = 'Promoted to org_007 on 2026-04-23 via tier1_review' WHERE candidate_id = 'orgcand_899651790f70';


COMMIT;

-- After execution, verify with:
--   SELECT org_id, COUNT(*) FROM org_wallets GROUP BY org_id ORDER BY org_id;
--   SELECT status, COUNT(*) FROM org_candidates GROUP BY status;
