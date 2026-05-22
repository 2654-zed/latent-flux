# Threat-Actor Indicators of Compromise — Layer 3 corpus

**Purpose.** Archive of off-chain IOCs (malware hashes, C2 domains, infrastructure URLs, malicious repositories) extracted from threat-actor incident reports relevant to the Layer 3 corpus. These are NOT actively monitored by Layer 3 pipelines (Layer 3 is on-chain only) but are archived for partner-grade incident response and reference.

**Convention.** Each section is one named threat actor. Entries within a section are organized by the incident they were observed in. SHA256 hashes are canonical. Domain bracketing follows the [defang convention](https://en.wikipedia.org/wiki/Defanging_(cybersecurity)) so IOCs are not accidentally clickable.

**Authority.** When a lexicon entry references "IOCs in `reports/threat_actor_iocs_layer3.md`," this file is the cited source.

---

## UNC4899 / TraderTraitor (DPRK)

Also tracked as: TraderTraitor, Jade Sleet, Pressure Chollima. Assessed with high confidence as aligned with the DPRK Reconnaissance General Bureau. Primary focus: state-backed financial gain via cryptocurrency-infrastructure compromise. See lexicon entry [UNC4899 / TraderTraitor](../docs/lexicon.md#unc4899--tradertraitor).

### KelpDAO incident (2026-04-18, $292M / 116,500 rsETH)

Source: LayerZero Labs official incident report 2026-05-18 (`kelpdao-incident-report.pdf` in corpus archive). Forensics by Mandiant + CrowdStrike + zeroShadow.

#### Initial-access vector

- **Malicious GitHub repository (1.37 GB):** `github[.]com/pi2infra-can-4/gtn-candidate-repo.git`
- **Malicious Terraform provider registry:** `registry.hashicorp-aws[.]com/hashicorp/awsbeta`
- **Terraform provider binary fetched on `terraform init` / `terraform plan`:** `terraform-provider-awsbeta_v1.0.0`
  - SHA256: `f1df3737c972c5caf070d21a86f132648e6fe1ca07ba541c73eb30f1e4e96390`
- **Secondary fetch URL:** `https[:]//diagnose.hashicorp-aws[.]com/plugins/grpc/v6/schema/metrics/333afe63-c5a2-43f0-b046-7cbaa7797e8a`

#### macOS backdoors (Rust, ARM64)

**FLATROOF** — Telegram-based C2, command execution + file upload/download + data theft.

| Field | Value |
|---|---|
| File name | `SystemUpdate` |
| File type | MACH-O64 |
| Size (bytes) | 2,346,048 |
| MD5 | `c586e6be49105a23af8f306b560e35e6` |
| SHA256 | `6328567511d88fdc2ae0939c5ef17b7a63d2a833881900de018a4f12f4982525` |

C2 / infrastructure:
- `api.telegram[.]org`
- `hxxps://io.caiai[.]net/staticscandatav15/upload`
- `hxxps://technicais.sytes[.]net/statics/v11/a83f7fua937/dg0041`

File paths observed:
- `~/Library/com.apple.iTunesCloud/SystemUpdate`
- `/temp/collected_data/*`
- `/temp/collected_data.zip`

---

**ROOFDECK** — Nostr-decentralized C2 (multiple relays), system reconnaissance + file manipulation + remote shell + persistence via Launch Agent.

| Field | Value |
|---|---|
| File name | `iSync` |
| File type | MACH-O64 |
| Size (bytes) | 4,929,424 |
| MD5 | `c22a69c45ec74af11a9f87f195ebd392` |
| SHA256 | `61a110681a70af3dc21634558e12b1c00964f0cf48e90c89896eb0fda1e60b2d` |

C2 / infrastructure (Nostr decentralized relays + custom):
- `hxxps://api.nostr[.]watch:443/v1/online`
- `wss://relay.damus[.]io:443`
- `wss://nos[.]lol:443`
- `wss://nostr[.]mom:443`
- `wss://relay.snort[.]social:443`
- `wss://offchain[.]pub:443`
- `wss://relay.nostr[.]band:443`
- `wss://nostr.oxtr[.]dev:443`
- `commsouthindia[.]com`

File paths observed:
- `~/Library/com.apple.internal.ck/iSync`
- `~/Library/Spelling/words-en.dat`

Xprotect behavioral-rule violations observed: `macOS.Browser.Generic`, `macOS.Network.Outgoing`, `macOS.Persistence.ShellProfileFiles`.

#### In-memory `op-geth` poisoning malware (ELF Linux, x86_64)

**CARVED1** — launcher. Manually allocates memory for shared libraries, copies them in, applies relocations, resolves symbols, executes constructors. Hard-coded paths to load: `/usr/lib/librpcd.so` (CARVED2) and `/lib/ld-musl-x86_64.so.1` (musl libc).

| Field | Value |
|---|---|
| Type | ELF Position-Independent Executable |
| Size (bytes) | 810,992 |
| SHA256 | `ce8a08a888457dd6f44041acbef5db011edf5c6cda29fe603f653087d62f4a5f` |

Note: Forensic file carved from `op-geth` core dump; hash may vary from original on disk.

---

**CARVED2** — monitor. Uses `funchook` library (`github.com/kubo/funchook`) to hook Go's `syscall.Syscall` and `syscall.Syscall6` and intercept `read`, `write`, `close`, `sendto`. Parses HTTP + JSON in transit. Loads external `librpcw.so` library (UNRECOVERED) and calls its `rpc_handler` to decide which responses to modify.

| Field | Value |
|---|---|
| File path on disk | `/usr/lib/librpcd.so` |
| Filename in header | `librpcmon.so` |
| Type | ELF shared object |
| Size (bytes) | 15,424,136 |
| SHA256 | `5a90ae020ae40965848b7a344cd6e1439b63ab6c56489d4a16c2cfb97ecb034b` |

Note: Forensic file carved from `op-geth` core dump; hash may vary from original on disk.

---

**Unrecovered components** (per Mandiant forensics):
- The injector (initial dropper that placed CARVED1 + CARVED2 onto the GKE host)
- `librpcw.so` ("Worker" library) — the component that **decides** whether to modify Ethereum RPC requests/responses, given a parsed HTTP/JSON payload from CARVED2

#### Operational signature

- **Discriminating malware**: returns *correct* responses to LayerZero monitoring tools, *tampered* responses only to the DVN signing service. Strong evidence of prior reconnaissance about LayerZero's internal architecture.
- **VPN providers** observed for laundered access (March 30 → April 16, 2026): ExpressVPN, NordVPN, Mullvad.
- **Multi-protocol C2** (Telegram + Nostr) — defeats single-protocol takedown.
- **Forcing function**: DoS attack on external RPC providers immediately before exploit to force failover onto poisoned internal nodes. The DoS was observable on external-provider usage graphs (PDF Figure 1) — a 10M-method-calls spike between ~10:00 and ~12:00 UTC on 2026-04-18.

#### Attack-recipient on-chain trail (Layer 3 corpus relevance)

- Attack recipient (Ethereum, fresh address): `0x8B1b6c9A6DB1304000412dd21Ae6A70a82d60D3b`
- Funded via Tornado Cash 0.1 ETH pool 6.5 hours pre-exploit
- 116,500 rsETH received in single mint; fanned out to 7 wallets within 98 seconds:
  - `0x1f4c1c2e…adef` (53,000 rsETH)
  - `0xeba786c9…129b` (30,000 rsETH)
  - `0xcbb24a6b…55cc` (10,000 rsETH)
  - `0xbb6a6006…c787` (6,000 rsETH)
  - `0x8d11aeac…2d49` (5,000 rsETH)
  - `0x1b748b68…644c` (8,000 rsETH)
  - `0xe9e2f48b…d181` (4,500 rsETH)

None of the 8 attacker-controlled addresses appear in Layer 3's L2 corpus (Phase 1 + Phase 4 of `kelp_retrospective_replay.md` established this). The downstream Aave V3 deposit on Arbitrum was real but the depositor address is presumably a different (bridged) wallet not yet identified.

### Bybit / Safe{Wallet} incident (Feb 2025, $1.5B)

Same threat actor (UNC4899). Documented in public reporting; not detailed here. Operational pattern reused at Kelp:
- macOS-targeted social engineering of a developer
- Session-key harvest bypassing SSO + MFA
- VPN-laundered infrastructure access
- Single-attack-window exploit followed by rapid laundering fan-out

The recurring operational shape across both incidents is the load-bearing attribution signal.

---

## How to use this file

**At session start (when working on Kelp / cross-chain / DPRK-attribution topics):**
1. Read the relevant threat-actor section.
2. If you're investigating any address on the attack-recipient trail, cross-reference against the laundering ring documented above.
3. If a partner/customer brief mentions UNC4899 attribution, cite the multi-source corroboration (Mandiant + CrowdStrike + tanuki42 + tayvano) — this is Tier A now, not Tier B inference.

**Before adding a new IOC:**
1. Confirm the source is primary (vendor/researcher incident report) or independently verified.
2. Defang URLs / domains per the convention.
3. Add a date + source citation for every entry.
4. Cross-reference from the relevant lexicon entry.

**What this file does NOT do:**
- Layer 3 does not actively monitor these IOCs. They are archived for reference and partner incident response, not for runtime pipeline matching.
- This file does not attempt comprehensive threat-actor coverage. Only IOCs surfaced in incidents directly relevant to Layer 3's corpus (cross-chain bridges, DeFi protocols on monitored chains, attacker infrastructure with on-chain residue) are tracked.
