# Architectural Decision Records

Each ADR records a non-trivial choice that a future agent might re-derive otherwise. Format: numbered, dated, with context / decision / consequences / alternatives.

---

## ADR-001 — Base64-framed Railway SSH transport for production sync

**Date:** 2026-05-09

**Context:** Production SQLite (10 GB) needed periodic sync to local. Railway SSH was the only inbound mechanism since the deployed API has no `/dump` endpoint. Empirical testing showed two transport-corruption modes:
- Raw binary bytes: 4 stray byte changes per 1024 random bytes (CRLF translation on `\n` bytes)
- stdout/stderr merging: banner content from the SSH session contaminates the data stream

**Decision:** Stream all binary payloads as base64 framed by `L3SYNC_PAYLOAD_START` / `L3SYNC_PAYLOAD_END` markers. Local wrapper trims banner content by searching for markers; mmap-search for performance; chunked base64-decode for bounded memory.

**Consequences:**
- 33% wire-byte overhead (acceptable for the use case)
- Local wrapper must handle marker absence as error (not silently corrupt)
- Approach codified in `scripts/sync_prod_db.py` + `scripts/sync_prod_db_remote.py`
- Becomes the template for all `railway ssh`-dispatched scripts (apply_correction_20_via_ssh.py uses same)

**Alternatives considered:**
- Gzip + raw binary stream: rejected — CRLF translation still corrupts
- `railway run` instead of `ssh`: rejected — `railway run` is local execution, not container execution
- Switch to `railway shell` with file copy: rejected — interactive only, doesn't pipe
- Add `/dump` endpoint to production: would have worked but adds attack surface; deferred

---

## ADR-002 — OLI guardrail at `classify_address` boundary, not per-detector

**Date:** 2026-05-09 (Correction #20)

**Context:** Correction #20 surfaced 18 institutional addresses misclassified as adversarial by behavioral detectors. The fix needed to prevent recurrence without rewriting every detector.

**Decision:** Add a guardrail at the boundary of `surveillance/entity_classifier.py:classify_address`. If the address has a HIGH-severity OLI tag AND the requested `subtype` is in `_OLI_GUARDED_TRAP_SUBTYPES`, redirect to `COMMERCIAL/institutional_oli_tagged` with the original request preserved in notes.

**Consequences:**
- Every detector inherits the guardrail without modification — the boundary check covers all upstream callers
- Adversarial classifications now require both behavioral signal AND absence of HIGH-severity public label
- Notes field preserves the original classification request for audit
- INV-007 codifies this as a system invariant

**Alternatives considered:**
- Per-detector guardrail: rejected — would have required modifying 11 detector functions and would have to be re-added to each new detector
- Post-classification filter (run guardrail after classify_address writes): rejected — leaves window where downstream consumers see misclassification
- Run OLI lookup INSIDE classify_address (live fetch): rejected — adds latency to every classification; chose cache-only with separate enrichment job

---

## ADR-003 — Two-detector pattern for hidden-drain functions (signature + semantic)

**Date:** 2026-05-13

**Context:** The `0xaeac0e69` honeypot template exposed a hidden `approev(address)` function for privileged-caller balance draining. Two approaches considered for detection:
- Signature match: search bytecode for the 4-byte selector `0x3ed67ecd` (= keccak256("approev(address)")[:4])
- Semantic match: search for the structural pattern (privileged-caller gate + balance-mapping write + no Transfer event)

**Decision:** Ship both. Run them in parallel via `PATTERN_REGISTRY`. They are complementary, not substitutive:
- Signature is fast and brittle to name rotation (operator renames `approev` → `appreve` → escapes)
- Semantic is slower and more durable; catches the pattern shape regardless of name. But misses cases where Solidity inlines `_msgSender()` into a helper function (proven on the canonical 0xaeac0e69 case — semantic detector missed it; signature detector caught it)

**Consequences:**
- Each detector contributes; neither claims full coverage
- New operator-naming rotations get added to `KNOWN_HIDDEN_DRAIN_SELECTORS` list (signature detector grows)
- Future semantic improvement: control-flow-aware EVM disassembly tracking function boundaries via JUMPDEST — would catch the helper-function case (deferred)

**Alternatives considered:**
- Signature-only: rejected — too brittle; one rename defeats it
- Semantic-only: rejected — proven inadequate on the canonical case
- Bytecode-disassembly-based: would be best but is a much bigger lift; chose pragmatic two-detector path

---

## ADR-004 — `memory/` is the canonical agent-memory directory

**Date:** 2026-05-13

**Context:** Agent memory was previously scattered: `memory.md` at repo root, `docs/INDEX.md`, `docs/lexicon.md`, `reports/correction_log.md`. No single place to read on session start. Phase 2 of the 2026-05-13 design proposed a unified directory; the user explicitly created `memory/LOOP.md`, signaling agreement.

**Decision:** Consolidate all agent-memory artifacts under `memory/`:
- `memory/LOOP.md` — session-end reflection protocol (mandatory)
- `memory/STATE.md` — current snapshot (200 lines max)
- `memory/JOURNAL.md` — long-form session journal (append-only; renamed from root `memory.md`)
- `memory/UNKNOWNS.md` — open questions with canonical schema
- `memory/DECISIONS.md` — this file
- `memory/INVARIANTS.md` — system invariants
- `memory/REFLECTION_LOG.csv` — audit trail for the loop itself

The pre-existing surveillance-side documents (`docs/INDEX.md`, `docs/lexicon.md`, `reports/correction_log.md`) stay where they are — they're domain artifacts, not generic agent-memory.

**Consequences:**
- Session start: agent reads `memory/STATE.md` first (~200 lines), then scans `memory/UNKNOWNS.md` and last entry of `memory/JOURNAL.md`. Total ~500-1000 lines of context-load.
- Session end: the LOOP.md 7-step protocol writes back to these files.
- Root `memory.md` migrated via `git mv` → `memory/JOURNAL.md` to preserve history.

**Alternatives considered:**
- Use `.agent-context/` hidden directory: rejected — useful files shouldn't be hidden from human readers
- Use `docs/` for everything: rejected — `docs/` is for surveillance-domain documentation, not agent operating state
- Keep scattered: rejected — proven not to scale across sessions

---

## ADR-005 — Adopt 7-step reflection loop with mandatory citation and skipped-step accounting

**Date:** 2026-05-13

**Context:** The Phase 2 memory design was write-only — append protocols without integration. Result was a coherent past slowly fragmenting into a stale present. User proposed the 7-step `memory/LOOP.md` protocol; I extended with three structural requirements.

**Decision:** Adopt the LOOP.md protocol at session-end, with three required refinements:
1. **Step 6 (Coherence Check) requires citations** — every "confirmed/refined/contradicted" claim must point to a STATE.md anchor line + the evidence file:line that did the work
2. **Step 5 (Surprise Logging) requires "Expected vs Observed" structure** — without prior prediction, surprise is hindsight bias dressed up
3. **Skipped-step accounting is mandatory** — skips are allowed but must record reason. Rule-of-three: same skip three sessions running means the step is malformed

**Consequences:**
- ~5-10 minute reflection cost per session
- REFLECTION_LOG.csv produces audit trail for the loop's own health
- Confidence calibration on UNKNOWNs (HIGH = primary-source + cross-ref; MEDIUM = primary only; LOW = secondary only) creates a "revisit-LOW" research workflow
- Step 6 transforms append-only memory into self-correcting memory

**Alternatives considered:**
- Skip reflection entirely (write-only): rejected — proven failure mode
- Fewer steps (3-4): considered but each step covers a distinct class of integration; conflating would lose specificity
- Don't enforce citations: rejected — vibes-checks aren't an integration mechanism
- Make all steps optional: rejected — Step 6 is load-bearing; making it optional defeats the loop's purpose

---

## ADR-006 — Local-only git hooks managed via `scripts/hooks/` + opt-in installer

**Date:** 2026-05-13 (forced decision after 2 prior skips; rule-of-three trigger)

**Context:** UNK-002 resolution revealed that this repo uses two local git hooks installed in `.git/hooks/`:
- `pre-commit` — runs `python scripts/update_readme.py` to refresh README AUTOGEN sections
- `post-commit` — runs `git push origin HEAD` (auto-push every commit)

`.git/hooks/` is not tracked by git. Fresh clones lack these hooks. A new contributor (or the same developer on a fresh machine) gets neither the README auto-regen nor the auto-push behavior.

Two skip cycles in the reflection-loop deferred this decision (sessions 2026-05-13 pass 2 and pass 3). Rule-of-three trigger: 3rd deferral would mark Step 3 (Decision Extraction) as malformed.

**Decision:** Hybrid — **tracked source-of-truth in `scripts/hooks/` + opt-in installer in `scripts/install_hooks.sh`**, NOT mandatory installation.

Concretely:
- `scripts/hooks/pre-commit` and `scripts/hooks/post-commit` are tracked copies of the current hook content
- `scripts/install_hooks.sh` copies them into `.git/hooks/` when run
- Neither runs automatically; a fresh clone deliberately omits them. Run `bash scripts/install_hooks.sh` once after clone to opt in.

**Consequences:**
- Source-of-truth for hook content is now in git history. Hooks can be reviewed, diffed, and updated like any other code.
- Fresh clones default to NO auto-push, which is the safer default. The developer choosing to enable it is an explicit consent step.
- The post-commit auto-push behavior is preserved for the primary developer (existing `.git/hooks/post-commit` is unchanged by this ADR; the tracked copy is just a reference).
- New invariant candidate INV-015 was considered but not needed — the installer ergonomics are documented in STATE.md "Git hooks" section + this ADR. Not a system invariant.

**Alternatives considered:**
- **Move hooks to `core.hooksPath`** (`git config core.hooksPath scripts/hooks`): would make hooks mandatory and require zero opt-in. Rejected because auto-push is a developer-preference, not a project policy — making it mandatory imposes choice on others.
- **Leave hooks fully local** (status quo, no tracked copies): rejected — source-of-truth is then in `.git/hooks/` which is fragile (lost on `.git` deletion or fresh clone with no backup).
- **Use `husky` or another hook manager**: rejected — adds a Node.js dependency for a Python repo; overkill for two shell scripts.
- **Encode in `pyproject.toml` via `pre-commit` framework**: rejected — overkill; the hooks are 4 lines each and the existing local setup works.

**Status:** RESOLVED. Tracked hook copies + installer landed 2026-05-13. STATE.md "Git hooks" section will reference this ADR.

---

## ADR-007 — Production sync v2: two-phase chunked protocol over per-call SSH sessions

**Date:** 2026-05-15

**Context:** v1 sync (`scripts/sync_prod_db.py`, ADR-001) failed three consecutive times against the 2026-05-15 production DB (11.6 GB raw → 3.3 GB gz → 4.4 GB base64) with `Error: WebSocket error: tungstenite error`. Diagnostic ladder isolated the failure mode:
- Test A (file-size ping, 350 char bootstrap): rc=0, 2.7s
- Test B (backup + gzip on remote, **no streaming**, 1234 char bootstrap): rc=0, **441.5s** — proves long-idle sessions OK
- Test C (backup + gzip + stream **first 50 MB only**, 1734 char bootstrap): rc=0, 50 MB streamed cleanly — proves small streams OK
- v1 sync (backup + gzip + stream full 4.4 GB): rc=1, tungstenite error

Conclusion: the failure is **total streamed volume per single SSH invocation**, threshold somewhere between 50 MB and 4.4 GB. The 2026-05-10 sync succeeded at ~4.0 GB base64; the 2026-05-15 sync at ~4.4 GB did not.

**Decision:** Replace v1 single-stream with v2 two-phase chunked retrieval.

1. **Remote `scripts/sync_prod_db_remote.py` exposes 4 modes** (via `sys.argv[1:]`):
   - `prepare` (default, no args): backup + gzip → `/tmp/l3sync_snapshot.db.gz`; emit `READY:<size>:<sha256>` on stdout.
   - `chunk <off> <len>`: open prepared gz, seek, stream the slice as base64 framed by `===L3SYNC_PAYLOAD_START===` / `===L3SYNC_PAYLOAD_END===`.
   - `cleanup`: remove prepared gz.
   - `sha256`: re-emit READY from existing prepared gz (for `--resume`).

2. **Local `scripts/sync_prod_db.py` orchestrates:**
   - Phase 1: prepare (one long SSH call, no streaming).
   - Phase 2: chunk loop (one SSH call per 100 MB binary slice). Each chunk → fresh WebSocket → size limit resets.
   - Phase 2.5: SHA-256 verify (catches stitching bugs).
   - Phase 3: decompress + integrity + atomic rename.
   - Phase 4: remote cleanup (best-effort).

3. **Resilience features:**
   - Per-chunk retries: 2 (3 attempts total).
   - `--resume` flag: skip prepare, reuse existing prepared gz on container.
   - `--chunk-mb N` flag: tune chunk size if 100 MB ever proves too aggressive.
   - SHA-256 from prepare prevents silent corruption from chunk-stitching errors.

**Consequences:**
- Sync time mostly unchanged (~10-20 min for ~10 GB DB): prepare phase dominates; the chunk loop adds ~5-10 min of overhead vs the single-stream that no longer works at all.
- Resilient to future DB growth — each chunk capped at 100 MB regardless of total size.
- ~33 SSH invocations per sync (was 1). Higher Railway API call volume but each call is small.
- Bootstrap size kept under 5 KB so it fits the Windows cmd.exe 8191-char limit (was: original v1 was 4124; v2 is 4794 with the larger 4-mode script).
- INV-011 (base64 framing) still holds; INV-011a added documenting the streaming-volume invariant.

**Alternatives considered:**
- **Increase compression level (compresslevel=9 instead of 6)** to shrink the stream: rejected — would maybe save 5-10% gz size; still well above the unknown threshold. Doesn't solve the root issue.
- **Stream via stdin instead of base64 in cmd-line** (`railway ssh "python3 -" < remote_script.py`): considered — would let us embed the script via stdin and pass args differently. Rejected because per-session stdin doesn't help with the streaming-volume cliff; would still need chunking.
- **Add a `/dump` HTTP endpoint to the deployed API:** still rejected for the same reasons as ADR-001 (attack surface).
- **Upload to external object storage** (S3, transfer.sh): rejected — adds credentials and a third-party dependency for what should be a self-contained workflow.
- **Run Railway's `database` backup feature**: rejected — Railway's managed backups exist for their Postgres/MySQL services, not for application-managed SQLite files inside the container's volume.

**Status:** [LANDED] 2026-05-15. Updates: `scripts/sync_prod_db.py`, `scripts/sync_prod_db_remote.py`, `memory/INVARIANTS.md` (INV-011a), `memory/JOURNAL.md` (2026-05-15 entry).

---

## ADR-008 — Adopt SAI (Self-Evolving Actuarial Intelligence) substrate as the primary question-management layer

**Date:** 2026-05-16

**Context:** The 2026-05-15 Phase A investigation falsified 3/3 pre-registered predictions because they cited named entities not present in the queryable corpus. The session's recent-drains analysis showed Layer 3 has structural blind spots (97.6% of drain volume executes through unflagged execution cells; OLI guardrail safe-by-accident; drain USD attribution gap unmoved for months). The SAI cycle output produced 18 structured questions that map directly onto these gaps. Without a persistent question-management substrate, each session re-derives the same gaps and the lessons don't compound.

**Decision:** Adopt the SAI architecture with `memory/questions.yaml` as the canonical question store. Build the SAI Python substrate at `surveillance/sai/` (question_store, question_runner, question_generator, capability_liveness, prediction_registry, adversarial_engine). Wire concrete executable modules to specific questions; the highest-value module (Q-002 approval-spike detector at `surveillance/analytics/approval_spike_detector.py`) is fully implemented and tested in this commit. Other questions ship as skeletons with explicit TODO markers and `implementation_target` paths pointing at where the code will live.

Ranking formula:
```
priority_score = predictive_power*0.30 + actionability*0.30 + failure_reduction*0.30 + uniqueness*0.10
```

Ranking output (verified by `tests/surveillance/test_sai.py::test_top_ranked_is_approval_z_score_question`):
- Tier 1 (≥4.5): Q-002 + Q-014 (approval Z-score, both at 4.90), Q-001 + Q-011 (role lattice, both at 4.70)
- Tier 2 (4.0-4.5): Q-009, Q-013, Q-003, Q-005, Q-012, Q-015, Q-004, Q-016
- Tier 3 (3.0-4.0): Q-006, Q-017, Q-010, Q-007, Q-018
- Tail (<3.0): Q-008

**Consequences:**
- Every new capability/data/pipeline must map back to at least one question in `questions.yaml`. INV-017 (added) enforces this.
- The 7-step reflection loop (LOOP.md) now has a structured output target: failures flow into UNKs (via existing path) AND into new draft questions (via question_generator when implemented).
- Future predictions go through `prediction_verifiability.py` (Q-004 implementation, skeleton this session) before being treated as pre-registered.
- Q-002 (approval spike detector) is a real-time imminent-discharge surveillance capability. Validated against the 0x80b12bd0 May-9 event: produces a Z=130 Tier-1 IMMINENT alert against the exact contract that drained 4,587 victims that day, with 0 false positives on adjacent days (May-8, May-15).
- The architecture is opinionated: questions are first-class; data, models, and pipelines exist to answer questions. Every module ships with its `implementation_target` path declared so the wiring is auditable via `question_runner.py`.

**Alternatives considered:**
- **Build the modules without a question store** (skip Phase 1): rejected — without the structured question registry, the modules become orphan utilities. The question store is what makes the cycle self-evolving.
- **Refactor LOOP.md to fold SAI into the existing loop**: rejected for this commit — SAI introduces a structured question schema that LOOP.md prose can't capture. SAI sits alongside LOOP.md; the two compose. SAI's question_generator is the bridge that consumes LOOP.md SURPRISE blocks.
- **Build all 10 executable modules in this commit**: rejected — scope explosion. Better to ship one fully (Q-002, validated end-to-end) plus skeletons for the rest with clear continuation paths. Q-002 chosen because it has the strongest empirical anchor (the 0x80b12bd0 May-9 event) and the highest priority score.

**Status:** [LANDED] 2026-05-16. Updates:
- `memory/questions.yaml` (18 structured questions)
- `surveillance/sai/` (6 modules: question_store, question_runner, question_generator, capability_liveness, prediction_registry, adversarial_engine + prediction_verifiability)
- `surveillance/analytics/approval_spike_detector.py` (Q-002, fully wired, validated against 2026-05-09 event with Z=130)
- `surveillance/ontology/role_classifier.py` (Q-001 skeleton)
- `tests/surveillance/test_sai.py` (13 new tests, all green)
- `memory/INVARIANTS.md` (INV-017 added: questions drive system structure)
- `memory/JOURNAL.md` (2026-05-16 SAI substrate entry)
