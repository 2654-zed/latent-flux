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
