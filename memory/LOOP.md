# Agent Reflection Loop (MANDATORY)

This file is the **session-end protocol**. The agent MUST execute these
seven steps before signing off on any working session. Skipping a step is
itself a logged event — note it explicitly with a reason.

**Operating principle:** the memory system is only valuable if it is
*epistemically integrated*. Append-only memory without an integration step
produces documentation that grows in size but degrades in coherence. The
loop below forces every session to both *contribute* and *integrate*.

**Cost:** ~5-10 minutes at session end.
**Payoff:** every future session begins with a coherent model rather than a
fragmented one.

---

## 1. State Update Check

**Trigger:** Did any system-level fact change?

System-level facts (non-exhaustive):
- Corpus size, watchlist count, deployed surface
- Last-sync timestamp, last-correction number
- Active background tasks, deploy state
- Production token / endpoint / URL
- Schema version (migrations applied)
- File-tree structure (new top-level dirs)

**If yes:** update `memory/STATE.md` in the relevant section.
**If no:** Note explicitly in your session-end journal entry:
"STATE.md unchanged this session."

The "no" case is itself information — a session that touched nothing
state-relevant tells future agents "this was diagnostic-only."

---

## 2. Unknown Detection

**Trigger:** Did I encounter anything unclear, implicit, or assumed?

Heuristic: if you used the words **"probably," "likely," "appears to,"
"UNKNOWN," "assumption," or "I think"** in your session output, those
are candidate unknowns.

Other markers that count:
- I made an inference without primary-source verification
- I trusted a single source where two would be safer
- I noted a code path I didn't read
- I encountered a name/abbreviation I had to guess at
- A library function returned something I didn't fully understand

**If yes:** append to `memory/UNKNOWNS.md` using the canonical schema (see
"UNKNOWNS schema" below).

---

## 3. Decision Extraction

**Trigger:** Did I make a non-trivial choice?

Calibration: **Would a future agent in your seat re-derive this choice from
scratch if it weren't recorded?** If yes → log it.

Threshold examples:
- ❌ DON'T log: variable names, indentation, file ordering, code-style choices
- ✅ DO log: library/protocol choices, error-handling philosophy, schema
  decisions, "trust source X over Y" methodology calls
- ✅ DO log: anything that would be a Correction if reversed

**If yes:** append an ADR to `memory/DECISIONS.md`:

```
# ADR-XXX — <Title>
## Date: <iso>
## Context: <what made this a decision-point>
## Decision: <what was chosen>
## Consequences: <what flows from this>
## Alternatives considered: <what was rejected and why>
```

---

## 4. Invariant Check

**Trigger:** Did I violate, discover, or refine a system invariant?

Three sub-cases:

| Sub-case | Action |
|---|---|
| **Violated** an existing invariant | This is a Correction. Append numbered entry to `reports/correction_log.md`. Also annotate the invariant in `memory/INVARIANTS.md` with the correction reference. |
| **Discovered** a new invariant | Append to `memory/INVARIANTS.md` with format below. |
| **Refined** an existing invariant (made tighter / looser / more precise) | Update the existing entry; record the prior version inline as "Previously: ..." |

Invariant entry format:

```
## INV-XXX — <Statement>
- Rationale: <why this must hold>
- Enforcement: <code path / file:line>
- How tested: <test name or "untested">
- Surfaced: <iso, session-id>
```

Many invariants are *implicit* until first violation. When the reflection
pass asks "what turned out to depend on X this session?" — those are
invariants in disguise. Promote them.

---

## 5. Failure / Surprise Logging

**Trigger:** What did not behave as expected?

Use the canonical SURPRISE format in your `memory/JOURNAL.md` entry:

```
SURPRISE: <one-line description>
- Expected: <what I predicted before observing>
- Observed: <what actually happened>
- Implication: <what this changes about my model of the system>
- Resolution: <fixed | open | logged-as-UNK-XXX>
```

**Why "Expected" is required, not optional:** without recording the prior
prediction, the surprise has no measurable epistemic weight. The "Expected"
field forces you to own the model that just got falsified. A SURPRISE entry
without "Expected" is hindsight bias dressed up as learning.

Examples of session-worthy surprises:
- A detector I expected to fire didn't
- A library function returned different types than its signature claimed
- A "completed" task left residual state
- A test that should have failed didn't
- A previously-working command now errors
- A cache hit appeared where I expected a miss (or vice versa)

---

## 6. System Coherence Check (CRITICAL)

**Trigger:** Always. This step runs even if all others were skipped.

This is the load-bearing step. Steps 1-5 are *transactional* ("this session
added X, Y, Z"). Step 6 is *integrative* ("does the whole still hold
together?"). Without it, the documentation accumulates without correcting
itself.

Answer in writing in your `memory/JOURNAL.md` session entry, with explicit
citations:

### 6a. Does my current understanding of the system still make sense?

List 1-3 anchor claims from `memory/STATE.md` that this session touched.
For each:

```
ANCHOR: "<exact text from STATE.md>" (STATE.md line ~N)
- Status this session: CONFIRMED | REFINED | CONTRADICTED
- Evidence: <file:line | command output | test result>
- Action: <none | updated STATE.md | logged correction>
```

### 6b. Did anything contradict previous assumptions?

If yes:
- Either update `memory/STATE.md` (silent revision is fine for new info)
- OR log a numbered Correction in `reports/correction_log.md` (required for
  retracting a previously-asserted claim)

**Silent contradiction = future debt. Surfaced contradiction = healthy.**

The rule: if you find yourself thinking "wait, that's different from what
I thought" — write down both versions (old + new), then choose which to
canonicalize.

---

## 7. Next Unknown Selection

**Trigger:** Always.

Pick 1-3 UNKNOWNs from `memory/UNKNOWNS.md` to target next session.

Selection heuristic (priority order):

1. **Blockers** — any UNKNOWN currently blocking a non-trivial planned action.
   These get top priority regardless of complexity.

2. **Confidence-recoverable** — LOW-confidence "resolutions" that warrant
   re-verification (see Confidence schema below). Promoting a LOW
   resolution to MEDIUM or HIGH is a research-grade behavior; this is
   the loop's "harden prior claims" surface.

3. **High-impact** — UNKNOWNs where resolution unlocks downstream work
   (e.g., "what does pma/ do" unblocks integration planning).

Avoid the gravitation-toward-easy trap: don't pick an UNKNOWN just because
it's tractable. Use **impact × tractability** mentally, not just tractability.

Write the targets at the end of your session entry:

```
NEXT TARGETS (for session starting after <date>):
- UNK-XXX — <one-line summary>  (why: blocker | LOW-confidence revisit | high-impact)
- UNK-YYY — <one-line summary>  (...)
```

---

## Reflection log (audit trail for the loop itself)

At the end of each session-end pass, append one line to
`memory/REFLECTION_LOG.csv`:

```csv
date_iso,session_id,step1_done,step2_done,step3_done,step4_done,step5_done,step6_done,step7_done,minutes,commits_made,notes
```

Use Y/N for the step columns. Use a short string for `notes` (e.g.,
"step3-skipped-no-decisions").

Why this matters:
- **Steps consistently skipped** → the format isn't working for that step. Revise.
- **Sessions with no UNKNOWNs** → the agent isn't catching them. Tighten Step 2 heuristics.
- **Sessions where Step 6 produced contradictions** → model drift. Look at adjacent sessions to find when the drift started.

The REFLECTION_LOG is the loop's own self-monitoring surface. If it's not
being filled out, the loop has collapsed back into write-only journaling.

---

## Skipped-step accounting (mandatory)

Skipping a step is allowed but must be logged in your session entry:

```
SKIPPED: Step 3 (Decision Extraction)
Reason: information-gathering session; no design choices made.
```

**Rule of three:** if a step is skipped for the same reason in three
sessions running, the loop itself is broken for that step. The step is
either malformed for your workflow or your work is genuinely not exercising
it. Either way: revisit the step's definition.

---

## UNKNOWNS schema (with Confidence)

When logging or resolving an UNKNOWN, use this canonical schema:

```
## UNK-XXX — <One-line question>
- Surfaced: <iso>, by <session-id or 'manual'>
- Category: Architecture | Subsystem | Theoretical | Operational
- Why it matters: <consequence of leaving open — what breaks?>
- Resolution plan: <specific files/code paths to inspect, tests to run>
- Status: OPEN | IN_PROGRESS | RESOLVED | SUPERSEDED
- Confidence (if RESOLVED): HIGH | MEDIUM | LOW
- Resolved at: <iso, if RESOLVED>
- Resolved by: <one-line answer + citation>
- Supersedes: <UNK-YYY, if SUPERSEDED>
```

### Confidence calibration

| Level | Evidence required |
|---|---|
| **HIGH** | Primary-source verified (code read end-to-end, test executed, official doc cited) AND at least one independent cross-reference (second source agrees) |
| **MEDIUM** | Single primary-source verification (code read in relevant section, or test result) plausibility-checked against prior knowledge |
| **LOW** | Single secondary source (label, summary, inference, hearsay) without primary-source verification. **Explicitly flagged for revisit.** |

### Revisit-LOW workflow

At session start, alongside reading STATE.md, scan UNKNOWNS.md for
`RESOLVED + Confidence: LOW` entries. These are *technically resolved* but
*epistemically thin*. Some sessions should be devoted purely to promoting
LOW resolutions to MEDIUM or HIGH — that's how a research system hardens
its model over time.

Example progression:
- 2026-05-09: UNK-007 resolved LOW ("0x4cfe37d2 = Architect alternate" — based on 0.799 behavioral similarity score only)
- 2026-05-10: UNK-007 reaffirmed MEDIUM (OLI tag `Fake_Phishing327625` confirms adversarial; behavior-only → behavior + identity)
- (future): UNK-007 promoted HIGH if a second source corroborates the phishing tag and a third source confirms the Architect-cluster topology

---

## Why this matters (read this when the loop feels like overhead)

Without this loop, the memory system is a write-only journal. Each session
deposits content; nothing reads or reconciles it. The fragmentation
accumulates silently until a future session re-derives something already
known, or worse, contradicts a prior finding without noticing.

With this loop, every session both contributes and integrates. The memory
system becomes a *model of the system* that updates as the system updates —
and that catches its own contradictions.

The bottleneck for this codebase is no longer:
- intelligence (a sufficiently capable agent is available per session)
- code (the codebase is large enough to support what's needed)
- architecture (the architecture is documented in STATE.md)

The bottleneck is **memory discipline**. The loop above is the discipline.
