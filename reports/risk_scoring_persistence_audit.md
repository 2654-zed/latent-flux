# `risk_scores` Table Persistence — Audit

**Date:** 2026-04-17
**Scope:** Reconcile CLAUDE.md's schema documentation with the actual behavior of `risk_scoring.py` and the `/api/v1/risk` endpoint.
**Output:** Findings + proposed CLAUDE.md correction. No code changes.

---

## Answer in one paragraph

`risk_scoring.py` is **pure live computation**. It does not create, write to, or maintain any table. The module's own docstring (line 12) says explicitly: *"Read-only analytics on existing surveillance data. Creates no new tables."* The `risk_scores` row in CLAUDE.md's schema table — listed with `Rows = Computed` and the note *"Stored potential scoring output"* — describes a table that does not exist and has never existed in this codebase. The API endpoint `/api/v1/risk/{chain}/{address}` invokes `surveillance.risk_scoring.score_contract(conn, addr)` on every request (`web/api_v1.py:818`), computing the score from source tables in real time.

---

## Evidence

### 1. No persistence in `risk_scoring.py`

Grep for write statements:
```
$ grep -nE "INSERT|CREATE\\s+TABLE|UPDATE\\s+\\w+\\s+SET|DELETE\\s+FROM" surveillance/risk_scoring.py
# (no matches)
```

The module is 1008 lines. Every `conn.execute` it issues is a `SELECT`. It never writes a row.

### 2. Table does not exist in the live DB

```python
>>> conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='risk_scores'").fetchone()
None
```

This is from the local 1.77 GB snapshot at `surveillance/data/surveillance.db` — the superset DB that preserves rows Railway has deleted. If `risk_scores` had ever been written there, it would still exist here. It does not.

### 3. API uses live computation

`web/api_v1.py:817–818`:

```python
from surveillance.risk_scoring import score_contract
risk = score_contract(conn, addr)
```

No lookup in a `risk_scores` cache table. No fallback path to "read precomputed if present, else compute." Just direct invocation per-request.

### 4. CLI flags do not match CLAUDE.md

CLAUDE.md's *Running Common Operations* block (line 325) lists:
```
python -m surveillance.risk_scoring --score-all
```

The actual module accepts `--address`, `--top N`, `--family <id>`, `--db`, `--json`. There is **no `--score-all` flag**. Running the documented command produces:
```
usage: risk_scoring.py [-h] [--address ADDRESS] [--top TOP] [--family FAMILY] [--db DB] [--json]
risk_scoring.py: error: unrecognized arguments: --score-all
```

Three things are off: a command that doesn't parse, a table that doesn't exist, and a "score-all" semantic that has no implementation.

---

## What `score_contract` actually does

Per-call cost measured against the local snapshot for five representative contracts:

```
0x4f835c9f... (CRITICAL, 62.5)  852.5 ms
0x471a4999... (LOW, 4.0)        115.8 ms
0xe9a4737f... (LOW, 5.0)         77.8 ms
0x95fef955... (MEDIUM, 14.0)     74.1 ms
0xbb2dc668... (HIGH, 26.0)       67.5 ms

median: 77.8 ms   max: 852.5 ms   n=5
```

Live computation reads `contracts`, `deployers`, `transaction_events`, `trap_events`, `approval_watchlist`, `bytecode_cache`, `approval_events`, `entity_classification`, and sometimes `bytecode_family_members`. The high outlier is likely a contract with many related rows — a heavy contract joins many children.

This puts `/api/v1/risk/{chain}/{addr}` at **p50 ~80ms, p99 possibly >1s**. At single-request screening load this is fine; at bulk-screening load (a customer running 1000 addresses) it's 80 seconds of sequential compute. A precomputed `risk_scores` table would be the architectural answer, but one has never been built.

---

## Why CLAUDE.md says otherwise (hypothesis)

CLAUDE.md's schema block reads like a target-state document: it names tables the system was designed to have, and several of those were built, but `risk_scores` is the one where the producer was never written. Two signals support that read:
- The deck `Stored_Potential_Risk_Model.pptx` (2026-04) presents the tier distribution as *"123,985 contracts. Daily review feed"* — language that implies a persisted, ranked table. The underlying compute is live, but the pitch treats it as a materialized product.
- The missing `--score-all` CLI flag is the exact entry-point a producer job would need. Its absence is consistent with "table and job both planned, neither shipped."

This is not inconsistency-for-free. Missing a persisted `risk_scores` means:
- **No freshness header** available for the API (nothing to stamp with `computed_at`).
- **No bulk-query path.** Customers who want "every CRITICAL contract on Base right now" must hit N endpoints.
- **No longitudinal tracking.** Whether a contract's score moved UP or DOWN over time is not recoverable — only the current value, recomputed on demand.
- **No tier-distribution monitoring.** The "tier boundaries" table in CLAUDE.md (§The Risk Scoring Model) implies a distribution the system can report on. The only way to measure it today is to score every contract sequentially.

None of these are correctness bugs. They are product-surface bugs.

---

## Two consistent stories — pick one

**Story A: the live-compute model is the design. CLAUDE.md is wrong.**

- Remove `risk_scores` from the schema table.
- Remove `python -m surveillance.risk_scoring --score-all` from *Running Common Operations*.
- Add a note to the risk-scoring section: *"Scores are computed live per API call. No persistence. p50 ~80ms."*
- Accept the bulk-query and longitudinal limitations as the current trade-off.

**Story B: the documented design is the intent. The code is incomplete.**

- Implement `--score-all` as a producer that writes to a new `risk_scores` table. Schema: `(contract_address, chain, score, tier, stored_potential, volatility, realized_value, components_json, computed_at)`.
- Add API read path: `/api/v1/risk/{chain}/{addr}` checks `risk_scores` first; falls back to live compute only if no row exists or row is older than N days.
- Wire the producer into the scheduler (same scheduler problem as Wave 1's Class B finding — this is the same bug at a different table).
- Add `computed_at` to the API response.

**Story A is honest about today's state. Story B matches the pitch but is net-new work.**

Choosing Story A costs a couple of paragraphs of CLAUDE.md edits. Story B costs a producer, a migration, a scheduler, and a policy on how stale a persisted score is allowed to be before the API recomputes.

---

## Proposed CLAUDE.md correction (if Story A)

Remove line `| \`risk_scores\` | Computed | Stored potential scoring output |` from the schema table (around line 130).

Change the *Running Common Operations* line from:
```
python -m surveillance.risk_scoring --score-all
```
to:
```
python -m surveillance.risk_scoring --top 100        # highest stored-potential contracts
python -m surveillance.risk_scoring --address 0x...  # score one contract
```

Add a new line to §*The Risk Scoring Model* (around line 162):
> *"Scores are computed live per API call against source tables (contracts, deployers, transaction_events, trap_events, approval_events, bytecode_cache, entity_classification). No persistence. Median latency ~80ms per contract; p99 around 1s for contracts with heavy transaction history. For bulk screening, use `--top N` CLI for ranked output."*

---

## Open questions for the user

1. **Story A or Story B?** If A, ship the CLAUDE.md edit now and close the discrepancy. If B, scope the producer work as a separate task — ideally aligned with the Wave 2 scheduler audit, since `risk_scores` would become the sixth producer-recompute table in the same family.

2. **If Story A, does the pitch (Stored_Potential_Risk_Model.pptx) need revision?** The deck's "123,985 contracts… daily review feed" language reads like a persisted product. Live-compute is defensible but the framing needs to match.

3. **Latency policy for `/api/v1/risk`.** If live-compute stays, the API should probably enforce a timeout (say 2s) and return a partial or cached response on the tail. Currently no such safety exists.
