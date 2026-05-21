"""SBM-style biclustering on the funder→deployer bipartite graph.

Question
--------
Are there latent organizational structures in the funder→deployer
bipartite graph that the existing `auto_funder_tracer` + entity_classification
pipeline does not surface? Specifically: do funder communities exist whose
member deployers are statistically over-enriched for confirmed-tier predator
contracts, beyond what the per-funder degree alone would predict?

Methodology
-----------
- "Bipartite SBM" via spectral biclustering (Newman-Girvan style adapted to
  bipartite graphs, no sklearn dep).
- Edges: (funder, deployer) pairs where `deployers.funding_trail.funder = F`
  and the deployer is F's child.
- Pre-filter: drop funders with degree < 5 (the long tail of one-shot
  funders is noise and inflates dimensionality without adding signal). Drop
  deployers with no contracts in the corpus (NULL `total_contracts_deployed`).
- Build sparse adjacency A (funders × deployers). Compute SVD truncated to
  K leading components. Cluster funders and deployers separately via
  k-means on the SVD-projected coordinates.
- For each (funder-cluster, deployer-cluster) BLOCK, compute:
    - block density: edges / (|F|×|D|)
    - confirmed-tier enrichment: P(confirmed | deployer in block) vs corpus rate
    - per-chain composition
- Permutation test: compare observed block enrichment vs N=1000 shuffled
  labels to compute p-value on each block's confirmed-tier rate.

Hypothesis (H1)
---------------
Some block has confirmed-tier deployer ratio significantly above corpus
baseline, indicating a funder-cluster that systematically incubates
predators beyond what individual funders' degree would predict.

If no block exceeds the permutation null at alpha=0.05/K (Bonferroni),
the result is null: the bipartite graph does NOT have latent
predator-incubator structure beyond what individual funder behavior
already explains.

CLI
---
    python -m surveillance.analytics.sbm_funder_deployer
    python -m surveillance.analytics.sbm_funder_deployer --K 8
    python -m surveillance.analytics.sbm_funder_deployer --min-degree 10
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds
    from scipy.cluster.vq import kmeans2
except ImportError as e:
    sys.stderr.write(f"numpy + scipy required: {e}\n")
    raise

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "surveillance.db"
RNG = np.random.default_rng(20260519)


def load_bipartite(conn: sqlite3.Connection, min_funder_degree: int) -> tuple:
    """Returns (funders, deployers, edges, deployer_tier, deployer_chain).

    `funders` and `deployers` are lists; `edges` is list of (fidx, didx)."""
    sys.stderr.write("  loading funder->deployer edges...\n")
    rows = conn.execute(
        """
        SELECT json_extract(funding_trail, '$.funder') AS funder,
               deployer_address,
               chain
        FROM deployers
        WHERE funding_trail IS NOT NULL AND funding_trail != ''
        """
    ).fetchall()

    # Build funder -> degree map first
    funder_deg = Counter()
    for funder, _, _ in rows:
        if funder:
            funder_deg[funder] += 1

    # Filter funders by degree threshold
    keep_funders = {f for f, d in funder_deg.items() if d >= min_funder_degree}
    sys.stderr.write(f"  {len(funder_deg):,} unique funders; "
                     f"{len(keep_funders):,} survive deg >= {min_funder_degree}\n")

    funders_list = sorted(keep_funders)
    fidx = {f: i for i, f in enumerate(funders_list)}

    # Collect deployers that appear in kept-funder edges
    deployer_set = set()
    raw_edges = []
    for funder, deployer, chain in rows:
        if funder in fidx:
            raw_edges.append((funder, deployer, chain))
            deployer_set.add(deployer)
    deployers_list = sorted(deployer_set)
    didx = {d: i for i, d in enumerate(deployers_list)}

    sys.stderr.write(f"  {len(deployers_list):,} deployers in retained bipartite graph\n")
    sys.stderr.write(f"  {len(raw_edges):,} edges\n")

    # Need deployer→contract aggregation for tier enrichment.
    # We classify deployer as "predator-host" if ANY of their contracts is
    # confirmed-tier, else "host-of-suspected" if any contract is suspected, etc.
    sys.stderr.write("  loading deployer tier (via contracts table)...\n")
    deployer_tier: dict[str, str] = {}
    deployer_chain: dict[str, str] = {}
    tier_rows = conn.execute(
        """
        SELECT c.deployer_address,
               SUM(CASE WHEN c.confidence_tier='confirmed' THEN 1 ELSE 0 END) AS n_conf,
               SUM(CASE WHEN c.confidence_tier='suspected' THEN 1 ELSE 0 END) AS n_susp,
               COUNT(*) AS n_total,
               MAX(c.chain) AS chain
        FROM contracts c
        WHERE c.deployer_address IN (
            SELECT DISTINCT deployer_address FROM deployers
            WHERE funding_trail IS NOT NULL AND funding_trail != ''
        )
        GROUP BY c.deployer_address
        """
    ).fetchall()
    for addr, n_conf, n_susp, n_total, chain in tier_rows:
        if n_conf and n_conf > 0:
            deployer_tier[addr] = "confirmed"
        elif n_susp and n_susp > 0:
            deployer_tier[addr] = "suspected"
        else:
            deployer_tier[addr] = "clean"
        deployer_chain[addr] = chain

    # Build edge list of (fidx, didx) for those that survived both filters
    edges = []
    for funder, deployer, chain in raw_edges:
        if deployer in didx:
            edges.append((fidx[funder], didx[deployer]))

    return funders_list, deployers_list, edges, deployer_tier, deployer_chain


def spectral_bicluster(F: int, D: int, edges: list, K: int) -> tuple:
    """SVD-based bipartite biclustering. Returns (funder_labels, deployer_labels)."""
    sys.stderr.write(f"  building sparse adjacency ({F:,} x {D:,})...\n")
    rows = np.fromiter((f for f, _ in edges), dtype=np.int32, count=len(edges))
    cols = np.fromiter((d for _, d in edges), dtype=np.int32, count=len(edges))
    data = np.ones(len(edges), dtype=np.float32)
    A = csr_matrix((data, (rows, cols)), shape=(F, D))

    # Normalized adjacency (Dhillon biclustering normalization)
    row_sums = np.asarray(A.sum(axis=1)).ravel()
    col_sums = np.asarray(A.sum(axis=0)).ravel()
    row_sums[row_sums == 0] = 1.0
    col_sums[col_sums == 0] = 1.0
    Dr_inv_sqrt = 1.0 / np.sqrt(row_sums)
    Dc_inv_sqrt = 1.0 / np.sqrt(col_sums)
    # A_n = Dr^{-1/2} A Dc^{-1/2}
    A_n = A.multiply(Dr_inv_sqrt[:, None]).multiply(Dc_inv_sqrt[None, :])
    A_n = A_n.tocsr()

    sys.stderr.write(f"  computing SVD (k={K})...\n")
    # svds returns ascending order, so take last K
    k_svd = min(K, min(F, D) - 1)
    U, S, Vt = svds(A_n, k=k_svd)
    # Reorder descending
    order = np.argsort(-S)
    U = U[:, order]
    Vt = Vt[order, :]

    # Embed funders and deployers into shared K-dim space
    funder_emb = U * Dr_inv_sqrt[:, None]
    deployer_emb = Vt.T * Dc_inv_sqrt[:, None]

    sys.stderr.write(f"  k-means clustering into K={K} blocks...\n")
    # Stack and cluster jointly to learn aligned block IDs
    combined = np.vstack([funder_emb, deployer_emb])
    # Normalize rows to unit length for stable k-means on directions
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    combined_n = combined / norms
    centroids, labels = kmeans2(combined_n, K, seed=20260519, minit="++", iter=100)

    funder_labels = labels[:F]
    deployer_labels = labels[F:]
    return funder_labels, deployer_labels, A


def permutation_pvalue(observed: float, n_block: int, n_pop_confirmed: int,
                      n_pop_total: int, n_iter: int = 1000) -> float:
    """Two-sided permutation p-value for block confirmed-rate."""
    if n_block == 0:
        return 1.0
    # Sample n_block deployers without replacement from the corpus; count confirmed
    null_rates = np.empty(n_iter)
    # Construct boolean array of confirmed (1) / not-confirmed (0)
    pop = np.zeros(n_pop_total, dtype=np.int8)
    pop[:n_pop_confirmed] = 1
    for i in range(n_iter):
        sample_idx = RNG.choice(n_pop_total, size=n_block, replace=False)
        null_rates[i] = pop[sample_idx].mean()
    obs_rate = observed
    delta_obs = abs(obs_rate - pop.mean())
    delta_null = np.abs(null_rates - pop.mean())
    return float((delta_null >= delta_obs).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--K", type=int, default=8, help="number of blocks (default 8)")
    ap.add_argument("--min-degree", type=int, default=5,
                    help="minimum funder degree to include (default 5)")
    ap.add_argument("--perm-iter", type=int, default=1000,
                    help="permutation test iterations (default 1000)")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)

    funders, deployers, edges, deployer_tier, deployer_chain = load_bipartite(
        conn, args.min_degree
    )
    conn.close()

    F, D = len(funders), len(deployers)
    if F == 0 or D == 0 or not edges:
        sys.stderr.write("No data after filtering\n")
        return 1

    funder_labels, deployer_labels, A = spectral_bicluster(F, D, edges, args.K)

    # Population baseline
    pop_confirmed = sum(1 for d in deployers if deployer_tier.get(d) == "confirmed")
    pop_total = len(deployers)
    baseline_rate = pop_confirmed / pop_total

    print()
    print("=" * 78)
    print(f"SBM BIPARTITE BICLUSTERING (K={args.K}, min_funder_degree={args.min_degree})")
    print("=" * 78)
    print(f"  Funders included:  {F:,}")
    print(f"  Deployers in BCC:  {D:,}")
    print(f"  Bipartite edges:   {len(edges):,}")
    print(f"  Baseline confirmed-rate: {baseline_rate:.4f} "
          f"({pop_confirmed:,}/{pop_total:,})")
    print()

    print("=" * 78)
    print("CLUSTER COMPOSITION")
    print("=" * 78)
    print(f"  {'cluster':>7s}  {'N_funders':>10s}  {'N_deployers':>12s}  "
          f"{'edges':>8s}  {'confirmed':>10s}  {'rate':>7s}  "
          f"{'enrich':>7s}  {'p-perm':>8s}")
    print("  " + "-" * 84)

    # Bonferroni alpha
    alpha_bonf = 0.05 / args.K

    results = []
    for k in range(args.K):
        f_in_k = np.where(funder_labels == k)[0]
        d_in_k = np.where(deployer_labels == k)[0]
        n_f = len(f_in_k)
        n_d = len(d_in_k)

        # Edges within block k
        if n_f and n_d:
            sub = A[f_in_k][:, d_in_k]
            n_edges = sub.nnz
        else:
            n_edges = 0

        # Confirmed-tier deployers within block k
        n_conf_k = sum(1 for di in d_in_k if deployer_tier.get(deployers[di]) == "confirmed")
        rate_k = n_conf_k / n_d if n_d else 0.0
        enrich = rate_k / baseline_rate if baseline_rate else 0.0

        # Permutation test
        if n_d >= 10:
            p = permutation_pvalue(rate_k, n_d, pop_confirmed, pop_total, args.perm_iter)
        else:
            p = float("nan")

        sig = ""
        if not math.isnan(p):
            if p < alpha_bonf:
                sig = " ***"
            elif p < 0.05:
                sig = " *"

        print(f"  {k:>7d}  {n_f:>10,}  {n_d:>12,}  {n_edges:>8,}  "
              f"{n_conf_k:>10,}  {rate_k:>7.4f}  {enrich:>7.2f}x  {p:>8.4f}{sig}")
        results.append({
            "cluster": k, "n_funders": n_f, "n_deployers": n_d,
            "n_edges": n_edges, "n_confirmed": n_conf_k, "rate": rate_k,
            "enrich": enrich, "p_perm": p,
        })

    print()
    print(f"  Bonferroni alpha = 0.05 / {args.K} = {alpha_bonf:.5f}")
    print()

    # Detailed dump of significant clusters
    print("=" * 78)
    print("BLOCK DETAIL — clusters with p < 0.05/K (Bonferroni)")
    print("=" * 78)
    sig_clusters = [r for r in results if not math.isnan(r["p_perm"])
                    and r["p_perm"] < alpha_bonf and r["n_deployers"] >= 10]
    if not sig_clusters:
        print("  No clusters pass Bonferroni threshold.")
    for r in sig_clusters:
        k = r["cluster"]
        f_in_k = np.where(funder_labels == k)[0]
        d_in_k = np.where(deployer_labels == k)[0]
        # Top 5 funders by degree within block
        funder_deg = []
        sub = A[f_in_k][:, d_in_k]
        deg_sums = np.asarray(sub.sum(axis=1)).ravel()
        order = np.argsort(-deg_sums)[:5]
        print(f"\n  Cluster {k}: {r['n_funders']} funders, {r['n_deployers']} deployers, "
              f"{r['n_edges']:,} edges, confirmed-rate {r['rate']:.4f} "
              f"({r['enrich']:.2f}x baseline, p={r['p_perm']:.4f})")
        print(f"    Top funders by within-block degree:")
        for i in order:
            fi = f_in_k[i]
            print(f"      {funders[fi]}  deg={int(deg_sums[i])}")
        # Chain breakdown of deployers
        chain_counts = Counter(deployer_chain.get(deployers[di], "?") for di in d_in_k)
        chains_str = ", ".join(f"{c}:{n}" for c, n in chain_counts.most_common())
        print(f"    Chain mix: {chains_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
