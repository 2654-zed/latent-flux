"""
RIB Scoring Module — Relational Intelligence Benchmark

Evaluates how well the surveillance system's automated detection
compares against baselines and known ground truth (org_001).

Usage:
    python -m surveillance.rib_scorer
    # or
    from surveillance.rib_scorer import run_self_evaluation
    results = run_self_evaluation("surveillance/data/surveillance.db")
"""

import json
import logging
import random
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


# ---- Ground Truth (same as rib_export.py) ----

ORG_001_NODES = {
    "0xf186cb00e49e18491db5783ff04fae3818102ff7": "treasury",
    "0xe93d64f3fbc352131e79fc5578cbe44b66697f86": "operator",
    "0xfd51e33d44b376ef346d24a130a51035db09c1dc": "operator_2",
    "0xc6962004f452be9203591991d15f6b388e09e8d0": "cashout",
    "0x01989c93890aed05a63d179b03424997075b6acf": "exit_cex",
    "0xfdaf1f1714810f8d88a57c9d551d442c68ace2bb": "laundry",
    "0x27920e8039d2b6e93e36f5d5f53b998e2e631a70": "lp_companion",
    "0x96daa0b8a5499ea9323421ed0cda06b345caab73": "lp_staging",
    "0x360e68faccca8ca495c1b759fd9eee466db9fb32": "treasury_branch",
    "0x3f2cdae910cd13638e38b45881e7a4fc3a9fe320": "victim",
    "0x8c826f795466e39acbff1bb4eeeb759609377ba1": "gas_station",
    "0x51c72848c68a965f66fa7a88855f9f7784502a7f": "defi_exit_channel",
}

ORG_001_EDGES = [
    ("treasury", "operator", "funds"),
    ("treasury", "treasury_branch", "funds"),
    ("treasury", "cashout", "funds"),
    ("gas_station", "operator_2", "provisions_gas"),
    ("treasury_branch", "lp_companion", "funds"),
    ("treasury_branch", "exit_cex", "funds"),
    ("operator", "cashout", "drains_to"),
    ("cashout", "exit_cex", "drains_to"),
    ("cashout", "defi_exit_channel", "obfuscates_through"),
    ("cashout", "lp_staging", "drains_to"),
    ("lp_staging", "cashout", "recycles_to"),
    ("laundry", "exit_cex", "converts_to"),
]


# ---- Data Models ----

@dataclass
class OrganizationPrediction:
    """A predicted organization structure."""
    nodes: Dict[str, str]  # address -> predicted role
    edges: List[Tuple[str, str, str]]  # (src, dst, relationship)
    confidence: float = 0.0

@dataclass
class ScoreCard:
    """Evaluation results for a single prediction."""
    precision: float
    recall: float
    role_accuracy: float
    depth_score: int
    false_positive_rate: float
    ris: float
    classification: str


# ---- Graph Builder ----

def build_surveillance_graph(db_path: str) -> nx.DiGraph:
    """
    Build a temporal directed graph from the surveillance database.
    Nodes = addresses. Edges = transactions/deployments.
    Edge attributes: timestamp, edge_type, weight.
    """
    G = nx.DiGraph()
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row

    # Deployment edges: deployer -> contract
    for r in conn.execute("SELECT deployer_address, contract_address, detection_timestamp FROM contracts"):
        src, dst = r["deployer_address"], r["contract_address"]
        G.add_edge(src, dst, edge_type="contract_deploy", timestamp=r["detection_timestamp"])

    # Transaction event edges: interacting_address -> contract
    tx_count = 0
    for r in conn.execute("SELECT interacting_address, contract_address, timestamp, function_selector, is_reverted FROM transaction_events"):
        src, dst = r["interacting_address"], r["contract_address"]
        # Use a unique key for multigraph behavior in DiGraph (keep last edge per pair)
        G.add_edge(src, dst, edge_type="contract_call", timestamp=r["timestamp"],
                   selector=r["function_selector"], reverted=r["is_reverted"])
        tx_count += 1

    # Funding trail edges: funder -> deployer
    import json as _json
    for r in conn.execute("SELECT deployer_address, funding_trail FROM deployers WHERE funding_trail IS NOT NULL AND funding_trail != ''"):
        try:
            trail = _json.loads(r["funding_trail"])
            funder = trail.get("funder") or trail.get("funding_source")
            if funder:
                G.add_edge(funder.lower(), r["deployer_address"], edge_type="eth_transfer",
                           timestamp=trail.get("timestamp", ""))
        except (ValueError, TypeError):
            pass

    # Add node attributes from deployers table
    for r in conn.execute("SELECT deployer_address, entity_type, total_contracts_deployed, behavioral_score FROM deployers"):
        addr = r["deployer_address"]
        if addr in G:
            G.nodes[addr]["entity_type"] = r["entity_type"]
            G.nodes[addr]["contracts"] = r["total_contracts_deployed"]
            G.nodes[addr]["behavioral_score"] = r["behavioral_score"]

    conn.close()

    if tx_count == 0:
        logger.warning("Zero transaction_events found — graph built from deployment edges only")

    logger.info("Graph built: %d nodes, %d edges (tx_events=%d)", G.number_of_nodes(), G.number_of_edges(), tx_count)
    return G


# ---- Scoring Engine ----

def calculate_ris(precision: float, recall: float, role_accuracy: float,
                  depth_score: int, false_positive_rate: float) -> float:
    """
    Calculate the Relational Intelligence Score (0-100).

    RIS = 100 * H(precision, recall) * depth_multiplier * (1 - fpr) * role_bonus
    """
    # Harmonic mean of precision and recall
    if precision + recall > 0:
        h = 2 * precision * recall / (precision + recall)
    else:
        h = 0.0

    depth_multiplier = min(depth_score / 5.0, 1.0)
    role_bonus = 0.5 + 0.5 * role_accuracy
    fpr_penalty = 1.0 - false_positive_rate

    return round(100.0 * h * depth_multiplier * fpr_penalty * role_bonus, 2)


def classify_ris(score: float) -> str:
    """Classify RIS score into tier."""
    if score <= 20:
        return "No RI"
    elif score <= 50:
        return "Basic"
    elif score <= 80:
        return "Advanced"
    else:
        return "Expert"


def score_prediction(prediction: OrganizationPrediction,
                     truth_nodes: Dict[str, str],
                     truth_edges: List[Tuple[str, str, str]]) -> ScoreCard:
    """
    Score an OrganizationPrediction against ground truth.

    truth_nodes: address -> role
    truth_edges: (src_role, dst_role, relationship)
    """
    # Build role-to-address lookup from ground truth
    role_to_addr = {}
    for addr, role in truth_nodes.items():
        role_to_addr[role] = addr

    # PRECISION: what fraction of predicted nodes are in ground truth
    pred_addrs = set(prediction.nodes.keys())
    true_addrs = set(truth_nodes.keys())
    true_positives = pred_addrs & true_addrs
    precision = len(true_positives) / len(pred_addrs) if pred_addrs else 0.0

    # RECALL: what fraction of ground truth nodes were found
    recall = len(true_positives) / len(true_addrs) if true_addrs else 0.0

    # ROLE ACCURACY: of correctly identified nodes, how many got the right role
    correct_roles = 0
    for addr in true_positives:
        if prediction.nodes.get(addr) == truth_nodes.get(addr):
            correct_roles += 1
    role_accuracy = correct_roles / len(true_positives) if true_positives else 0.0

    # DEPTH SCORE: max hop distance in predicted edges that matches truth
    # Convert truth edges from roles to addresses for comparison
    truth_addr_edges = set()
    for src_role, dst_role, rel in truth_edges:
        src_addr = role_to_addr.get(src_role)
        dst_addr = role_to_addr.get(dst_role)
        if src_addr and dst_addr:
            truth_addr_edges.add((src_addr, dst_addr))

    pred_edge_set = set((s, d) for s, d, _ in prediction.edges)
    matched_edges = pred_edge_set & truth_addr_edges

    # Depth = longest chain of consecutive matched edges from treasury
    depth = 0
    if matched_edges:
        # BFS from treasury through matched edges
        treasury = role_to_addr.get("treasury")
        if treasury:
            visited = set()
            queue = [(treasury, 0)]
            while queue:
                node, d = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                depth = max(depth, d)
                for src, dst in matched_edges:
                    if src == node and dst not in visited:
                        queue.append((dst, d + 1))

    # FALSE POSITIVE RATE
    false_positives = pred_addrs - true_addrs
    total_predicted = len(pred_addrs)
    fpr = len(false_positives) / total_predicted if total_predicted > 0 else 0.0

    ris = calculate_ris(precision, recall, role_accuracy, depth, fpr)
    classification = classify_ris(ris)

    return ScoreCard(
        precision=round(precision, 4),
        recall=round(recall, 4),
        role_accuracy=round(role_accuracy, 4),
        depth_score=depth,
        false_positive_rate=round(fpr, 4),
        ris=ris,
        classification=classification,
    )


# ---- Baselines ----

ROLES = ["treasury", "operator", "cashout", "exit_cex", "laundry", "lp_staging", "victim"]

class RandomBaseline:
    """Pick random nodes, assign random roles."""
    def __init__(self, seed: int = 42):
        self.seed = seed

    def predict(self, G: nx.DiGraph, n_org_nodes: int = 10) -> OrganizationPrediction:
        rng = random.Random(self.seed)
        nodes = list(G.nodes())
        if len(nodes) < n_org_nodes:
            n_org_nodes = len(nodes)
        selected = rng.sample(nodes, n_org_nodes)
        node_roles = {addr: rng.choice(ROLES) for addr in selected}
        edges = []
        for i in range(min(len(selected) - 1, 5)):
            edges.append((selected[i], selected[i+1], "predicted_link"))
        return OrganizationPrediction(nodes=node_roles, edges=edges, confidence=0.1)


class DegreeCentralityBaseline:
    """Rank by degree centrality. Highest out-degree = treasury, highest in-degree = exit."""
    def predict(self, G: nx.DiGraph, n_org_nodes: int = 10) -> OrganizationPrediction:
        out_deg = sorted(G.out_degree(), key=lambda x: -x[1])
        in_deg = sorted(G.in_degree(), key=lambda x: -x[1])

        nodes = {}
        edges = []

        # Highest out-degree = treasury
        if out_deg:
            nodes[out_deg[0][0]] = "treasury"
        # Next highest out-degree = operator
        for addr, d in out_deg[1:4]:
            if addr not in nodes:
                nodes[addr] = "operator"
                if out_deg:
                    edges.append((out_deg[0][0], addr, "funds"))
        # Highest in-degree = exit
        for addr, d in in_deg[:3]:
            if addr not in nodes:
                nodes[addr] = "exit_cex"
        # Nodes with high both = cashout
        out_set = {a for a, _ in out_deg[:20]}
        in_set = {a for a, _ in in_deg[:20]}
        both = out_set & in_set
        for addr in list(both)[:2]:
            if addr not in nodes:
                nodes[addr] = "cashout"
                # Add edges from operators to cashout
                for op_addr, op_role in nodes.items():
                    if op_role == "operator":
                        edges.append((op_addr, addr, "drains_to"))

        # Fill remaining
        for addr, d in out_deg:
            if len(nodes) >= n_org_nodes:
                break
            if addr not in nodes:
                nodes[addr] = "lp_staging"

        return OrganizationPrediction(nodes=nodes, edges=edges, confidence=0.3)


class PageRankBaseline:
    """Use PageRank for apex detection, BFS outward for org structure."""
    def predict(self, G: nx.DiGraph, n_org_nodes: int = 10) -> OrganizationPrediction:
        try:
            pr = nx.pagerank(G, max_iter=100)
        except Exception:
            pr = {n: 1.0/G.number_of_nodes() for n in G.nodes()}

        ranked = sorted(pr.items(), key=lambda x: -x[1])
        nodes = {}
        edges = []

        # Top PageRank = treasury
        if ranked:
            nodes[ranked[0][0]] = "treasury"

        # BFS from treasury to find connected high-PR nodes
        treasury = ranked[0][0] if ranked else None
        if treasury:
            for neighbor in G.successors(treasury):
                if len(nodes) >= n_org_nodes:
                    break
                if neighbor not in nodes:
                    # Assign role based on PageRank rank
                    rank = next((i for i, (a, _) in enumerate(ranked) if a == neighbor), 999)
                    if rank < 50:
                        nodes[neighbor] = "operator"
                    else:
                        nodes[neighbor] = "lp_staging"
                    edges.append((treasury, neighbor, "funds"))

            # Second hop
            for op_addr in [a for a, r in nodes.items() if r == "operator"]:
                for neighbor in G.successors(op_addr):
                    if len(nodes) >= n_org_nodes:
                        break
                    if neighbor not in nodes:
                        nodes[neighbor] = "cashout"
                        edges.append((op_addr, neighbor, "drains_to"))

        # Fill remaining
        for addr, score in ranked:
            if len(nodes) >= n_org_nodes:
                break
            if addr not in nodes:
                nodes[addr] = "exit_cex"

        return OrganizationPrediction(nodes=nodes, edges=edges, confidence=0.4)


class LouvainBaseline:
    """Community detection via Louvain, assign roles within best community."""
    def predict(self, G: nx.DiGraph, n_org_nodes: int = 10) -> OrganizationPrediction:
        # Louvain works on undirected
        UG = G.to_undirected()
        try:
            communities = nx.community.louvain_communities(UG, seed=42)
        except Exception:
            # Fallback
            return RandomBaseline(seed=99).predict(G, n_org_nodes)

        if not communities:
            return RandomBaseline(seed=99).predict(G, n_org_nodes)

        # Find community with most deployer->contract edges (most "interesting")
        best_community = max(communities, key=len)

        # Within this community, rank by degree
        sub = G.subgraph(best_community)
        out_deg = sorted(sub.out_degree(), key=lambda x: -x[1])
        in_deg = sorted(sub.in_degree(), key=lambda x: -x[1])

        nodes = {}
        edges = []

        role_queue = ["treasury", "operator", "operator_2", "cashout", "exit_cex",
                       "lp_staging", "laundry", "lp_companion", "treasury_branch", "victim"]

        # Assign by out-degree rank
        for i, (addr, d) in enumerate(out_deg):
            if i >= len(role_queue) or len(nodes) >= n_org_nodes:
                break
            nodes[addr] = role_queue[i]

        # Build edges between consecutive role assignments
        assigned = list(nodes.keys())
        for i in range(len(assigned) - 1):
            if G.has_edge(assigned[i], assigned[i+1]):
                edges.append((assigned[i], assigned[i+1], "predicted_link"))

        return OrganizationPrediction(nodes=nodes, edges=edges, confidence=0.5)


# ---- System Self-Evaluation ----

def extract_system_prediction(db_path: str) -> OrganizationPrediction:
    """
    Extract what the surveillance system found automatically.
    Reads entity_type labels from deployers table and builds a prediction.
    """
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row

    nodes = {}
    edges = []

    # Get all deployers with non-unknown entity_type
    for r in conn.execute("""
        SELECT deployer_address, entity_type FROM deployers
        WHERE entity_type IS NOT NULL AND entity_type NOT IN ('unknown', 'mev_bot_factory', 'protocol',
              'ground_truth', 'known_attacker', 'trap_inventory_operator')
    """):
        nodes[r["deployer_address"]] = r["entity_type"]

    # Build edges from funding_trail (funder -> deployer)
    if nodes:
        for r in conn.execute("SELECT deployer_address, funding_trail FROM deployers WHERE funding_trail IS NOT NULL AND deployer_address IN (%s)" %
                              ",".join("?" * len(nodes)), list(nodes.keys())):
            try:
                trail = json.loads(r["funding_trail"])
                funder = (trail.get("funder") or trail.get("funding_source") or "").lower()
                if funder and funder in nodes:
                    edges.append((funder, r["deployer_address"], "funds"))
            except (ValueError, TypeError):
                pass

    # Build edges from pattern_matches
    for r in conn.execute("SELECT address, evidence FROM pattern_matches WHERE pattern_type = 'treasury_pattern'"):
        try:
            ev = json.loads(r["evidence"])
            treasury = r["address"].lower()
            for funded in ev.get("funded_addresses", []):
                funded = funded.lower()
                if funded in nodes:
                    edges.append((treasury, funded, "funds"))
        except (ValueError, TypeError):
            pass

    conn.close()
    return OrganizationPrediction(nodes=nodes, edges=edges, confidence=0.8)


def run_self_evaluation(db_path: str) -> Dict:
    """
    Run full self-evaluation: build graph, run all baselines, score everything.
    Returns a comparison dict.
    """
    db_path_str = str(db_path)

    # Build graph
    G = build_surveillance_graph(db_path_str)

    # Ground truth
    truth_nodes = {k.lower(): v for k, v in ORG_001_NODES.items()}
    truth_edges = ORG_001_EDGES

    results = {
        "graph": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()},
        "baselines": {},
        "system": None,
    }

    # Run baselines
    baselines = [
        ("random", RandomBaseline(seed=42)),
        ("degree_centrality", DegreeCentralityBaseline()),
        ("pagerank", PageRankBaseline()),
        ("louvain", LouvainBaseline()),
    ]

    for name, baseline in baselines:
        try:
            pred = baseline.predict(G, n_org_nodes=12)
            # Lowercase all predicted addresses for comparison
            pred.nodes = {k.lower(): v for k, v in pred.nodes.items()}
            pred.edges = [(s.lower(), d.lower(), r) for s, d, r in pred.edges]
            card = score_prediction(pred, truth_nodes, truth_edges)
            results["baselines"][name] = {
                "precision": card.precision,
                "recall": card.recall,
                "role_accuracy": card.role_accuracy,
                "depth_score": card.depth_score,
                "false_positive_rate": card.false_positive_rate,
                "ris": card.ris,
                "classification": card.classification,
                "predicted_nodes": len(pred.nodes),
                "predicted_edges": len(pred.edges),
            }
        except Exception as e:
            logger.error("Baseline %s failed: %s", name, e)
            results["baselines"][name] = {"error": str(e)}

    # Score surveillance system's own detection
    try:
        sys_pred = extract_system_prediction(db_path_str)
        sys_pred.nodes = {k.lower(): v for k, v in sys_pred.nodes.items()}
        sys_pred.edges = [(s.lower(), d.lower(), r) for s, d, r in sys_pred.edges]
        card = score_prediction(sys_pred, truth_nodes, truth_edges)
        results["system"] = {
            "precision": card.precision,
            "recall": card.recall,
            "role_accuracy": card.role_accuracy,
            "depth_score": card.depth_score,
            "false_positive_rate": card.false_positive_rate,
            "ris": card.ris,
            "classification": card.classification,
            "predicted_nodes": len(sys_pred.nodes),
            "predicted_edges": len(sys_pred.edges),
        }
    except Exception as e:
        logger.error("System evaluation failed: %s", e)
        results["system"] = {"error": str(e)}

    return results


def print_comparison(results: Dict) -> None:
    """Print a human-readable comparison table."""
    print("=" * 80)
    print("RIB SELF-EVALUATION — Relational Intelligence Benchmark")
    print("=" * 80)
    print(f"Graph: {results['graph']['nodes']} nodes, {results['graph']['edges']} edges")
    print()

    header = f"{'Method':<22} {'Prec':>6} {'Rec':>6} {'Role':>6} {'Depth':>5} {'FPR':>6} {'RIS':>6} {'Class':<10}"
    print(header)
    print("-" * len(header))

    for name, data in results["baselines"].items():
        if "error" in data:
            print(f"{name:<22} ERROR: {data['error']}")
        else:
            print(f"{name:<22} {data['precision']:>6.3f} {data['recall']:>6.3f} {data['role_accuracy']:>6.3f} {data['depth_score']:>5d} {data['false_positive_rate']:>6.3f} {data['ris']:>6.1f} {data['classification']:<10}")

    print("-" * len(header))
    sys_data = results["system"]
    if sys_data and "error" not in sys_data:
        print(f"{'SURVEILLANCE SYSTEM':<22} {sys_data['precision']:>6.3f} {sys_data['recall']:>6.3f} {sys_data['role_accuracy']:>6.3f} {sys_data['depth_score']:>5d} {sys_data['false_positive_rate']:>6.3f} {sys_data['ris']:>6.1f} {sys_data['classification']:<10}")
    else:
        print(f"{'SURVEILLANCE SYSTEM':<22} ERROR: {sys_data}")
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_self_evaluation("surveillance/data/surveillance.db")
    print_comparison(results)

    # Verify edge cases
    assert calculate_ris(1.0, 1.0, 1.0, 5, 0.0) == 100.0, "Perfect score should be 100"
    assert calculate_ris(0.0, 0.0, 0.0, 0, 1.0) == 0.0, "Worst score should be 0"
    print("Edge case assertions passed.")
