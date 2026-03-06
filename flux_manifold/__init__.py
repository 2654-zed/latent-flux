"""Latent Flux Interpreter – all 8 primitives for transformer-native computation."""

# Core FluxManifold (⟼)
from flux_manifold.core import flux_flow, flux_flow_traced, flux_flow_batch, flux_flow_traced_batch
from flux_manifold.flows import normalize_flow, sin_flow, damped_flow, adaptive_flow

# Baselines
from flux_manifold.baselines import random_walk, gradient_descent, static_baseline

# Superposition Tensor (∑_ψ)
from flux_manifold.superposition import SuperpositionTensor

# DriftEquivalence (≅)
from flux_manifold.drift_equivalence import DriftEquivalence

# Commitment Sink (↓!)
from flux_manifold.commitment_sink import CommitmentSink

# Abstraction Cascade (⇑)
from flux_manifold.abstraction_cascade import AbstractionCascade

# Fold-Reference (◉)
from flux_manifold.fold_reference import FoldReference, no_nan_critique, norm_bound_critique, reservoir_norm_critique

# ⧖ Reservoir State
from flux_manifold.reservoir_state import ReservoirState, SuperpositionReservoir

# Convergence Contracts
from flux_manifold.convergence import ConvergenceTier, ConvergenceContract, ConvergenceResult

# Dimensional Squeeze (∇↓)
from flux_manifold.dimensional_squeeze import DimensionalSqueeze

# Topological Squeeze (§4 ontology)
from flux_manifold.topological_squeeze import TopologicalSqueeze, TopologyDiagnostics

# Flow Trace (structured error diagnostics)
from flux_manifold.flow_trace import FlowTrace, FlowTraceEntry, analyze_convergence

# Hamiltonian Flows (§2 ontology)
from flux_manifold.hamiltonian import (
    HamiltonianFlowEngine, HamiltonianState,
    hamiltonian_flow, hamiltonian_flow_batch,
)

# Quantum Interference (§3 ontology)
from flux_manifold.quantum_interference import (
    QuantumInterferenceEngine, InterferenceResult,
)

# Full interpreter
from flux_manifold.interpreter import LatentFluxInterpreter

# TSP solver
from flux_manifold.tsp_solver import LatentFluxTSP, solve_tsp

# Parser + REPL
from flux_manifold.parser import (
    parse, parse_program, evaluate, evaluate_program,
    run, run_file, EvalContext,
    LFPipeline, LFOp, LFVector, LFNumber, LFSymbol, LFFuncCall,
    LFImport, LFLet, LFProgram,
)

__all__ = [
    # ⟼ FluxManifold
    "flux_flow", "flux_flow_traced", "flux_flow_batch", "flux_flow_traced_batch",
    "normalize_flow", "sin_flow", "damped_flow", "adaptive_flow",
    # Baselines
    "random_walk", "gradient_descent", "static_baseline",
    # ∑_ψ Superposition
    "SuperpositionTensor",
    # ≅ DriftEquivalence
    "DriftEquivalence",
    # ↓! Commitment Sink
    "CommitmentSink",
    # ⇑ Abstraction Cascade
    "AbstractionCascade",
    # ◉ Fold-Reference
    "FoldReference", "no_nan_critique", "norm_bound_critique", "reservoir_norm_critique",
    # ⧖ Reservoir State
    "ReservoirState", "SuperpositionReservoir",
    # Convergence Contracts
    "ConvergenceTier", "ConvergenceContract", "ConvergenceResult",
    # ∇↓ Dimensional Squeeze
    "DimensionalSqueeze",
    # Topological Squeeze
    "TopologicalSqueeze", "TopologyDiagnostics",
    # Flow Trace
    "FlowTrace", "FlowTraceEntry", "analyze_convergence",
    # Hamiltonian Flows
    "HamiltonianFlowEngine", "HamiltonianState",
    "hamiltonian_flow", "hamiltonian_flow_batch",
    # Quantum Interference
    "QuantumInterferenceEngine", "InterferenceResult",
    # Interpreter
    "LatentFluxInterpreter",
    # TSP
    "LatentFluxTSP", "solve_tsp",
    # Parser + REPL
    "parse", "parse_program", "evaluate", "evaluate_program",
    "run", "run_file", "EvalContext",
    "LFPipeline", "LFOp", "LFVector", "LFNumber", "LFSymbol", "LFFuncCall",
    "LFImport", "LFLet", "LFProgram",
]
