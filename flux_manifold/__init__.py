"""Latent Flux Interpreter – all 7 primitives for transformer-native computation."""

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
from flux_manifold.fold_reference import FoldReference, no_nan_critique, norm_bound_critique

# Dimensional Squeeze (∇↓)
from flux_manifold.dimensional_squeeze import DimensionalSqueeze

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
    "FoldReference", "no_nan_critique", "norm_bound_critique",
    # ∇↓ Dimensional Squeeze
    "DimensionalSqueeze",
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
