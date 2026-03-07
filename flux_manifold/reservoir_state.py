"""Reservoir State (⧖) — continuous memory primitive.

Embeds the current state into a high-dimensional continuous dynamical
reservoir, allowing information to echo and persist across sequential
flow operations without triggering a CommitmentSink.

Based on Echo State Network (ESN) dynamics:
    h(t+1) = tanh(W_in · x(t) + W_res · h(t))
    y(t)   = W_out · h(t)

where:
    h(t) ∈ ℝ^r is the reservoir hidden state (r = reservoir_scale × d)
    x(t) ∈ ℝ^d is the input state
    y(t) ∈ ℝ^d is the readout (projected back to state space)

Reservoir dimensionality: r = reservoir_scale × d (default reservoir_scale=4).
    - r < d tends to lose information
    - r > 8d rarely adds capacity but always adds cost
    - 4d sits in the empirically validated ESN sweet spot

Symbol: ⧖ (Hourglass / Time)
"""

from __future__ import annotations

import warnings
import numpy as np


class ReservoirState:
    """Continuous memory via echo state network dynamics.

    The reservoir maintains a hidden state h ∈ ℝ^r that evolves with
    each input, preserving temporal information across flow steps.

    Attributes:
        d: Input/output dimensionality.
        r: Reservoir dimensionality (= reservoir_scale × d).
        reservoir_scale: Multiplier for reservoir size (default 4).
        spectral_radius: Scaling for W_res eigenvalues (default 0.9).
                         Must be < 1.0 for echo state property.
    """

    def __init__(
        self,
        d: int,
        reservoir_scale: int = 4,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.1,
        leak_rate: float = 0.3,
        seed: int = 42,
        reservoir_memory_warning_threshold: int = 100_000_000,  # 100 MB in bytes
    ):
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        if reservoir_scale < 1:
            raise ValueError(f"reservoir_scale must be >= 1, got {reservoir_scale}")
        if not (0 < spectral_radius < 1.0):
            raise ValueError(
                f"spectral_radius must be in (0, 1.0) for echo state property, "
                f"got {spectral_radius}"
            )
        if not (0.0 < leak_rate <= 1.0):
            raise ValueError(
                f"leak_rate must be in (0, 1.0] — 1.0=no memory (Markovian), "
                f"near 0=long memory. Got {leak_rate}"
            )

        self.d = d
        self.reservoir_scale = reservoir_scale
        self.r = d * reservoir_scale
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.seed = seed
        self._warning_threshold = reservoir_memory_warning_threshold

        rng = np.random.default_rng(seed)

        # W_in: input → reservoir (r × d), scaled by input_scaling
        self.W_in = (rng.standard_normal((self.r, d)) * input_scaling).astype(np.float32)

        # W_res: reservoir → reservoir (r × r), sparse, scaled to spectral_radius
        # Sparse initialization (~10% density) is standard ESN practice
        W_raw = rng.standard_normal((self.r, self.r)).astype(np.float32)
        mask = rng.random((self.r, self.r)) < 0.1
        W_raw *= mask
        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(W_raw)
        max_eig = np.max(np.abs(eigenvalues))
        if max_eig > 1e-10:
            W_raw = W_raw * (spectral_radius / max_eig)
        self.W_res = W_raw.astype(np.float32)

        # W_out: reservoir → output (d × r), random projection like DimensionalSqueeze
        self.W_out = (rng.standard_normal((d, self.r)) / np.sqrt(self.r)).astype(np.float32)

        # Hidden state
        self._h: np.ndarray = np.zeros(self.r, dtype=np.float32)
        self._history: list[np.ndarray] = []
        self._committed = False

    @property
    def hidden_state(self) -> np.ndarray:
        """Current reservoir hidden state h ∈ ℝ^r."""
        return self._h.copy()

    def get_history(self, last_n: int | None = None) -> list[np.ndarray]:
        """Reservoir state history.

        Args:
            last_n: If provided, return only the last N entries.
                    None returns the full history.
        """
        if last_n is None:
            return list(self._history)
        return list(self._history[-last_n:])

    @property
    def history(self) -> list[np.ndarray]:
        """Full history of hidden states (property alias for get_history())."""
        return list(self._history)

    @property
    def history_length(self) -> int:
        return len(self._history)

    def step(self, x: np.ndarray) -> np.ndarray:
        """Evolve reservoir by one step with input x ∈ ℝ^d.

        Returns the readout y = W_out · h(t+1) ∈ ℝ^d.

        Raises RuntimeError if this reservoir has been committed.
        """
        if self._committed:
            raise RuntimeError(
                "Reservoir has been committed — ⧖ after ↓! is a linear type error. "
                "The reservoir history was discarded on commitment. "
                "Create a new ReservoirState if you need to continue flowing."
            )
        if x.shape != (self.d,):
            raise ValueError(f"Expected input shape ({self.d},), got {x.shape}")

        x = x.astype(np.float32, copy=False)

        # ESN dynamics with leak rate:
        # h(t+1) = (1 - leak_rate) * h(t) + leak_rate * tanh(W_in · x + W_res · h(t))
        activation = np.tanh(
            self.W_in @ x + self.W_res @ self._h
        ).astype(np.float32)
        self._h = ((1 - self.leak_rate) * self._h + self.leak_rate * activation).astype(np.float32)

        self._history.append(self._h.copy())

        # Readout: y = W_out · h
        y = (self.W_out @ self._h).astype(np.float32)
        return y

    def readout(self) -> np.ndarray:
        """Current readout y = W_out · h ∈ ℝ^d without advancing the reservoir."""
        if self._committed:
            raise RuntimeError(
                "Reservoir has been committed — cannot read after ↓!."
            )
        return (self.W_out @ self._h).astype(np.float32)

    def commit(self) -> np.ndarray:
        """Commit the reservoir: return W_out · h, discard history.

        After commitment:
            - The readout (W_out · h) is returned as a plain ℝ^d vector
            - The reservoir history is discarded
            - Further step() / readout() calls raise RuntimeError
        """
        if self._committed:
            raise RuntimeError("Reservoir already committed — ↓! is irreversible.")

        output = (self.W_out @ self._h).astype(np.float32)
        self._committed = True
        self._history.clear()
        self._h = np.zeros(self.r, dtype=np.float32)
        return output

    @property
    def is_committed(self) -> bool:
        return self._committed

    # ── Method aliases for spec compatibility ─────────────────

    def update(self, state: np.ndarray) -> np.ndarray:
        """Alias for step() — updates reservoir with new state, returns readout."""
        return self.step(state)

    def read(self) -> np.ndarray:
        """Alias for readout() — returns current readout without advancing."""
        return self.readout()

    def memory_bytes(self) -> int:
        """Current memory allocation in bytes.

        Includes weight matrices, hidden state, and history.
        """
        base = (
            self.W_in.nbytes + self.W_res.nbytes + self.W_out.nbytes
            + self._h.nbytes
        )
        history_bytes = sum(h.nbytes for h in self._history)
        return base + history_bytes

    def reset(self) -> None:
        """Reset reservoir to initial state (zero hidden, empty history).

        Only allowed if NOT committed. Use a new ReservoirState after commitment.
        """
        if self._committed:
            raise RuntimeError("Cannot reset a committed reservoir.")
        self._h = np.zeros(self.r, dtype=np.float32)
        self._history.clear()


class SuperpositionReservoir:
    """Per-candidate reservoir states for a SuperpositionTensor.

    Each candidate in the superposition has its own ReservoirState.
    Reservoir histories stay COUPLED with their candidate states:
        - When candidates are reweighted, reservoir histories reorder
        - When candidates are pruned, reservoir histories are pruned

    Reservoirs are stored as a list (not a tensor) so coupling is
    structural rather than enforced by convention.
    """

    def __init__(
        self,
        n: int,
        d: int,
        reservoir_scale: int = 4,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.1,
        leak_rate: float = 0.3,
        seed: int = 42,
        reservoir_memory_warning_threshold: int = 100_000_000,
    ):
        self.n = n
        self.d = d
        self.reservoir_scale = reservoir_scale
        self.leak_rate = leak_rate
        self._warning_threshold = reservoir_memory_warning_threshold

        # Check memory before allocating
        self._check_memory_budget(n, d, reservoir_scale)

        # Each candidate gets its own reservoir, sharing W_in/W_res/W_out
        # (same dynamics, independent hidden states)
        # The weight matrices are shared to save memory — only hidden states differ
        self._template = ReservoirState(
            d=d, reservoir_scale=reservoir_scale,
            spectral_radius=spectral_radius, input_scaling=input_scaling,
            leak_rate=leak_rate, seed=seed,
            reservoir_memory_warning_threshold=reservoir_memory_warning_threshold,
        )

        # Per-candidate hidden states and histories
        self._hidden_states: list[np.ndarray] = [
            np.zeros(self._template.r, dtype=np.float32) for _ in range(n)
        ]
        self._histories: list[list[np.ndarray]] = [[] for _ in range(n)]
        self._committed = False

    def _check_memory_budget(self, n: int, d: int, scale: int) -> None:
        """Warn if total reservoir allocation exceeds threshold."""
        r = d * scale
        # Per-candidate: hidden state (r floats) + history grows over time
        # Base allocation: n × r × 4 bytes (float32)
        base_bytes = n * r * 4
        # Weight matrices: shared, so just one set
        matrix_bytes = (r * d + r * r + d * r) * 4
        total_bytes = base_bytes + matrix_bytes
        if total_bytes > self._warning_threshold:
            warnings.warn(
                f"SuperpositionReservoir allocation: {total_bytes / 1e6:.1f} MB "
                f"(N={n}, d={d}, r={r}) exceeds warning threshold "
                f"({self._warning_threshold / 1e6:.0f} MB). "
                f"Proceeding — set reservoir_memory_warning_threshold higher "
                f"to suppress.",
                ResourceWarning,
                stacklevel=3,
            )

    def step_all(self, states: np.ndarray) -> np.ndarray:
        """Step all N reservoirs with their corresponding candidate states.

        Args:
            states: (N, d) current candidate states.

        Returns:
            (N, d) readouts for each candidate.
        """
        if self._committed:
            raise RuntimeError("SuperpositionReservoir committed — cannot step.")
        if states.shape[0] != self.n:
            raise ValueError(f"Expected {self.n} states, got {states.shape[0]}")

        readouts = np.empty((self.n, self.d), dtype=np.float32)
        for i in range(self.n):
            x = states[i].astype(np.float32, copy=False)
            # Shared weight matrices, per-candidate hidden state
            # ESN with leak rate
            activation = np.tanh(
                self._template.W_in @ x + self._template.W_res @ self._hidden_states[i]
            ).astype(np.float32)
            self._hidden_states[i] = (
                (1 - self.leak_rate) * self._hidden_states[i]
                + self.leak_rate * activation
            ).astype(np.float32)
            self._histories[i].append(self._hidden_states[i].copy())
            readouts[i] = self._template.W_out @ self._hidden_states[i]

        return readouts

    def readout_all(self) -> np.ndarray:
        """Current readouts for all candidates without advancing."""
        if self._committed:
            raise RuntimeError("SuperpositionReservoir committed — cannot read.")
        readouts = np.empty((self.n, self.d), dtype=np.float32)
        for i in range(self.n):
            readouts[i] = self._template.W_out @ self._hidden_states[i]
        return readouts

    def reorder(self, indices: np.ndarray) -> None:
        """Reorder reservoir states to match reordered candidates.

        Called when SuperpositionTensor candidates are reordered
        (e.g., by reweight_by_drift or prune). Keeps reservoir
        histories coupled with their candidate states.
        """
        if self._committed:
            raise RuntimeError("SuperpositionReservoir committed — cannot reorder.")
        self._hidden_states = [self._hidden_states[i].copy() for i in indices]
        self._histories = [list(self._histories[i]) for i in indices]
        self.n = len(indices)

    def prune(self, indices: np.ndarray) -> None:
        """Prune reservoir states to match pruned candidates."""
        self.reorder(indices)

    def commit_best(self, best_idx: int) -> np.ndarray:
        """Commit the reservoir of the best candidate.

        Returns W_out · h_best ∈ ℝ^d and discards all histories.
        """
        if self._committed:
            raise RuntimeError("SuperpositionReservoir already committed.")

        output = (self._template.W_out @ self._hidden_states[best_idx]).astype(np.float32)
        self._committed = True
        self._hidden_states.clear()
        self._histories.clear()
        return output

    def get_history(self, candidate_idx: int) -> list[np.ndarray]:
        """Get the reservoir history for a specific candidate."""
        if candidate_idx >= self.n:
            raise IndexError(f"Candidate {candidate_idx} out of range (n={self.n})")
        return list(self._histories[candidate_idx])

    @property
    def is_committed(self) -> bool:
        return self._committed
