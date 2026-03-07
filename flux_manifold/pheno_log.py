"""Phenomenological Log — first-person experiential data from building in Latent Flux.

Records moments of native expression, translation cost, friction, and flow
as experienced during implementation. This is primary research data, not
commentary.

Moment types:
    "native"      — geometry wrote itself, no encoding decision required
    "translation" — continuous concept encoded into available constructs,
                    something approximate about the result
    "friction"    — the language couldn't express what the problem needed,
                    a workaround was necessary
    "flow"        — multiple pieces connected simultaneously,
                    entropy collapsed toward the answer
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "phenomenological_log.jsonl")


def log_entry(
    feature: str,
    moment: str,
    description: str,
    code_context: str,
    resolution: str | None = None,
) -> None:
    """Append a phenomenological log entry.

    Args:
        feature: Which feature is being built (e.g., "reservoir_state",
                 "attractor_competition", "recursive_flow").
        moment: One of "native", "translation", "friction", "flow".
        description: What happened, in plain language.
        code_context: The specific line or construct that triggered this moment.
        resolution: If friction — how was it resolved?
    """
    valid_moments = {"native", "translation", "friction", "flow"}
    if moment not in valid_moments:
        raise ValueError(f"moment must be one of {valid_moments}, got {moment!r}")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feature": feature,
        "moment_type": moment,
        "description": description,
        "code_context": code_context,
        "resolution": resolution,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
