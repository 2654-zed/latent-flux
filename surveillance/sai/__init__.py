"""Self-Evolving Actuarial Intelligence (SAI) substrate.

The SAI layer treats QUESTIONS as the primary system abstraction. Data,
models, and pipelines exist to answer questions. Failures evolve into new
questions. The system improves by asking better questions than its past
self — not by accumulating more answers to the same questions.

Modules:
  question_store        — load/rank/save memory/questions.yaml
  question_runner       — dispatch active questions to executable modules
  question_generator    — turn evaluation failures into new questions [TODO]
  capability_liveness   — Q-008 self-surveillance [TODO]
  prediction_registry   — MG3 calibration tracker [TODO]
  episode_engine        — MG4 alert coalescence [TODO]
  adversarial_engine    — Phase 5 attacker-emulation against own INVs [TODO]

See `memory/LOOP.md` (the 7-step reflection cycle) and the Operational
Doctrine section of `docs/lexicon.md` (Adversarial Maneuver framework)
for the methodological context.
"""
__version__ = "0.1.0"
