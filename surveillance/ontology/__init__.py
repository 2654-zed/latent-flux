"""Ontology modules — entity-role classification for SAI Q-001 and successors.

The role lattice is a multi-axis classification:
    {role: operator | execution_cell | funder | infrastructure | intermediary}
    × {scope: single_contract | fleet | cross_chain}
    × {phase_specialty: positioning | trust_establishment | trigger | exploitation | exfiltration}

See `surveillance/ontology/role_classifier.py` for the current implementation
status (SKELETON; full build is a separate effort).
"""
