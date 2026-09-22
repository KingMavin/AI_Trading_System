"""
shared/dna.py

Canonical SHA-256 DNA hash generation for ATS strategy parameter/filter integrity.
Per PRD §8.1: Computed over canonical JSON serialization of the parameters and filters
objects (keys sorted alphabetically, no whitespace).
"""

import json
import hashlib
from typing import Dict, Any


def compute_dna_hash(parameters: Dict[str, Any], filters: Dict[str, Any]) -> str:
    """
    Computes a 64-character SHA-256 hex digest over the canonical JSON serialization
    of parameters and filters (keys sorted, no whitespace).
    """
    combined = {
        "filters": filters or {},
        "parameters": parameters or {}
    }
    canonical_json = json.dumps(combined, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
