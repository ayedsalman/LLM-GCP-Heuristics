"""Candidate registry (plan S0.3, C3, A8).

One JSON object per line in registry/candidates.jsonl (committed; supplementary
data of the paper). One record per *candidate formula*, with the full
provenance of the query that produced it. Raw responses are stored in
runs/raw/<query_id>.json and referenced by hash.

Fields
  candidate_id, query_id, parent_id    identifiers (parent for refinements)
  model, model_snapshot, provider_host query identity (B7)
  queried_at_utc, temperature, seed    sampling parameters
  experiment, arm, round               e.g. "E-B1"/"zero-shot"/0, "E-B3"/"T"/1
  prompt_kind, prompt_sha256           which template, exact text hash
  attempt, n_requeries                 validator-triggered re-queries (C1)
  shown_instances                      VAL instance ids shown to the model (A8)
  response_sha256, raw_path            raw response
  index, name, raw_formula, expr       parsed candidate
  rationale, tie_break                 model's stated design intuition (G4)
  strictly_linear, violations, features_used, coefficients  validator output
  admitted                             passed validation and colouring check
  val_mean_colors, selected            filled by the selection step
  source                               "llm" | "paper-table6" | "control"
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid

REGISTRY = "registry/candidates.jsonl"
RAW_DIR = "runs/raw"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def save_raw(query_id: str, response) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{query_id}.json")
    with open(path, "w") as f:
        json.dump({"text": response.text, "model_snapshot": response.model_snapshot,
                   "queried_at_utc": response.queried_at_utc, "usage": response.usage,
                   "raw": response.raw}, f)
    return path


def append(records, path: str = REGISTRY) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def load(path: str = REGISTRY, **filters) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if all(r.get(k) == v for k, v in filters.items()):
                    out.append(r)
    return out


def update(path: str, key: str, updates: dict) -> None:
    """Rewrite records whose candidate_id is in ``updates`` (dict id -> fields)."""
    recs = load(path)
    with open(path, "w") as f:
        for r in recs:
            if r.get(key) in updates:
                r.update(updates[r[key]])
            f.write(json.dumps(r, sort_keys=True) + "\n")
