"""Build (state, action, Q*) training examples from a cached exact-DP dataset.

Q*(s, a) is computed directly from what's already in the exact-DP cache — no new
DP solve. Same formula _q_hat() uses at eval time in exputils/eval.py, except the
target here is the exact V_star (ground truth), not a learned v_hat.

Results are cached to disk per config, since this loop is pure-Python over every
(state, untested person) pair and doesn't need to be redone across runs/models.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import torch

HERE        = Path(__file__).resolve().parent
ROOT        = HERE.parent.parent
EXPERIMENTS = ROOT / "ground-up-experiments"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(HERE.parent))

from exputils.eval import _get_entry, _test_r  # noqa: E402

QSA_CACHE_DIR = HERE / "cache"


def build_qsa_index(ds, device="cpu", cache_key: str | None = None, force: bool = False):
    """
    Returns:
        state_idx  : (M,) long  — row into ds["states"] / nf / gf
        action_idx : (M,) long  — person index, per ds["individuals"] order
        y          : (M,) float — Q*(state, person), exact
    """
    if cache_key is not None:
        cache_path = QSA_CACHE_DIR / f"{cache_key}_qsa.pt"
        if cache_path.exists() and not force:
            payload = torch.load(cache_path, map_location=device)
            return payload["state_idx"].to(device), payload["action_idx"].to(device), payload["y"].to(device)

    states      = ds["states"]
    individuals = ds["individuals"]
    person_idx  = {p: i for i, p in enumerate(individuals)}
    belief      = ds["belief"]
    config      = ds["config"]
    genes       = ds.get("genes", ("GeneA", "GeneB", "GeneC"))
    V_star      = ds["V_star"]

    state_idx, action_idx, y = [], [], []

    for i, state in enumerate(states):
        tested   = {p for p, _ in state}
        untested = [p for p in individuals if p not in tested]
        if not untested:
            continue

        per_gene, tuple_pmfs = _get_entry(belief, state, genes)

        for person in untested:
            r_i   = _test_r(person, per_gene, config)
            pmf   = tuple_pmfs.get(person, {})
            exp_v = sum(
                prob * V_star.get(frozenset(state | {(person, g)}), 0.0)
                for g, prob in pmf.items()
                if prob > 1e-12 and frozenset(state | {(person, g)}) in belief
            )
            state_idx.append(i)
            action_idx.append(person_idx[person])
            y.append(r_i + exp_v)

    state_idx_t  = torch.tensor(state_idx,  dtype=torch.long)
    action_idx_t = torch.tensor(action_idx, dtype=torch.long)
    y_t          = torch.tensor(y,          dtype=torch.float32)

    if cache_key is not None:
        QSA_CACHE_DIR.mkdir(exist_ok=True)
        torch.save({"state_idx": state_idx_t, "action_idx": action_idx_t, "y": y_t}, cache_path)

    return state_idx_t.to(device), action_idx_t.to(device), y_t.to(device)


def load_qsa_config(cache_dir: Path, key: str, device="cpu", force: bool = False):
    """Load ds pickle + build (nf, gf via caller) + qsa index for one config, cached."""
    with open(cache_dir / f"{key}.pkl", "rb") as f:
        ds = pickle.load(f)
    state_idx, action_idx, y = build_qsa_index(ds, device=device, cache_key=key, force=force)
    return ds, state_idx, action_idx, y
