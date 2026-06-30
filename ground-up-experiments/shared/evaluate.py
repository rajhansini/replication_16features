"""Evaluation utilities: ratio2 computation and sanity checks.

ratio2 = (V* - L) / (V* - V_stop)   [lower = better, 0 = perfect]

L is computed by:
  1. At every state, use the net greedily to pick an action (test who or stop)
  2. Compute the TRUE expected value of following that greedy policy,
     using the exact belief map (not the net's predictions).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from genetic_dp.exact_dp.utils import GENOTYPE_STATES, lift_tuple_posteriors_to_genes
from genetic_dp.models.belief import InferenceResult
from genetic_dp.models.reward import r_reward, r_reward_test


# ─── helpers ─────────────────────────────────────────────────────────────────

def _state_to_vec_single(state, belief, individuals) -> np.ndarray:
    p_s = belief[state]
    x = []
    for person in individuals:
        dist = p_s[person]
        x.extend([dist[0], dist[1], dist[2]])
    return np.array(x, dtype=np.float32)


def _state_to_vec_two(state, belief, individuals, genes) -> np.ndarray:
    entry = belief[state]
    per_gene = lift_tuple_posteriors_to_genes(entry, genes, GENOTYPE_STATES)
    x = []
    for person in individuals:
        for gene in genes:
            dist = per_gene[gene][person]
            x.extend([dist[0], dist[1], dist[2]])
    return np.array(x, dtype=np.float32)


def _net_predict(net, x_np: np.ndarray, device: str) -> float:
    with torch.no_grad():
        t = torch.FloatTensor(x_np).unsqueeze(0).to(device)
        return net(t).item()


# ─── greedy policy value (L) ─────────────────────────────────────────────────

def compute_greedy_policy_value(net, dataset: dict, device: str = "cpu") -> float:
    """
    Simulate the net's greedy policy and return L = its expected value from root.

    The net picks WHICH action to take; then we propagate using the TRUE
    belief map so the result is an honest lower bound on V*.
    """
    belief      = dataset["belief"]
    individuals = dataset["individuals"]
    config      = dataset["config"]
    n_people    = len(individuals)
    genes       = dataset.get("genes")
    two_gene    = genes is not None
    gen_states  = dataset.get("two_gene_states", GENOTYPE_STATES)

    net.eval()
    memo: dict = {}
    target_dim = dataset.get("input_dim")   # set by step4 when padding mixed families

    def _to_vec(state):
        if two_gene:
            entry = belief[state]
            if isinstance(entry, InferenceResult):
                pg = entry.get_per_gene_probs()
                x = []
                for person in individuals:
                    for gene in genes:
                        dist = pg[gene][person]
                        x.extend([dist[0], dist[1], dist[2]])
                vec = np.array(x, dtype=np.float32)
            else:
                vec = _state_to_vec_two(state, belief, individuals, genes)
            if target_dim is not None and len(vec) < target_dim:
                vec = np.concatenate([vec, np.zeros(target_dim - len(vec), dtype=np.float32)])
            return vec
        return _state_to_vec_single(state, belief, individuals)

    def _per_gene(entry):
        if isinstance(entry, InferenceResult):
            return entry.get_per_gene_probs()
        return lift_tuple_posteriors_to_genes(entry, genes, GENOTYPE_STATES)

    def _marginals(entry):
        if isinstance(entry, InferenceResult):
            return entry.marginals
        return entry

    def _tuple_pmf(entry, person):
        """Return {outcome: prob} for person from a belief entry."""
        if isinstance(entry, InferenceResult) and entry.has_tuple_pmfs():
            return entry.get_tuple_pmfs().get(person, {})
        # Single-gene: entry is {person: {g: prob}}
        return entry[person]

    def _stop_val(state, entry, tested):
        marg = _marginals(entry)
        if two_gene:
            pg = _per_gene(entry)
            return float(sum(
                r_reward(
                    k, marg, config.a, config.b, config.c, config.delta,
                    per_gene_probs=pg,
                    a_gene=config.a_gene, b_gene=config.b_gene,
                    c_gene=config.c_gene, delta_gene=config.delta_gene,
                )
                for k in individuals if k not in tested
            ))
        return float(sum(
            r_reward(k, marg, config.a, config.b, config.c, config.delta)
            for k in individuals if k not in tested
        ))

    def _test_reward(i, entry):
        marg = _marginals(entry)
        if two_gene:
            pg = _per_gene(entry)
            return float(r_reward_test(
                i, marg, config.a, config.b, config.c, config.delta,
                config.fixed_cost, config.variable_cost,
                per_gene_probs=pg,
                a_gene=config.a_gene, c_gene=config.c_gene, delta_gene=config.delta_gene,
            ))
        return float(r_reward_test(
            i, marg, config.a, config.b, config.c, config.delta,
            config.fixed_cost, config.variable_cost,
        ))

    def value_at(state) -> float:
        if state in memo:
            return memo[state]

        entry   = belief[state]
        tested  = {i for i, _ in state}
        v_stop  = _stop_val(state, entry, tested)

        if len(tested) == n_people:
            memo[state] = 0.0
            return 0.0

        # ── ask the net: which action is best? ──────────────────────────────
        best_net_q  = v_stop
        best_person = None

        for i in individuals:
            if i in tested:
                continue
            r_i      = _test_reward(i, entry)
            pmf_i    = _tuple_pmf(entry, i)
            exp_net  = 0.0
            for g, prob_g in pmf_i.items():
                if prob_g <= 1e-12:
                    continue
                next_s = frozenset(state | {(i, g)})
                if next_s not in belief:
                    continue
                exp_net += prob_g * _net_predict(net, _to_vec(next_s), device)
            q_i = r_i + exp_net
            if q_i > best_net_q:
                best_net_q  = q_i
                best_person = i

        # ── if stop is best, return V_stop ──────────────────────────────────
        if best_person is None:
            memo[state] = v_stop
            return v_stop

        # ── compute TRUE value of testing best_person ────────────────────────
        r_best   = _test_reward(best_person, entry)
        pmf_best = _tuple_pmf(entry, best_person)
        exp_true = 0.0
        for g, prob_g in pmf_best.items():
            if prob_g <= 1e-12:
                continue
            next_s = frozenset(state | {(best_person, g)})
            if next_s not in belief:
                continue
            exp_true += prob_g * value_at(next_s)

        result       = r_best + exp_true
        memo[state]  = result
        return result

    return value_at(frozenset())


def compute_ratio2(net, dataset: dict, device: str = "cpu") -> tuple[float, float]:
    """Returns (ratio2, L)."""
    V_root      = dataset["V_root"]
    V_stop_root = dataset["V_stop_root"]
    denom       = V_root - V_stop_root

    L = compute_greedy_policy_value(net, dataset, device)

    if abs(denom) < 1e-12:
        return 0.0, L
    ratio2 = (V_root - L) / denom
    return float(ratio2), float(L)


# ─── sanity checks ───────────────────────────────────────────────────────────

def sanity_checks(dataset: dict, verbose: bool = True) -> list[str]:
    """
    Structural checks on V*(s). Returns list of error strings (empty = all pass).

    Checks:
      1. V*(s) >= V_stop(s) for every reachable state
      2. V*(leaf) = 0  (all tested, nothing left to gain)
      3. Tested person's belief is degenerate (prob=1 for observed genotype)
      4. Root state exists
      5. V*(root) >= V_stop(root)
    """
    belief      = dataset["belief"]
    V_star      = dataset["V_star"]
    individuals = dataset["individuals"]
    config      = dataset["config"]
    genes       = dataset.get("genes")
    two_gene    = genes is not None
    gen_states  = dataset.get("two_gene_states", GENOTYPE_STATES)
    errors: list[str] = []

    def _stop_val(state, p_s, tested):
        if two_gene:
            per_gene = lift_tuple_posteriors_to_genes(p_s, genes, GENOTYPE_STATES)
            return float(sum(
                r_reward(
                    k, p_s, config.a, config.b, config.c, config.delta,
                    per_gene_probs=per_gene,
                    a_gene=config.a_gene, b_gene=config.b_gene,
                    c_gene=config.c_gene, delta_gene=config.delta_gene,
                )
                for k in individuals if k not in tested
            ))
        return float(sum(
            r_reward(k, p_s, config.a, config.b, config.c, config.delta)
            for k in individuals if k not in tested
        ))

    # Check 4: root exists
    if frozenset() not in V_star:
        errors.append("ROOT STATE MISSING from V_star")
    if frozenset() not in belief:
        errors.append("ROOT STATE MISSING from belief")

    # Check 5
    if dataset["V_root"] < dataset["V_stop_root"] - 1e-9:
        errors.append(
            f"V*(root)={dataset['V_root']:.6f} < V_stop(root)={dataset['V_stop_root']:.6f}"
        )

    for state, v_star in V_star.items():
        entry = belief.get(state)
        if entry is None:
            errors.append(f"State in V_star but not in belief: {state}")
            continue

        # Get marginals (works for both InferenceResult and plain dict)
        if isinstance(entry, InferenceResult):
            marg = entry.marginals
            pg   = entry.get_per_gene_probs() if two_gene else None
        else:
            marg = entry
            pg   = lift_tuple_posteriors_to_genes(entry, genes, GENOTYPE_STATES) if two_gene else None

        # reachability: first person's probs should sum to 1
        if sum(marg[individuals[0]].values()) < 1e-9:
            continue

        tested = {i for i, _ in state}

        # Check 1: V*(s) >= V_stop(s)
        if two_gene:
            v_stop = float(sum(
                r_reward(k, marg, config.a, config.b, config.c, config.delta,
                         per_gene_probs=pg,
                         a_gene=config.a_gene, b_gene=config.b_gene,
                         c_gene=config.c_gene, delta_gene=config.delta_gene)
                for k in individuals if k not in tested
            ))
        else:
            v_stop = float(sum(
                r_reward(k, marg, config.a, config.b, config.c, config.delta)
                for k in individuals if k not in tested
            ))
        if v_star < v_stop - 1e-9:
            errors.append(
                f"V*(s) < V_stop(s): {v_star:.6f} < {v_stop:.6f}  state={state}"
            )

        # Check 2: leaf value = 0
        if len(tested) == len(individuals) and abs(v_star) > 1e-9:
            errors.append(f"Leaf V*(s) = {v_star:.6f} ≠ 0  state={state}")

        # Check 3: tested person has degenerate belief (primary gene only for two-gene)
        for person, obs_g in state:
            primary_g = obs_g[0] if isinstance(obs_g, tuple) else obs_g
            dist = marg[person]
            prob_obs = dist.get(primary_g, 0.0)
            if abs(prob_obs - 1.0) > 1e-6:
                errors.append(
                    f"Tested {person}={obs_g} non-degenerate primary belief: {dict(dist)}"
                )

    if verbose:
        n = len(V_star)
        if errors:
            print(f"  [SANITY FAIL] {len(errors)} error(s) over {n} states:")
            for e in errors[:5]:
                print(f"    {e}")
            if len(errors) > 5:
                print(f"    ... ({len(errors) - 5} more)")
        else:
            print(f"  [SANITY PASS] All checks passed over {n} states.")

    return errors
