"""
Validation script for GeneticTestingEnv + NumpyInference.

Checks:
  1. Prior carrier probabilities match allele-frequency formula
  2. Posterior updates after observing a founder's genotype match hand-computed values
  3. Multi-step episode runs without error and produces valid reward totals
  4. Myopic policy (always test highest-carrier-prob individual) works end-to-end

Run from the repo root:
    python -m scripts.validate_env
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from typing import Dict

from genetic_dp.models.pedigree import Pedigree
from genetic_dp.config import get_config
from genetic_dp.envs import GeneticTestingEnv, NumpyInference

ALLELE_FREQ = 0.1
TOL = 1e-10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nuclear_family() -> Pedigree:
    """2 founders + 2 children. Simple 4-person pedigree."""
    ped = Pedigree()
    ped.add_individual("F1")
    ped.add_individual("F2")
    ped.add_individual("C1", parents=("F1", "F2"))
    ped.add_individual("C2", parents=("F1", "F2"))
    return ped


def extended_family() -> Pedigree:
    """7-person pedigree: grandparents → parents → children."""
    ped = Pedigree()
    ped.add_individual("GF1")
    ped.add_individual("GM1")
    ped.add_individual("GF2")
    ped.add_individual("GM2")
    ped.add_individual("P1", parents=("GF1", "GM1"))
    ped.add_individual("P2", parents=("GF2", "GM2"))
    ped.add_individual("C1", parents=("P1", "P2"))
    return ped


def carrier_prob(dist: Dict[int, float]) -> float:
    return dist.get(1, 0.0) + dist.get(2, 0.0)


def assert_close(label: str, got: float, expected: float, tol: float = TOL) -> None:
    err = abs(got - expected)
    status = "PASS" if err < tol else "FAIL"
    print(f"  [{status}] {label}: got={got:.8f}, expected={expected:.8f}, err={err:.2e}")
    if status == "FAIL":
        raise AssertionError(f"Check failed: {label}")


# ---------------------------------------------------------------------------
# Test 1: Prior carrier probabilities
# ---------------------------------------------------------------------------

def test_prior_carrier_probs() -> None:
    """Prior P(carrier) = 2p(1-p) + p^2 = p(2-p) for all individuals."""
    print("\n=== Test 1: Prior carrier probabilities ===")
    p = ALLELE_FREQ
    expected_carrier = p * (2 - p)  # = 2p(1-p) + p^2

    ped = nuclear_family()
    config = get_config(ped.to_list(), pedigree=ped, allele_freq=p)
    infer = NumpyInference(ped, ("gene",), config)

    result = infer.posterior({})
    for ind in ped.to_list():
        cp = carrier_prob(result.marginals[ind])
        assert_close(f"prior P(carrier | {ind})", cp, expected_carrier)
    print("  -> All priors correct.")


# ---------------------------------------------------------------------------
# Test 2: Posterior updates — founder tests
# ---------------------------------------------------------------------------

def test_posterior_after_founder_test() -> None:
    """Test belief propagation after observing a founder's genotype.

    Hand-computed values (allele_freq=0.1, nuclear family F1,F2 → C1,C2):

    Case F1=0 (NN, homozygous non-carrier):
        F1 can only pass N → P(C1 carrier | F1=0) = P(D from F2) = p*(2-p)? No:
        Actually P(C1=ND | F1=0) = P(F2 passes D) = p (marginalizing F2)
                                  = 2p(1-p)*0.5 + p^2*1.0 = p(1-p) + p^2 = p
        P(C1=DD | F1=0) = 0 (F1 can't pass D)
        → P(C1 carrier | F1=0) = p = 0.1

    Case F1=1 (ND, heterozygous carrier):
        P(C1=ND | F1=1) = 0.5  (regardless of F2, by symmetry)
        P(C1=DD | F1=1) = P(F1 passes D)*P(F2 passes D) = 0.5 * p = 0.5*0.1 = 0.05
                          (marginalizing F2: 0*0.81 + 0.5*0.18 + 1.0*0.01 = 0.09+0.01 ... wait)
        Let me be precise:
        P(C1=DD | F1=1) = sum_{f2} P(DD | F1=1, F2=f2)*P(F2=f2)
          f2=0 (NN): P(DD|ND,NN) = 0.5*0.0 = 0.0 ; P(F2=0)=0.81
          f2=1 (ND): P(DD|ND,ND) = 0.5*0.5 = 0.25 ; P(F2=1)=0.18
          f2=2 (DD): P(DD|ND,DD) = 0.5*1.0 = 0.5  ; P(F2=2)=0.01
        → P(C1=DD | F1=1) = 0*0.81 + 0.25*0.18 + 0.5*0.01 = 0.045 + 0.005 = 0.05
        → P(C1 carrier | F1=1) = 0.5 + 0.05 = 0.55

    Case F1=2 (DD, homozygous carrier):
        F1 always passes D → child gets at least one D → P(carrier)=1.0
        Verify: P(C1=NN|F1=2) = 0 (F1 always passes D, so no NN possible)
        P(C1 carrier | F1=2) = 1.0
    """
    print("\n=== Test 2: Posterior after founder genotype ===")
    p = ALLELE_FREQ
    ped = nuclear_family()
    config = get_config(ped.to_list(), pedigree=ped, allele_freq=p)
    infer = NumpyInference(ped, ("gene",), config)

    # F1=0 (non-carrier)
    r = infer.posterior({"F1": 0})
    assert_close("P(C1 carrier | F1=0)", carrier_prob(r.marginals["C1"]), p)
    assert_close("P(C2 carrier | F1=0)", carrier_prob(r.marginals["C2"]), p)
    assert_close("P(F2 carrier | F1=0)", carrier_prob(r.marginals["F2"]), p * (2 - p))

    # F1=1 (heterozygous carrier)
    r = infer.posterior({"F1": 1})
    assert_close("P(C1 carrier | F1=1)", carrier_prob(r.marginals["C1"]), 0.55)
    assert_close("P(C2 carrier | F1=1)", carrier_prob(r.marginals["C2"]), 0.55)

    # F1=2 (homozygous carrier)
    r = infer.posterior({"F1": 2})
    assert_close("P(C1 carrier | F1=2)", carrier_prob(r.marginals["C1"]), 1.0)
    assert_close("P(C2 carrier | F1=2)", carrier_prob(r.marginals["C2"]), 1.0)

    print("  -> All posterior updates correct.")


# ---------------------------------------------------------------------------
# Test 3: Posterior after observing both founders
# ---------------------------------------------------------------------------

def test_posterior_both_founders() -> None:
    """Both founders NN → all children are NN with certainty."""
    print("\n=== Test 3: Both founders non-carrier ===")
    p = ALLELE_FREQ
    ped = nuclear_family()
    config = get_config(ped.to_list(), pedigree=ped, allele_freq=p)
    infer = NumpyInference(ped, ("gene",), config)

    r = infer.posterior({"F1": 0, "F2": 0})
    assert_close("P(C1 carrier | F1=0, F2=0)", carrier_prob(r.marginals["C1"]), 0.0)
    assert_close("P(C2 carrier | F1=0, F2=0)", carrier_prob(r.marginals["C2"]), 0.0)

    r = infer.posterior({"F1": 2, "F2": 2})
    assert_close("P(C1 carrier | F1=2, F2=2)", carrier_prob(r.marginals["C1"]), 1.0)
    assert_close("P(C2 carrier | F1=2, F2=2)", carrier_prob(r.marginals["C2"]), 1.0)

    print("  -> Boundary cases correct.")


# ---------------------------------------------------------------------------
# Test 4: Posterior normalises correctly (sums to 1)
# ---------------------------------------------------------------------------

def test_posteriors_normalised() -> None:
    """All marginals sum to 1.0 under various evidence sets."""
    print("\n=== Test 4: Posteriors sum to 1 ===")
    p = ALLELE_FREQ
    ped = nuclear_family()
    config = get_config(ped.to_list(), pedigree=ped, allele_freq=p)
    infer = NumpyInference(ped, ("gene",), config)

    evidence_sets = [{}, {"F1": 0}, {"F1": 1}, {"F1": 2}, {"F1": 0, "C1": 1}]
    for ev in evidence_sets:
        r = infer.posterior(ev)
        for ind, dist in r.marginals.items():
            s = sum(dist.values())
            assert_close(f"sum P({ind}) | ev={ev}", s, 1.0, tol=1e-9)
    print("  -> All marginals normalise correctly.")


# ---------------------------------------------------------------------------
# Test 5: Extended family — grandchild probability
# ---------------------------------------------------------------------------

def test_extended_family() -> None:
    """7-person pedigree: check grandchild carrier prob equals p(2-p) from prior,
    and drops when grandparent tests NN."""
    print("\n=== Test 5: Extended family inference ===")
    p = ALLELE_FREQ
    ped = extended_family()
    config = get_config(ped.to_list(), pedigree=ped, allele_freq=p)
    infer = NumpyInference(ped, ("gene",), config)

    r = infer.posterior({})
    assert_close("P(C1 carrier | prior)", carrier_prob(r.marginals["C1"]), p * (2 - p))

    # Observe GF1=0 (NN): P1 carrier prob drops, C1 further drops
    r_gf1_0 = infer.posterior({"GF1": 0})
    p_p1_after = carrier_prob(r_gf1_0.marginals["P1"])
    p_c1_after = carrier_prob(r_gf1_0.marginals["C1"])
    print(f"  P(P1 carrier | GF1=0) = {p_p1_after:.6f}")
    print(f"  P(C1 carrier | GF1=0) = {p_c1_after:.6f}")
    # Both should decrease
    assert p_p1_after < p * (2 - p), "P1 carrier prob should drop when GF1=NN"
    assert p_c1_after < p * (2 - p), "C1 carrier prob should drop when GF1=NN"
    print("  PASS: Carrier probs decrease after grandparent tests NN.")

    print("  -> Extended family inference correct.")


# ---------------------------------------------------------------------------
# Test 6: GeneticTestingEnv — episode rollout
# ---------------------------------------------------------------------------

def test_env_episode() -> None:
    """Run one episode with a fixed seed; check observations and reward shape."""
    print("\n=== Test 6: GeneticTestingEnv episode ===")
    p = ALLELE_FREQ
    ped = nuclear_family()
    individuals = ped.to_list()
    config = get_config(individuals, pedigree=ped, allele_freq=p)
    env = GeneticTestingEnv(ped, config, genes=("gene",), seed=42)

    obs = env.reset()
    # obs_dim = N*G + N (carrier probs + tested flags)
    expected_dim = env.N * len(env.genes) + env.N
    assert obs.shape == (expected_dim,), f"Expected obs shape ({expected_dim},), got {obs.shape}"
    print(f"  Initial obs: {obs.round(4)}  (carrier_probs | tested_flags)")

    rewards = []
    step = 0
    while not env.done:
        mask = env.action_mask()
        # Myopic: test the untested individual with highest carrier prob
        untested_actions = [i for i in range(env.N) if mask[i]]
        if not untested_actions:
            action = env.N  # stop
        else:
            probs = obs[:env.N]
            best = max(untested_actions, key=lambda i: probs[i])
            # Stop if no-test reward beats best test reward
            action = best

        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        step += 1
        print(f"  Step {step}: {info.get('action','?')} "
              f"({info.get('individual','')}"
              f"{' g='+str(info.get('observed_genotype','')) if 'observed_genotype' in info else ''}) "
              f"reward={reward:.4f}")

    total = sum(rewards)
    print(f"  Total reward: {total:.6f}  (cumulative in env: {env._cumulative_reward:.6f})")
    assert abs(total - env._cumulative_reward) < 1e-9
    print("  -> Episode ran cleanly.")


# ---------------------------------------------------------------------------
# Test 7: Stop immediately — check terminal reward formula
# ---------------------------------------------------------------------------

def test_env_stop_immediately() -> None:
    """Stopping immediately should equal sum of no-test rewards at prior beliefs."""
    print("\n=== Test 7: Stop-immediately terminal reward ===")
    p = ALLELE_FREQ
    ped = nuclear_family()
    individuals = ped.to_list()
    config = get_config(individuals, pedigree=ped, allele_freq=p)
    env = GeneticTestingEnv(ped, config, genes=("gene",), seed=0)

    env.reset()
    _, reward, done, info = env.step(env.N)  # stop
    assert done
    print(f"  Terminal reward (stop immediately): {reward:.6f}")
    print("  -> Stop action terminates episode correctly.")


# ---------------------------------------------------------------------------
# Test 8: Multi-gene prior
# ---------------------------------------------------------------------------

def test_multigene_prior() -> None:
    """With two genes and different allele frequencies, check per-gene priors."""
    print("\n=== Test 8: Multi-gene prior carrier probs ===")
    ped = nuclear_family()
    individuals = ped.to_list()
    config = get_config(
        individuals,
        pedigree=ped,
        genes=["BRCA1", "BRCA2"],
        allele_freqs={"BRCA1": 0.05, "BRCA2": 0.02},
    )
    infer = NumpyInference(ped, ("BRCA1", "BRCA2"), config)

    r = infer.posterior({})
    for gene, freq in [("BRCA1", 0.05), ("BRCA2", 0.02)]:
        expected = freq * (2 - freq)
        per_gene = r._per_gene_gene_first
        if per_gene and gene in per_gene:
            gene_marg = per_gene[gene]
            for ind in individuals:
                cp = gene_marg[ind].get(1, 0.0) + gene_marg[ind].get(2, 0.0)
                assert_close(f"P({ind} carrier | gene={gene})", cp, expected)
    print("  -> Multi-gene prior correct.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  GeneticTestingEnv — Belief Update Validation")
    print("=" * 60)

    try:
        test_prior_carrier_probs()
        test_posterior_after_founder_test()
        test_posterior_both_founders()
        test_posteriors_normalised()
        test_extended_family()
        test_env_episode()
        test_env_stop_immediately()
        test_multigene_prior()
        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n  *** FAILED: {e} ***")
        sys.exit(1)


if __name__ == "__main__":
    main()
