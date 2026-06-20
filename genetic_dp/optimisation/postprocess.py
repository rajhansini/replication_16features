from collections.abc import Mapping, Sequence

from .state_features import (
    build_state_feature_bundle,
    build_trio_share_key_map,
    iter_trio_feature_rows,
    resolve_trio_blocks,
)
from .theta_model import evaluate_theta_model_correction, resolve_theta_mode
from .myopic_adp import myopic_adp_term_value, regime_residual_term_value
from .oracle_adp import oracle_adp_term_value


def _parse_stage_gene_key(key):
    if isinstance(key, tuple) and len(key) == 2:
        stage_raw, gene_raw = key
        try:
            stage_val = int(stage_raw)
        except (TypeError, ValueError):
            return None
        if not isinstance(gene_raw, str):
            return None
        return stage_val, gene_raw
    if isinstance(key, str) and key.startswith("stage=") and "|gene=" in key:
        left, right = key.split("|gene=", 1)
        if not left.startswith("stage="):
            return None
        try:
            stage_val = int(left[len("stage=") :])
        except ValueError:
            return None
        return stage_val, right
    return None


def phi_hat(
    s,
    *,
    theta_star,
    W_star,
    belief,
    gen_states,
    individuals,
    theta_mode=None,
    pedigree=None,
    tuple_pmfs=None,
    tuple_mode=False,
    aaub_star=None,
    W_edge_star=None,
    pedigree_edges=None,
    W_trio_star=None,
    pedigree_trios=None,
    infer=None,
    genes=None,
    feature_cache=None,
    theta_model=None,
    theta_model_spec=None,
    myopic_adp_star=None,
    oracle_adp_star=None,
    regime_residual_star=None,
):
    """
    Reconstruct Φ★(s) = θ★(s) + Σ_{tested}(i,g) W★[i][g]
                       + Σ_{untested i} Σ_g p_s(i=g)·W★[i][g]
                       + optional AAUB uplift terms

    θ★ semantics are controlled explicitly by ``theta_mode``. If omitted,
    THETA_MODE from the environment is used.
    """

    mode = resolve_theta_mode(theta_mode=theta_mode, theta_model=theta_model)
    if mode not in {"scalar", "stage", "person", "person_stage", "stage_gene"}:
        raise ValueError(
            "Unknown theta mode for phi_hat="
            f"{mode!r} (expected 'scalar', 'stage', 'person', 'person_stage', or 'stage_gene')."
        )
    if not isinstance(s, frozenset):
        raise AssertionError(
            f"State must be evidence-only frozenset[(person,outcome)], got {type(s).__name__}: {s!r}"
        )
    evidence_state = s

    if len(evidence_state) >= len(individuals):
        return 0.0

    belief_entry = belief[s]
    posterior_entry = belief_entry[0] if isinstance(belief_entry, tuple) else belief_entry
    if isinstance(posterior_entry, Mapping):
        p_s = posterior_entry
    elif hasattr(posterior_entry, "marginals"):
        p_s = posterior_entry.marginals
    else:
        p_s = posterior_entry

    def _theta_for_state(state) -> float:
        stage = len(state)
        if mode == "scalar":
            if isinstance(theta_star, Mapping):
                return float(theta_star.get(stage, 0.0))
            if isinstance(theta_star, Sequence) and not isinstance(theta_star, (str, bytes)):
                return float(theta_star[stage])
            return float(theta_star)
        if mode == "stage":
            if isinstance(theta_star, Mapping):
                return float(theta_star.get(stage, 0.0))
            if isinstance(theta_star, Sequence) and not isinstance(theta_star, (str, bytes)):
                return float(theta_star[stage])
            return float(theta_star)
        if mode == "person":
            tested_people = {person for person, _ in state}
            if isinstance(theta_star, Mapping):
                return float(sum(theta_star.get(person, 0.0) for person in tested_people))
            return float(theta_star)
        if mode == "stage_gene":
            if isinstance(theta_star, Mapping):
                total = 0.0
                found = False
                for key, value in theta_star.items():
                    parsed = _parse_stage_gene_key(key)
                    if parsed is None:
                        continue
                    key_stage, _gene = parsed
                    if key_stage != stage:
                        continue
                    total += float(value)
                    found = True
                if found:
                    return total
                # Compatibility fallback: accept stage-indexed maps when explicitly requested.
                return float(theta_star.get(stage, 0.0))
            if isinstance(theta_star, Sequence) and not isinstance(theta_star, (str, bytes)):
                return float(theta_star[stage])
            return float(theta_star)
        # person_stage: evidence-only interaction by stage.
        tested_people = {person for person, _ in state}
        if not isinstance(theta_star, Mapping):
            return 0.0
        return float(sum(theta_star.get((person, stage), 0.0) for person in tested_people))

    # Part 1: Sum for tested individuals
    def _normalize_outcome(person, outcome):
        if not tuple_mode:
            return outcome
        candidate_keys = W_star.get(person, {})
        if not candidate_keys:
            return outcome
        sample_key = next(iter(candidate_keys))
        if isinstance(sample_key, tuple):
            if isinstance(outcome, tuple):
                if len(outcome) == len(sample_key):
                    return outcome
                # broadcast scalar tuple to expected length
                if len(outcome) == 1:
                    return tuple(outcome[0] for _ in sample_key)
                return tuple(outcome[: len(sample_key)])
            return tuple(outcome for _ in sample_key)
        return outcome

    term_tested = 0.0
    for (i, g) in evidence_state:
        normalized = _normalize_outcome(i, g)
        term_tested += W_star[i][normalized]

    # Part 2: Sum for untested individuals
    term_untested = 0.0
    tested_persons = {i for (i, _) in evidence_state}
    
    untested_individuals = [i for i in individuals if i not in tested_persons]
    
    if untested_individuals:
        if tuple_mode and tuple_pmfs:
            for i in untested_individuals:
                tuple_dist = tuple_pmfs.get(s, {}).get(i, {})
                if not tuple_dist:
                    continue
                for outcome, prob in tuple_dist.items():
                    normalized = _normalize_outcome(i, outcome)
                    term_untested += prob * W_star[i][normalized]
        else:
            for i in untested_individuals:
                for g in gen_states:
                    prob = p_s[i][g]
                    if prob <= 0.0:
                        continue
                    normalized = _normalize_outcome(i, g)
                    term_untested += prob * W_star[i][normalized]

    aaub_term = 0.0
    if isinstance(aaub_star, Mapping):
        fixed_enabled = bool(aaub_star.get("fixed_enabled", True))
        p12_enabled = bool(aaub_star.get("p12_enabled", False))
        u_map = aaub_star.get("u", {}) if isinstance(aaub_star.get("u"), Mapping) else {}
        v_map = aaub_star.get("v", {}) if isinstance(aaub_star.get("v"), Mapping) else {}

        for i in untested_individuals:
            if fixed_enabled:
                aaub_term += float(u_map.get(i, 0.0))
            if p12_enabled:
                person_probs = p_s.get(i, {})
                p12_i = float(person_probs.get(1, 0.0) + person_probs.get(2, 0.0))
                aaub_term += float(v_map.get(i, 0.0)) * p12_i

    # Part 4: Edge feature terms
    edge_term = 0.0
    if W_edge_star is not None and pedigree_edges:
        tested_obs = dict(evidence_state)

        def _edge_block_maps(payload):
            if not isinstance(payload, Mapping):
                return {}
            if "__mode__" not in payload and "__blocks__" not in payload:
                return {"raw": payload}
            blocks = payload.get("__blocks__", ())
            if not isinstance(blocks, Sequence):
                blocks = ()
            resolved = {}
            for block in blocks:
                if not isinstance(block, str):
                    continue
                block_map = payload.get(block)
                if isinstance(block_map, Mapping):
                    resolved[block] = block_map
            if not resolved:
                for candidate in ("raw", "residual"):
                    block_map = payload.get(candidate)
                    if isinstance(block_map, Mapping):
                        resolved[candidate] = block_map
            return resolved

        edge_blocks = _edge_block_maps(W_edge_star)
        if not edge_blocks:
            edge_blocks = {"raw": W_edge_star}

        from ..models.belief import get_pairwise_marginals
        pair_marginals_all = None
        if infer is not None:
            cache_key = None
            if isinstance(feature_cache, dict):
                cache_key = (
                    "edge_pairwise",
                    evidence_state,
                    tuple(pedigree_edges),
                    tuple(genes) if genes else tuple(),
                )
                pair_marginals_all = feature_cache.get(cache_key)
            if pair_marginals_all is None:
                pair_marginals_all = get_pairwise_marginals(
                    infer,
                    individuals,
                    gen_states,
                    tested_obs,
                    pedigree_edges,
                    genes=genes if genes else None,
                )
                if isinstance(feature_cache, dict) and cache_key is not None:
                    feature_cache[cache_key] = pair_marginals_all

        def _coerce_gene_value(outcome, gene_idx):
            if isinstance(outcome, tuple):
                if gene_idx < len(outcome):
                    return outcome[gene_idx]
                return outcome[0]
            return outcome

        if genes:
            gene_list = list(genes)
            per_gene_probs = {}
            if hasattr(posterior_entry, "get_per_gene_probs"):
                per_gene_probs = posterior_entry.get_per_gene_probs() or {}
            for gene_idx, gene in enumerate(gene_list):
                raw_map = edge_blocks.get("raw", {}).get(gene, {}) if isinstance(edge_blocks.get("raw"), Mapping) else {}
                residual_map = (
                    edge_blocks.get("residual", {}).get(gene, {})
                    if isinstance(edge_blocks.get("residual"), Mapping)
                    else {}
                )
                if not raw_map and not residual_map:
                    continue
                gene_pairs = {}
                if isinstance(pair_marginals_all, Mapping):
                    gene_pairs = pair_marginals_all.get(gene, {}) or {}
                for parent, child in pedigree_edges:
                    edge_raw = raw_map.get((parent, child), {}) if isinstance(raw_map, Mapping) else {}
                    edge_resid = residual_map.get((parent, child), {}) if isinstance(residual_map, Mapping) else {}
                    if not edge_raw and not edge_resid:
                        continue
                    pm = gene_pairs.get((parent, child), {})
                    if not pm:
                        parent_probs = per_gene_probs.get(gene, {}).get(parent, p_s.get(parent, {}))
                        child_probs = per_gene_probs.get(gene, {}).get(child, p_s.get(child, {}))
                        for gp in gen_states:
                            for gc in gen_states:
                                pm[(gp, gc)] = float(parent_probs.get(gp, 0.0)) * float(child_probs.get(gc, 0.0))
                        if parent in tested_obs and child in tested_obs:
                            obs_p = _coerce_gene_value(tested_obs[parent], gene_idx)
                            obs_c = _coerce_gene_value(tested_obs[child], gene_idx)
                            pm = {
                                (gp, gc): (1.0 if gp == obs_p and gc == obs_c else 0.0)
                                for gp in gen_states
                                for gc in gen_states
                            }
                    parent_probs = per_gene_probs.get(gene, {}).get(parent, p_s.get(parent, {}))
                    child_probs = per_gene_probs.get(gene, {}).get(child, p_s.get(child, {}))
                    for (gp, gc), joint_prob in pm.items():
                        if joint_prob <= 0.0 and not edge_resid:
                            continue
                        if edge_raw:
                            edge_term += float(joint_prob) * float(edge_raw.get((gp, gc), 0.0))
                        if edge_resid:
                            residual = float(joint_prob) - float(parent_probs.get(gp, 0.0)) * float(child_probs.get(gc, 0.0))
                            if abs(residual) > 0.0:
                                edge_term += residual * float(edge_resid.get((gp, gc), 0.0))
        else:
            raw_map = edge_blocks.get("raw", {})
            residual_map = edge_blocks.get("residual", {})
            for parent, child in pedigree_edges:
                edge_raw = raw_map.get((parent, child), {}) if isinstance(raw_map, Mapping) else {}
                edge_resid = residual_map.get((parent, child), {}) if isinstance(residual_map, Mapping) else {}
                if not edge_raw and not edge_resid:
                    continue
                pm = {}
                if isinstance(pair_marginals_all, Mapping):
                    pm = pair_marginals_all.get((parent, child), {}) or {}
                if not pm:
                    parent_probs = p_s.get(parent, {})
                    child_probs = p_s.get(child, {})
                    for gp in gen_states:
                        for gc in gen_states:
                            pm[(gp, gc)] = float(parent_probs.get(gp, 0.0)) * float(child_probs.get(gc, 0.0))
                    if parent in tested_obs and child in tested_obs:
                        obs_p = tested_obs[parent]
                        obs_c = tested_obs[child]
                        pm = {
                            (gp, gc): (1.0 if gp == obs_p and gc == obs_c else 0.0)
                            for gp in gen_states
                            for gc in gen_states
                        }
                parent_probs = p_s.get(parent, {})
                child_probs = p_s.get(child, {})
                for (gp, gc), joint_prob in pm.items():
                    if joint_prob <= 0.0 and not edge_resid:
                        continue
                    if edge_raw:
                        edge_term += float(joint_prob) * float(edge_raw.get((gp, gc), 0.0))
                    if edge_resid:
                        residual = float(joint_prob) - float(parent_probs.get(gp, 0.0)) * float(child_probs.get(gc, 0.0))
                        if abs(residual) > 0.0:
                            edge_term += residual * float(edge_resid.get((gp, gc), 0.0))

    trio_term = 0.0
    if W_trio_star is not None and pedigree_trios and infer is not None:
        trio_blocks = resolve_trio_blocks(W_trio_star)
        trio_sharing = (
            W_trio_star.get("__sharing__", "free")
            if isinstance(W_trio_star, Mapping)
            else "free"
        )
        trio_share_key_map, _ = build_trio_share_key_map(
            [tuple(trio) for trio in pedigree_trios],
            trio_sharing=trio_sharing,
            share_key_entries=(
                W_trio_star.get("__share_keys__", ())
                if isinstance(W_trio_star, Mapping)
                else ()
            ),
        )
        bundle = build_state_feature_bundle(
            infer=infer,
            individuals=individuals,
            gen_states=gen_states,
            evidence_state=evidence_state,
            pedigree_trios=[tuple(trio) for trio in pedigree_trios],
            posterior_entry=posterior_entry,
            genes=tuple(genes) if genes else None,
            feature_cache=feature_cache,
            trio_sharing=trio_sharing,
            share_key_map=trio_share_key_map,
        )

        for gene, _trio, share_key, trio_dist, pure3_residuals in iter_trio_feature_rows(bundle):
            if genes:
                raw_map = trio_blocks.get("raw", {}).get(gene, {}) if isinstance(trio_blocks.get("raw"), Mapping) else {}
                pure3_map = trio_blocks.get("pure3", {}).get(gene, {}) if isinstance(trio_blocks.get("pure3"), Mapping) else {}
            else:
                raw_map = trio_blocks.get("raw", {})
                pure3_map = trio_blocks.get("pure3", {})
            trio_raw = raw_map.get(share_key, {}) if isinstance(raw_map, Mapping) else {}
            trio_pure3 = pure3_map.get(share_key, {}) if isinstance(pure3_map, Mapping) else {}
            if not trio_raw and not trio_pure3:
                continue
            for g_f in gen_states:
                for g_m in gen_states:
                    for g_c in gen_states:
                        triple = (g_f, g_m, g_c)
                        if trio_raw:
                            trio_term += float(trio_dist.get(triple, 0.0)) * float(trio_raw.get(triple, 0.0))
                        if trio_pure3:
                            trio_term += float(pure3_residuals.get(triple, 0.0)) * float(trio_pure3.get(triple, 0.0))

    theta_correction = evaluate_theta_model_correction(
        s,
        belief=belief,
        individuals=individuals,
        gen_states=gen_states,
        pedigree=pedigree,
        theta_model=theta_model,
        theta_model_spec=theta_model_spec,
    )
    myopic_adp_term = myopic_adp_term_value(
        s,
        myopic_adp_star,
        belief=belief,
        individuals=individuals,
        pedigree=pedigree,
        genes=genes,
    )
    oracle_adp_term = oracle_adp_term_value(
        s,
        oracle_adp_star,
        belief=belief,
        individuals=individuals,
    )
    regime_residual_term = regime_residual_term_value(
        s,
        regime_residual_star,
        belief=belief,
        individuals=individuals,
        pedigree=pedigree,
        genes=genes,
    )
    oracle_plumbing_mode = None
    if isinstance(oracle_adp_star, Mapping):
        oracle_diagnostics = oracle_adp_star.get("diagnostics", {})
        if not isinstance(oracle_diagnostics, Mapping):
            oracle_diagnostics = {}
        oracle_plumbing_mode = (
            oracle_adp_star.get("oracle_plumbing_mode")
            or oracle_adp_star.get("plumbing_mode")
            or oracle_diagnostics.get("oracle_plumbing_mode")
            or oracle_diagnostics.get("plumbing_mode")
        )
    if oracle_plumbing_mode == "oracle_only_fixed_phi":
        return oracle_adp_term

    return (
        _theta_for_state(s)
        + term_tested
        + term_untested
        + aaub_term
        + edge_term
        + trio_term
        + theta_correction
        + myopic_adp_term
        + oracle_adp_term
        + regime_residual_term
    )
