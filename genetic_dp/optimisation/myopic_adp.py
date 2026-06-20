from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
import math
import os


FEATURE_BANK_FB0_PROXY = "FB0_PROXY"
FEATURE_BANK_FB1_STRICT = "FB1_STRICT"
FEATURE_BANK_FB1R_CALIB = "FB1R_CALIB"
FEATURE_BANK_FB2_HYBRID = "FB2_HYBRID"
FEATURE_BANK_ABCD_HAND = "ABCD_HAND"
SUPPORTED_FEATURE_BANKS = (
    FEATURE_BANK_FB0_PROXY,
    FEATURE_BANK_FB1_STRICT,
    FEATURE_BANK_FB1R_CALIB,
    FEATURE_BANK_FB2_HYBRID,
    FEATURE_BANK_ABCD_HAND,
)

FEATURE_SEMANTICS_PROXY_V0 = "proxy_v0"
FEATURE_SEMANTICS_SPEC_V1 = "spec_v1"
FEATURE_SEMANTICS_FB1R_CALIB = "fb1r_calib"
FEATURE_SEMANTICS_FB2_HYBRID = "fb2_hybrid"
FEATURE_SEMANTICS_ABCD_HAND = "abcd_hand_v1"
SUPPORTED_FEATURE_SEMANTICS = (
    FEATURE_SEMANTICS_PROXY_V0,
    FEATURE_SEMANTICS_SPEC_V1,
    FEATURE_SEMANTICS_FB1R_CALIB,
    FEATURE_SEMANTICS_FB2_HYBRID,
    FEATURE_SEMANTICS_ABCD_HAND,
)
_FEATURE_BANK_ALIASES = {
    "fb0_proxy": FEATURE_BANK_FB0_PROXY,
    "proxy_v0": FEATURE_BANK_FB0_PROXY,
    "v0_proxy": FEATURE_BANK_FB0_PROXY,
    "fb1_strict": FEATURE_BANK_FB1_STRICT,
    "spec_v1": FEATURE_BANK_FB1_STRICT,
    "strict": FEATURE_BANK_FB1_STRICT,
    "fb1r_calib": FEATURE_BANK_FB1R_CALIB,
    "spec_v1_soft": FEATURE_BANK_FB1R_CALIB,
    "spec_v1_calib": FEATURE_BANK_FB1R_CALIB,
    "calibrated_spec": FEATURE_BANK_FB1R_CALIB,
    "fb2_hybrid": FEATURE_BANK_FB2_HYBRID,
    "hybrid": FEATURE_BANK_FB2_HYBRID,
    "abcd": FEATURE_BANK_ABCD_HAND,
    "abcd_hand": FEATURE_BANK_ABCD_HAND,
    "abcd_hand_v1": FEATURE_BANK_ABCD_HAND,
}
_FEATURE_BANK_TO_SEMANTICS = {
    FEATURE_BANK_FB0_PROXY: FEATURE_SEMANTICS_PROXY_V0,
    FEATURE_BANK_FB1_STRICT: FEATURE_SEMANTICS_SPEC_V1,
    FEATURE_BANK_FB1R_CALIB: FEATURE_SEMANTICS_FB1R_CALIB,
    FEATURE_BANK_FB2_HYBRID: FEATURE_SEMANTICS_FB2_HYBRID,
    FEATURE_BANK_ABCD_HAND: FEATURE_SEMANTICS_ABCD_HAND,
}
_EPS = 1e-12


REGIME_RESIDUAL_BASE_CANDIDATE_FEATURES = (
    "bridge_depth_mass",
    "descendant_bridge_mass",
    "sibling_breadth",
    "collateral_block_count",
    "frontier_carrier_variance",
    "bridge_dominated_bridge_depth_mass",
    "breadth_dominated_sibling_breadth",
    "collateral_active_collateral_block_count",
    "allele_asymmetry_high_bridge_depth_mass",
    "allele_asymmetry_low_frontier_carrier_variance",
    "reward_pressure_high_bridge_depth_mass",
    "reward_pressure_high_collateral_block_count",
    "test_cost_pressure_high_frontier_carrier_variance",
)
REGIME_RESIDUAL_CALIB_CANDIDATE_FEATURES = (
    "strict_bridge_depth_uncertainty",
    "strict_descendant_bridge_uncertainty",
    "strict_sibling_breadth_uncertainty",
    "strict_collateral_block_count",
    "strict_frontier_carrier_variance",
    "soft_bridge_dominated_bridge_depth",
    "soft_breadth_dominated_sibling_breadth",
    "soft_reward_pressure_bridge_depth",
    "soft_reward_pressure_collateral_block_count",
    "soft_test_cost_pressure_frontier_carrier_variance",
    "carrier_depth_mass_honest",
    "child_count_weighted_carrier_mass_honest",
    "parent_pair_sibling_count_honest",
    "parent_pair_collateral_proxy_count_honest",
    "all_untested_carrier_variance_honest",
)
REGIME_RESIDUAL_HYBRID_CANDIDATE_FEATURES = (
    "parent_pair_breadth_active_sibling_count_honest",
    "carrier_depth_mass_honest",
    "child_count_weighted_carrier_mass_honest",
    "all_untested_carrier_variance_honest",
    "parent_pair_collateral_proxy_count_honest",
    "child_count_bridge_dominated_carrier_depth_honest",
    "collateral_active_parent_pair_block_count_honest",
    "reward_pressure_carrier_depth_mass_honest",
    "reward_pressure_parent_pair_collateral_count_honest",
    "test_cost_pressure_all_untested_carrier_variance_honest",
)
ABCD_HAND_REGIME_FEATURES = (
    "collateral_active_parent_pair_block_count_honest",
    "all_untested_carrier_variance_honest",
    "allele_asymmetry_high_gene_GeneA_carrier_depth_mass_honest",
)
ABCD16_DIRECT_MYOPIC_FEATURES = (
    "best_second_best_test_margin",
    "boundary_state_indicator",
    "bridge_depth_mass",
    "collateral_block_count",
    "cost_adjusted_continuation_margin",
    "descendant_bridge_mass",
    "frontier_carrier_mass",
    "frontier_carrier_max",
    "frontier_carrier_variance",
    "myopic_stop_gate_pressure",
    "myopic_tests",
    "sibling_breadth",
    "stop_test_margin",
)
ABCD16_DIRECT_OBSERVED_EXTRA_MYOPIC_FEATURES = (
    "frontier_carrier_variance",
    "sibling_breadth",
)
ABCD16_DIRECT_CANONICAL14_MYOPIC_FEATURES = tuple(
    feature
    for feature in ABCD16_DIRECT_MYOPIC_FEATURES
    if feature not in ABCD16_DIRECT_OBSERVED_EXTRA_MYOPIC_FEATURES
)
ABCD16_DIRECT_MYOPIC_EMBEDDING_ORDER = (
    "descendant_bridge_mass",
    "bridge_depth_mass",
    "frontier_carrier_mass",
    "myopic_tests",
    "frontier_carrier_max",
    "collateral_block_count",
    "boundary_state_indicator",
    "stop_test_margin",
    "best_second_best_test_margin",
    "myopic_stop_gate_pressure",
    "cost_adjusted_continuation_margin",
    "frontier_carrier_variance",
    "sibling_breadth",
)
ABCD16_DIRECT_CANONICAL14_MYOPIC_EMBEDDING_ORDER = tuple(
    feature
    for feature in ABCD16_DIRECT_MYOPIC_EMBEDDING_ORDER
    if feature in ABCD16_DIRECT_CANONICAL14_MYOPIC_FEATURES
)
ABCD16_DIRECT_REGIME_FEATURES = (
    "all_untested_carrier_variance_honest",
    "allele_asymmetry_high_gene_GeneA_carrier_depth_mass_honest",
    "collateral_active_parent_pair_block_count_honest",
)
ABCD16_DIRECT_CANONICAL14_REGIME_FEATURES = ABCD16_DIRECT_REGIME_FEATURES
ABCD16_DIRECT_REGIME_EMBEDDING_ORDER = ABCD_HAND_REGIME_FEATURES
ABCD16_DIRECT_FEATURES = ABCD16_DIRECT_MYOPIC_FEATURES + ABCD16_DIRECT_REGIME_FEATURES
ABCD16_DIRECT_CANONICAL14_FEATURES = (
    ABCD16_DIRECT_CANONICAL14_MYOPIC_FEATURES
    + ABCD16_DIRECT_CANONICAL14_REGIME_FEATURES
)
REGIME_RESIDUAL_CANDIDATE_FEATURES = REGIME_RESIDUAL_BASE_CANDIDATE_FEATURES


def resolve_feature_bank(value=None, *, default=FEATURE_BANK_FB0_PROXY, require=False):
    raw = value
    if raw is None:
        raw = os.getenv("GAUGED_REGIME_FEATURE_BANK")
    if raw is None or str(raw).strip() == "":
        if require:
            raise ValueError(
                "GAUGED_REGIME_FEATURE_BANK must be set to one of "
                "'FB0_PROXY', 'FB1_STRICT', 'FB1R_CALIB', 'FB2_HYBRID', or "
                "'ABCD_HAND' "
                "for benchmark-facing gauged-regime residual runs."
            )
        raw = default
    key = str(raw).strip().lower()
    if key in _FEATURE_BANK_ALIASES:
        return _FEATURE_BANK_ALIASES[key]
    if str(raw).strip().upper() in SUPPORTED_FEATURE_BANKS:
        return str(raw).strip().upper()
    raise ValueError(
        f"Unknown GAUGED_REGIME_FEATURE_BANK={raw!r}; expected one of "
        "'FB0_PROXY', 'FB1_STRICT', 'FB1R_CALIB', 'FB2_HYBRID', or "
        "'ABCD_HAND'."
    )


def feature_semantics_for_bank(feature_bank):
    bank = resolve_feature_bank(feature_bank, require=True)
    return _FEATURE_BANK_TO_SEMANTICS[bank]


def resolve_feature_semantics(value=None, *, default=FEATURE_SEMANTICS_PROXY_V0, require=False):
    raw = value
    if raw is None:
        raw = os.getenv("GAUGED_REGIME_FEATURE_SEMANTICS")
    if raw is None or str(raw).strip() == "":
        if require:
            raise ValueError(
                "GAUGED_REGIME_FEATURE_SEMANTICS must be set to "
                "'proxy_v0', 'spec_v1', 'fb1r_calib', 'fb2_hybrid', or "
                "'abcd_hand_v1' for "
                "gauged-regime residual runs."
            )
        raw = default
    semantics = str(raw).strip().lower()
    if semantics in _FEATURE_BANK_ALIASES:
        semantics = feature_semantics_for_bank(_FEATURE_BANK_ALIASES[semantics])
    if semantics in {"abcd", "abcd_hand"}:
        semantics = FEATURE_SEMANTICS_ABCD_HAND
    if semantics not in SUPPORTED_FEATURE_SEMANTICS:
        raise ValueError(
            f"Unknown GAUGED_REGIME_FEATURE_SEMANTICS={raw!r}; "
            "expected 'proxy_v0', 'spec_v1', 'fb1r_calib', 'fb2_hybrid', "
            "or 'abcd_hand_v1'."
        )
    return semantics


def regime_residual_candidate_features(feature_bank=None):
    bank = resolve_feature_bank(feature_bank)
    if bank == FEATURE_BANK_ABCD_HAND:
        return tuple(ABCD_HAND_REGIME_FEATURES)
    features = list(REGIME_RESIDUAL_BASE_CANDIDATE_FEATURES)
    if bank == FEATURE_BANK_FB1R_CALIB:
        features.extend(REGIME_RESIDUAL_CALIB_CANDIDATE_FEATURES)
    elif bank == FEATURE_BANK_FB2_HYBRID:
        features = list(REGIME_RESIDUAL_HYBRID_CANDIDATE_FEATURES)
    return tuple(features)


def regime_residual_v2_candidate_features(genes=None, *, feature_bank=None):
    """Return deterministic V2 structural features, including per-gene LowHigh terms."""

    bank = resolve_feature_bank(feature_bank)
    if bank == FEATURE_BANK_ABCD_HAND:
        return tuple(ABCD_HAND_REGIME_FEATURES)
    features = list(regime_residual_candidate_features(bank))
    for gene in tuple(genes or ()):
        prefix = f"gene_{gene}"
        if bank == FEATURE_BANK_FB2_HYBRID:
            features.extend(
                [
                    f"{prefix}_carrier_depth_mass_honest",
                    f"{prefix}_child_count_weighted_carrier_mass_honest",
                    f"{prefix}_all_untested_carrier_variance_honest",
                    f"allele_asymmetry_high_{prefix}_carrier_depth_mass_honest",
                    f"reward_pressure_{prefix}_child_count_weighted_carrier_mass_honest",
                    f"test_cost_pressure_{prefix}_all_untested_carrier_variance_honest",
                ]
            )
            continue
        features.extend(
            [
                f"{prefix}_bridge_depth_mass",
                f"{prefix}_descendant_bridge_mass",
                f"{prefix}_frontier_carrier_variance",
                f"allele_asymmetry_high_{prefix}_bridge_depth_mass",
                f"reward_pressure_high_{prefix}_descendant_bridge_mass",
                f"test_cost_pressure_high_{prefix}_frontier_carrier_variance",
            ]
        )
        if bank == FEATURE_BANK_FB1R_CALIB:
            features.extend(
                [
                    f"{prefix}_strict_bridge_depth_uncertainty",
                    f"{prefix}_strict_descendant_bridge_uncertainty",
                    f"{prefix}_strict_frontier_carrier_variance",
                    f"{prefix}_carrier_depth_mass_honest",
                    f"{prefix}_child_count_weighted_carrier_mass_honest",
                    f"soft_reward_pressure_{prefix}_descendant_bridge_uncertainty",
                    f"soft_test_cost_pressure_{prefix}_frontier_carrier_variance",
                ]
            )
    return tuple(features)


def state_key(state):
    return tuple(sorted(state))


def _posterior(entry):
    payload = entry[0] if isinstance(entry, tuple) else entry
    return payload.marginals if hasattr(payload, "marginals") else payload


def _per_gene_probs(entry):
    payload = entry[0] if isinstance(entry, tuple) else entry
    if hasattr(payload, "get_per_gene_probs"):
        return payload.get_per_gene_probs() or {}
    return {}


def _carrier_prob(dist):
    if not isinstance(dist, Mapping):
        return 0.0
    return float(dist.get(1, 0.0) + dist.get(2, 0.0))


def _numeric_leaf_values(value):
    if isinstance(value, Mapping):
        values = []
        for child in value.values():
            values.extend(_numeric_leaf_values(child))
        return values
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return []
    if not isfinite(numeric):
        return []
    return [numeric]


def _mean_abs(values):
    clean = [abs(float(value)) for value in values if isfinite(float(value))]
    return sum(clean) / float(len(clean)) if clean else 0.0


def _plain_nested_mapping(value):
    if isinstance(value, Mapping):
        return {str(key): _plain_nested_mapping(child) for key, child in value.items()}
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    return numeric


def regime_parameter_gates(
    *,
    genes=None,
    a_gene=None,
    b_gene=None,
    delta_gene=None,
    fixed_cost=0.0,
    variable_cost=0.0,
):
    """Return deterministic non-oracle economic gates for regime residual features."""

    gene_scores = []
    if genes:
        for gene in genes:
            score = 0.0
            for container in (a_gene, b_gene, delta_gene):
                raw = container.get(gene, {}) if isinstance(container, Mapping) else {}
                score += _mean_abs(_numeric_leaf_values(raw))
            gene_scores.append(score)
    if gene_scores:
        reward_pressure = sum(gene_scores) / float(len(gene_scores))
    else:
        reward_pressure = sum(
            _mean_abs(_numeric_leaf_values(container))
            for container in (a_gene, b_gene, delta_gene)
        )
    try:
        cost_pressure_raw = float(fixed_cost) + float(variable_cost)
    except (TypeError, ValueError):
        cost_pressure_raw = 0.0
    cost_pressure = cost_pressure_raw / max(abs(reward_pressure), 1e-12)
    return {
        "reward_pressure_score": float(reward_pressure),
        "test_cost_pressure_score": float(cost_pressure),
        "reward_pressure_high_region": 1.0 if reward_pressure >= 0.83 else 0.0,
        "test_cost_pressure_high_region": 1.0 if cost_pressure >= 0.035 else 0.0,
        "genes": list(genes or ()),
        "a_gene": _plain_nested_mapping(a_gene or {}),
        "b_gene": _plain_nested_mapping(b_gene or {}),
        "delta_gene": _plain_nested_mapping(delta_gene or {}),
        "fixed_cost": float(fixed_cost or 0.0),
        "variable_cost": float(variable_cost or 0.0),
    }


def _carrier_mass(entry, person, genes=None):
    if genes:
        per_gene = _per_gene_probs(entry)
        if per_gene:
            return sum(_carrier_prob(per_gene.get(gene, {}).get(person, {})) for gene in genes)
    posterior = _posterior(entry)
    return _carrier_prob(posterior.get(person, {})) if isinstance(posterior, Mapping) else 0.0


def _gene_carrier_masses(entry, untested, genes=None):
    if not genes:
        return []
    per_gene = _per_gene_probs(entry)
    if not per_gene:
        return []
    masses = []
    for gene in genes:
        gene_probs = per_gene.get(gene, {})
        masses.append(sum(_carrier_prob(gene_probs.get(person, {})) for person in untested))
    return [float(value) for value in masses]


def _depths(pedigree, individuals):
    if not pedigree or not hasattr(pedigree, "get_parents"):
        return {person: 0 for person in individuals}
    memo = {}

    def _depth(person):
        if person in memo:
            return memo[person]
        parents = tuple(pedigree.get_parents(person) or ())
        if not parents:
            memo[person] = 0
        else:
            memo[person] = 1 + max(_depth(parent) for parent in parents)
        return memo[person]

    return {person: _depth(person) for person in individuals}


def _child_counts(pedigree, individuals):
    counts = {person: 0 for person in individuals}
    if not pedigree or not hasattr(pedigree, "get_parents"):
        return counts
    for child in individuals:
        for parent in tuple(pedigree.get_parents(child) or ()):
            if parent in counts:
                counts[parent] += 1
    return counts


def _sibling_block_count(pedigree, untested):
    if not pedigree or not hasattr(pedigree, "get_parents"):
        return 0.0
    parent_sets = {}
    for person in untested:
        parents = tuple(sorted(pedigree.get_parents(person) or ()))
        if len(parents) != 2:
            continue
        parent_sets.setdefault(parents, 0)
        parent_sets[parents] += 1
    return float(sum(max(0, count - 1) for count in parent_sets.values()))


def _collateral_count(pedigree, untested):
    if not pedigree or not hasattr(pedigree, "get_parents"):
        return 0.0
    child_by_parents = {}
    for person in untested:
        parents = tuple(sorted(pedigree.get_parents(person) or ()))
        if len(parents) != 2:
            continue
        child_by_parents.setdefault(parents, []).append(person)
    return float(sum(1 for children in child_by_parents.values() if len(children) >= 2))


def _children_by_parent(pedigree, individuals):
    children = {person: [] for person in individuals}
    if not pedigree or not hasattr(pedigree, "get_parents"):
        return children
    for child in individuals:
        for parent in tuple(pedigree.get_parents(child) or ()):
            children.setdefault(parent, [])
            children[parent].append(child)
    return children


def _descendants_by_person(pedigree, individuals):
    child_map = _children_by_parent(pedigree, individuals)
    descendants = {person: set() for person in individuals}

    def _walk(root, person):
        for child in child_map.get(person, ()):
            if child in descendants[root]:
                continue
            descendants[root].add(child)
            _walk(root, child)

    for person in individuals:
        _walk(person, person)
    return descendants


def _undirected_adjacency(pedigree, individuals):
    adjacency = {person: set() for person in individuals}
    if not pedigree or not hasattr(pedigree, "get_parents"):
        return adjacency
    for child in individuals:
        adjacency.setdefault(child, set())
        for parent in tuple(pedigree.get_parents(child) or ()):
            if parent not in adjacency:
                continue
            adjacency[parent].add(child)
            adjacency[child].add(parent)
    return adjacency


def _component_labels_without(adjacency, removed):
    labels = {}
    component_id = 0
    for start in sorted(node for node in adjacency if node != removed):
        if start in labels:
            continue
        stack = [start]
        labels[start] = component_id
        while stack:
            node = stack.pop()
            for neighbor in adjacency.get(node, ()):
                if neighbor == removed or neighbor in labels:
                    continue
                labels[neighbor] = component_id
                stack.append(neighbor)
        component_id += 1
    return labels


def _component_labels(adjacency):
    labels = {}
    component_id = 0
    for start in sorted(adjacency):
        if start in labels:
            continue
        stack = [start]
        labels[start] = component_id
        while stack:
            node = stack.pop()
            for neighbor in adjacency.get(node, ()):
                if neighbor in labels:
                    continue
                labels[neighbor] = component_id
                stack.append(neighbor)
        component_id += 1
    return labels


def _bridge_scores(pedigree, individuals):
    adjacency = _undirected_adjacency(pedigree, individuals)
    raw_scores = {}
    for person in individuals:
        neighbors = [neighbor for neighbor in adjacency.get(person, ()) if neighbor in adjacency]
        if not neighbors:
            raw_scores[person] = 0.0
            continue
        labels = _component_labels_without(adjacency, person)
        split_count = len({labels[neighbor] for neighbor in neighbors if neighbor in labels})
        raw_scores[person] = float(max(0, split_count - 1))
    max_score = max(raw_scores.values(), default=1.0)
    if max_score <= 0.0:
        return {person: 0.0 for person in raw_scores}
    return {person: float(score) / max_score for person, score in raw_scores.items()}


def _collateral_blocks(pedigree, individuals):
    child_map = _children_by_parent(pedigree, individuals)
    descendants = _descendants_by_person(pedigree, individuals)
    blocks = set()
    for _parent, children in child_map.items():
        if len(children) < 2:
            continue
        for child in children:
            block = frozenset({child} | set(descendants.get(child, ())))
            if block:
                blocks.add(block)
    return tuple(sorted(blocks, key=lambda block: tuple(sorted(block))))


def _has_explicit_reward_maps(regime_gates):
    gates = regime_gates if isinstance(regime_gates, Mapping) else {}
    a_gene = gates.get("a_gene", {})
    b_gene = gates.get("b_gene", {})
    delta_gene = gates.get("delta_gene", {})
    return any(isinstance(container, Mapping) and container for container in (a_gene, b_gene, delta_gene))


def _explicit_reward_weight(regime_gates, gene, person):
    gates = regime_gates if isinstance(regime_gates, Mapping) else {}
    a_gene = gates.get("a_gene", {})
    b_gene = gates.get("b_gene", {})
    delta_gene = gates.get("delta_gene", {})

    def _value(container, default=0.0):
        if not isinstance(container, Mapping):
            return default
        gene_values = container.get(gene, {})
        if not isinstance(gene_values, Mapping):
            return default
        try:
            return float(gene_values.get(person, default) or 0.0)
        except (TypeError, ValueError):
            return default

    a_value = _value(a_gene)
    b_value = _value(b_gene)
    delta_value = _value(delta_gene)
    return abs(a_value) * (1.0 + abs(delta_value)) + abs(b_value)


def _reward_weight(regime_gates, gene, person):
    if not _has_explicit_reward_maps(regime_gates):
        return 1.0
    return _explicit_reward_weight(regime_gates, gene, person)


def _component_reward_weights(pedigree, individuals, genes, regime_gates):
    gene_list = tuple(genes or ()) or ("gene",)
    if not _has_explicit_reward_maps(regime_gates):
        return {
            person: {gene: 1.0 for gene in gene_list}
            for person in individuals
        }
    adjacency = _undirected_adjacency(pedigree, individuals)
    labels = _component_labels(adjacency)
    component_max = {}
    for person in individuals:
        label = labels.get(person)
        if label is None:
            continue
        bucket = component_max.setdefault(label, {gene: 0.0 for gene in gene_list})
        for gene in gene_list:
            bucket[gene] = max(bucket[gene], _explicit_reward_weight(regime_gates, gene, person))
    return {
        person: dict(component_max.get(labels.get(person), {}))
        for person in individuals
    }


def _per_gene_distributions(entry, person, genes=None):
    gene_list = tuple(genes or ())
    if gene_list:
        per_gene = _per_gene_probs(entry)
        if isinstance(per_gene, Mapping) and per_gene:
            return {
                gene: (per_gene.get(gene, {}) or {}).get(person, {})
                for gene in gene_list
            }
        posterior = _posterior(entry)
        dist = posterior.get(person, {}) if isinstance(posterior, Mapping) else {}
        return {gene: dist for gene in gene_list}
    posterior = _posterior(entry)
    return {"gene": posterior.get(person, {}) if isinstance(posterior, Mapping) else {}}


def _spec_person_quantities(entry, untested, individuals=None, pedigree=None, genes=None, regime_gates=None):
    gene_list = tuple(genes or ()) or ("gene",)
    individual_list = tuple(individuals or untested)
    component_reward = _component_reward_weights(pedigree, individual_list, gene_list, regime_gates)
    explicit_reward_maps = _has_explicit_reward_maps(regime_gates)
    carrier_by_gene = {}
    uncertainty_by_gene = {}
    weighted_uncertainty_by_gene = {}
    carrier_plus = {}
    uncertainty_mass = {}
    for person in untested:
        dists = _per_gene_distributions(entry, person, genes=genes)
        product_noncarrier = 1.0
        carrier_by_gene[person] = {}
        uncertainty_by_gene[person] = {}
        weighted_uncertainty_by_gene[person] = {}
        total_uncertainty = 0.0
        for gene in gene_list:
            carrier = _carrier_prob(dists.get(gene, {}))
            carrier_by_gene[person][gene] = carrier
            uncertainty = carrier * (1.0 - carrier)
            uncertainty_by_gene[person][gene] = uncertainty
            direct_weight = _explicit_reward_weight(regime_gates, gene, person) if explicit_reward_maps else 1.0
            propagated_weight = component_reward.get(person, {}).get(gene, 0.0)
            weight = max(direct_weight, propagated_weight)
            weighted = weight * uncertainty
            weighted_uncertainty_by_gene[person][gene] = weighted
            total_uncertainty += weighted
            product_noncarrier *= max(0.0, 1.0 - carrier)
        carrier_plus[person] = 1.0 - product_noncarrier
        uncertainty_mass[person] = float(total_uncertainty)
    return {
        "genes": gene_list,
        "carrier_by_gene": carrier_by_gene,
        "uncertainty_by_gene": uncertainty_by_gene,
        "weighted_uncertainty_by_gene": weighted_uncertainty_by_gene,
        "carrier_plus": carrier_plus,
        "uncertainty_mass": uncertainty_mass,
    }


def _variance(values):
    clean = [float(value) for value in values]
    if not clean:
        return 0.0
    mean = sum(clean) / float(len(clean))
    return float(sum((value - mean) ** 2 for value in clean) / float(len(clean)))


def _sibling_uncertainty_breadth(pedigree, untested, uncertainty_mass):
    if not pedigree or not hasattr(pedigree, "get_parents"):
        return 0.0
    parent_sets = {}
    active = [person for person in untested if uncertainty_mass.get(person, 0.0) > _EPS]
    for person in active:
        parents = tuple(sorted(pedigree.get_parents(person) or ()))
        if len(parents) != 2:
            continue
        parent_sets.setdefault(parents, []).append(person)
    return float(
        sum(
            sum(float(uncertainty_mass.get(person, 0.0)) for person in people)
            for people in parent_sets.values()
            if len(people) >= 2
        )
    )


def _active_collateral_block_count(pedigree, individuals, untested, uncertainty_mass):
    active_untested = {person for person in untested if uncertainty_mass.get(person, 0.0) > _EPS}
    count = 0
    for block in _collateral_blocks(pedigree, individuals):
        if block & active_untested:
            count += 1
    return float(count)


def _build_state_features_proxy_v0(
    state,
    *,
    belief,
    individuals,
    pedigree=None,
    genes=None,
    myopic_policy=None,
    myopic_values=None,
    myopic_residuals=None,
    regime_gates=None,
):
    """Small, deterministic feature bank for myopic-ADP experiments."""

    entry = belief[state]
    tested = {person for person, _ in state}
    untested = [person for person in individuals if person not in tested]
    n = max(1, len(individuals))
    stage = len(tested)
    depths = _depths(pedigree, individuals)
    child_counts = _child_counts(pedigree, individuals)

    carrier = {person: _carrier_mass(entry, person, genes=genes) for person in untested}
    carrier_values = list(carrier.values())
    carrier_total = float(sum(carrier_values))
    carrier_max = float(max(carrier_values)) if carrier_values else 0.0
    carrier_mean = carrier_total / float(len(carrier_values)) if carrier_values else 0.0
    carrier_var = (
        sum((value - carrier_mean) ** 2 for value in carrier_values) / float(len(carrier_values))
        if carrier_values
        else 0.0
    )
    bridge_mass = float(sum(carrier[p] * (1.0 + child_counts.get(p, 0)) for p in untested))
    depth_mass = float(sum(carrier[p] * (1.0 + depths.get(p, 0)) for p in untested))
    sibling_breadth = _sibling_block_count(pedigree, untested)
    collateral = _collateral_count(pedigree, untested)
    gene_masses = _gene_carrier_masses(entry, untested, genes=genes)
    allele_asymmetry = float(max(gene_masses) - min(gene_masses)) if len(gene_masses) >= 2 else 0.0

    action = myopic_policy.get(state) if isinstance(myopic_policy, Mapping) else None
    myopic_tests = 1.0 if action and action[0] == "test" else 0.0
    myopic_stops = 1.0 if action and action[0] == "stop" else 0.0
    myopic_value = 0.0
    if isinstance(myopic_values, Mapping):
        raw_value = myopic_values.get(state, 0.0)
        if raw_value is not None and isfinite(float(raw_value)):
            myopic_value = float(raw_value)
    myopic_residual = 0.0
    if isinstance(myopic_residuals, Mapping):
        raw_residual = myopic_residuals.get(state, 0.0)
        if raw_residual is not None and isfinite(float(raw_residual)):
            myopic_residual = float(raw_residual)

    features = {
        "bias": 1.0,
        "stage_fraction": float(stage) / float(n),
        "untested_fraction": float(len(untested)) / float(n),
        "frontier_carrier_mass": carrier_total,
        "frontier_carrier_max": carrier_max,
        "frontier_carrier_variance": carrier_var,
        "bridge_depth_mass": depth_mass,
        "descendant_bridge_mass": bridge_mass,
        "sibling_breadth": sibling_breadth,
        "collateral_block_count": collateral,
        "allele_asymmetry": allele_asymmetry,
        "myopic_tests": myopic_tests,
        "myopic_stops": myopic_stops,
        "myopic_policy_value": myopic_value,
        "myopic_bellman_residual": myopic_residual,
    }
    features["bridge_dominated_region"] = 1.0 if bridge_mass > carrier_total + 1e-12 else 0.0
    features["breadth_dominated_region"] = 1.0 if sibling_breadth >= 1.0 else 0.0
    features["collateral_active_region"] = 1.0 if collateral >= 1.0 else 0.0
    features["allele_asymmetry_high_region"] = 1.0 if allele_asymmetry > 0.05 else 0.0
    features["allele_asymmetry_low_region"] = 1.0 if allele_asymmetry <= 0.05 else 0.0
    features["myopic_adp_disagreement_region"] = 1.0 if myopic_residual > 1e-9 else 0.0
    residual_scale = 1.0 + float(stage)
    features["stop_test_margin"] = myopic_residual
    features["boundary_state_indicator"] = 1.0 if 0.0 < myopic_residual <= 1e-3 else 0.0
    features["best_second_best_test_margin"] = myopic_residual * (1.0 - myopic_stops)
    features["myopic_stop_gate_pressure"] = myopic_stops * (1.0 + myopic_residual)
    features["cost_adjusted_continuation_margin"] = myopic_residual / residual_scale
    gates = regime_gates if isinstance(regime_gates, Mapping) else {}
    reward_pressure_high = float(gates.get("reward_pressure_high_region", 0.0) or 0.0)
    test_cost_pressure_high = float(gates.get("test_cost_pressure_high_region", 0.0) or 0.0)
    features["reward_pressure_high_region"] = reward_pressure_high
    features["test_cost_pressure_high_region"] = test_cost_pressure_high
    features["bridge_dominated_bridge_depth_mass"] = (
        features["bridge_dominated_region"] * features["bridge_depth_mass"]
    )
    features["breadth_dominated_sibling_breadth"] = (
        features["breadth_dominated_region"] * features["sibling_breadth"]
    )
    features["collateral_active_collateral_block_count"] = (
        features["collateral_active_region"] * features["collateral_block_count"]
    )
    features["allele_asymmetry_high_bridge_depth_mass"] = (
        features["allele_asymmetry_high_region"] * features["bridge_depth_mass"]
    )
    features["allele_asymmetry_low_frontier_carrier_variance"] = (
        features["allele_asymmetry_low_region"] * features["frontier_carrier_variance"]
    )
    features["reward_pressure_high_bridge_depth_mass"] = (
        reward_pressure_high * features["bridge_depth_mass"]
    )
    features["reward_pressure_high_collateral_block_count"] = (
        reward_pressure_high * features["collateral_block_count"]
    )
    features["test_cost_pressure_high_frontier_carrier_variance"] = (
        test_cost_pressure_high * features["frontier_carrier_variance"]
    )
    if genes:
        per_gene = _per_gene_probs(entry)
        for gene in tuple(genes):
            prefix = f"gene_{gene}"
            gene_probs = per_gene.get(gene, {}) if isinstance(per_gene, Mapping) else {}
            gene_carrier = {
                person: _carrier_prob(gene_probs.get(person, {}))
                for person in untested
            }
            gene_values = list(gene_carrier.values())
            gene_total = float(sum(gene_values))
            gene_mean = gene_total / float(len(gene_values)) if gene_values else 0.0
            gene_var = (
                sum((value - gene_mean) ** 2 for value in gene_values) / float(len(gene_values))
                if gene_values
                else 0.0
            )
            gene_bridge = float(sum(gene_carrier[p] * (1.0 + child_counts.get(p, 0)) for p in untested))
            gene_depth = float(sum(gene_carrier[p] * (1.0 + depths.get(p, 0)) for p in untested))
            features[f"{prefix}_bridge_depth_mass"] = gene_depth
            features[f"{prefix}_descendant_bridge_mass"] = gene_bridge
            features[f"{prefix}_frontier_carrier_variance"] = gene_var
            features[f"allele_asymmetry_high_{prefix}_bridge_depth_mass"] = (
                features["allele_asymmetry_high_region"] * gene_depth
            )
            features[f"reward_pressure_high_{prefix}_descendant_bridge_mass"] = (
                reward_pressure_high * gene_bridge
            )
            features[f"test_cost_pressure_high_{prefix}_frontier_carrier_variance"] = (
                test_cost_pressure_high * gene_var
            )
    return features


def _build_state_features_spec_v1(
    state,
    *,
    belief,
    individuals,
    pedigree=None,
    genes=None,
    myopic_policy=None,
    myopic_values=None,
    myopic_residuals=None,
    regime_gates=None,
):
    features = _build_state_features_proxy_v0(
        state,
        belief=belief,
        individuals=individuals,
        pedigree=pedigree,
        genes=genes,
        myopic_policy=myopic_policy,
        myopic_values=myopic_values,
        myopic_residuals=myopic_residuals,
        regime_gates=regime_gates,
    )
    entry = belief[state]
    tested = {person for person, _ in state}
    untested = [person for person in individuals if person not in tested]
    depths = _depths(pedigree, individuals)
    max_depth = max(depths.values(), default=0)
    depth_score = {
        person: (1.0 + float(depths.get(person, 0))) / (1.0 + float(max_depth))
        for person in individuals
    }
    bridge_scores = _bridge_scores(pedigree, individuals)
    descendants = _descendants_by_person(pedigree, individuals)
    quantities = _spec_person_quantities(
        entry,
        untested,
        individuals=individuals,
        pedigree=pedigree,
        genes=genes,
        regime_gates=regime_gates,
    )
    gene_list = tuple(quantities["genes"])
    carrier_plus = quantities["carrier_plus"]
    uncertainty_mass = quantities["uncertainty_mass"]
    weighted_by_gene = quantities["weighted_uncertainty_by_gene"]
    carrier_by_gene = quantities["carrier_by_gene"]

    frontier = [person for person in untested if uncertainty_mass.get(person, 0.0) > _EPS]
    frontier_uncertainty = float(sum(uncertainty_mass.get(person, 0.0) for person in frontier))
    frontier_carrier_values = [carrier_plus.get(person, 0.0) for person in frontier]
    carrier_total = float(sum(frontier_carrier_values))
    carrier_max = float(max(frontier_carrier_values)) if frontier_carrier_values else 0.0
    carrier_var = _variance(frontier_carrier_values)

    bridge_pressure = float(
        sum(uncertainty_mass[p] * bridge_scores.get(p, 1.0) for p in frontier)
    )
    bridge_depth_mass = float(
        sum(
            uncertainty_mass[p] * bridge_scores.get(p, 1.0) * depth_score.get(p, 1.0)
            for p in frontier
        )
    )
    descendant_side_mass = {}
    for person in frontier:
        descendant_side_mass[person] = float(
            sum(
                uncertainty_mass.get(descendant, 0.0)
                for descendant in descendants.get(person, ())
                if descendant in untested
            )
        )
    descendant_bridge_mass = float(
        sum(
            uncertainty_mass[p] * bridge_scores.get(p, 1.0) * descendant_side_mass.get(p, 0.0)
            for p in frontier
        )
    )
    sibling_breadth = _sibling_uncertainty_breadth(pedigree, untested, uncertainty_mass)
    collateral = _active_collateral_block_count(pedigree, individuals, untested, uncertainty_mass)

    q_by_gene = {}
    for gene in gene_list:
        q_by_gene[gene] = float(
            sum(weighted_by_gene.get(person, {}).get(gene, 0.0) for person in frontier)
        )
    reward_pressure = float(sum(q_by_gene.values()))
    if len(q_by_gene) >= 2 and reward_pressure > _EPS:
        allele_asymmetry = (max(q_by_gene.values()) - min(q_by_gene.values())) / (reward_pressure + _EPS)
    else:
        allele_asymmetry = 0.0
    reward_pressure_score = reward_pressure / (reward_pressure + 1.0)

    gates = regime_gates if isinstance(regime_gates, Mapping) else {}
    fixed_cost = float(gates.get("fixed_cost", 0.0) or 0.0)
    variable_cost = float(gates.get("variable_cost", 0.0) or 0.0)
    avg_test_cost = (
        sum(fixed_cost + variable_cost * carrier_plus.get(person, 0.0) for person in frontier)
        / float(len(frontier))
        if frontier
        else 0.0
    )
    cost_pressure_score = avg_test_cost / (avg_test_cost + reward_pressure + _EPS)
    bridge_ratio = bridge_pressure / (frontier_uncertainty + _EPS)
    breadth_ratio = sibling_breadth / (sibling_breadth + bridge_pressure + _EPS)

    features.update(
        {
            "frontier_carrier_mass": carrier_total,
            "frontier_carrier_max": carrier_max,
            "frontier_carrier_variance": carrier_var,
            "bridge_depth_mass": bridge_depth_mass,
            "descendant_bridge_mass": descendant_bridge_mass,
            "sibling_breadth": sibling_breadth,
            "collateral_block_count": collateral,
            "allele_asymmetry": float(allele_asymmetry),
            "bridge_pressure_score": bridge_ratio,
            "breadth_pressure_score": breadth_ratio,
            "reward_pressure_score": reward_pressure_score,
            "test_cost_pressure_score": cost_pressure_score,
            "active_frontier_count": float(len(frontier)),
            "active_frontier_uncertainty_mass": frontier_uncertainty,
        }
    )
    features["bridge_dominated_region"] = 1.0 if bridge_ratio + _EPS >= 0.50 else 0.0
    features["breadth_dominated_region"] = 1.0 if breadth_ratio + _EPS >= 0.50 else 0.0
    features["collateral_active_region"] = 1.0 if collateral >= 1.0 else 0.0
    features["allele_asymmetry_high_region"] = 1.0 if allele_asymmetry >= 0.05 else 0.0
    features["allele_asymmetry_low_region"] = 1.0 if allele_asymmetry <= 0.05 else 0.0
    features["reward_pressure_high_region"] = 1.0 if reward_pressure_score >= 0.45 else 0.0
    features["test_cost_pressure_high_region"] = 1.0 if cost_pressure_score >= 0.035 else 0.0

    features["bridge_dominated_bridge_depth_mass"] = (
        features["bridge_dominated_region"] * features["bridge_depth_mass"]
    )
    features["breadth_dominated_sibling_breadth"] = (
        features["breadth_dominated_region"] * features["sibling_breadth"]
    )
    features["collateral_active_collateral_block_count"] = (
        features["collateral_active_region"] * features["collateral_block_count"]
    )
    features["allele_asymmetry_high_bridge_depth_mass"] = (
        features["allele_asymmetry_high_region"] * features["bridge_depth_mass"]
    )
    features["allele_asymmetry_low_frontier_carrier_variance"] = (
        features["allele_asymmetry_low_region"] * features["frontier_carrier_variance"]
    )
    features["reward_pressure_high_bridge_depth_mass"] = (
        features["reward_pressure_high_region"] * features["bridge_depth_mass"]
    )
    features["reward_pressure_high_collateral_block_count"] = (
        features["reward_pressure_high_region"] * features["collateral_block_count"]
    )
    features["test_cost_pressure_high_frontier_carrier_variance"] = (
        features["test_cost_pressure_high_region"] * features["frontier_carrier_variance"]
    )

    if genes:
        for gene in tuple(genes):
            prefix = f"gene_{gene}"
            gene_frontier = [
                person
                for person in frontier
                if weighted_by_gene.get(person, {}).get(gene, 0.0) > _EPS
            ]
            gene_carriers = [carrier_by_gene.get(person, {}).get(gene, 0.0) for person in gene_frontier]
            gene_var = _variance(gene_carriers)
            gene_descendant_side_mass = {}
            for person in gene_frontier:
                gene_descendant_side_mass[person] = float(
                    sum(
                        weighted_by_gene.get(descendant, {}).get(gene, 0.0)
                        for descendant in descendants.get(person, ())
                        if descendant in untested
                    )
                )
            gene_depth = float(
                sum(
                    weighted_by_gene[person][gene]
                    * bridge_scores.get(person, 1.0)
                    * depth_score.get(person, 1.0)
                    for person in gene_frontier
                )
            )
            gene_bridge = float(
                sum(
                    weighted_by_gene[person][gene]
                    * bridge_scores.get(person, 1.0)
                    * gene_descendant_side_mass.get(person, 0.0)
                    for person in gene_frontier
                )
            )
            features[f"{prefix}_bridge_depth_mass"] = gene_depth
            features[f"{prefix}_descendant_bridge_mass"] = gene_bridge
            features[f"{prefix}_frontier_carrier_variance"] = gene_var
            features[f"allele_asymmetry_high_{prefix}_bridge_depth_mass"] = (
                features["allele_asymmetry_high_region"] * gene_depth
            )
            features[f"reward_pressure_high_{prefix}_descendant_bridge_mass"] = (
                features["reward_pressure_high_region"] * gene_bridge
            )
            features[f"test_cost_pressure_high_{prefix}_frontier_carrier_variance"] = (
                features["test_cost_pressure_high_region"] * gene_var
            )
    return features


def _build_state_features_fb1r_calib(
    state,
    *,
    belief,
    individuals,
    pedigree=None,
    genes=None,
    myopic_policy=None,
    myopic_values=None,
    myopic_residuals=None,
    regime_gates=None,
):
    """Calibrated strict bank plus honestly named proxy-strength signals."""

    proxy = _build_state_features_proxy_v0(
        state,
        belief=belief,
        individuals=individuals,
        pedigree=pedigree,
        genes=genes,
        myopic_policy=myopic_policy,
        myopic_values=myopic_values,
        myopic_residuals=myopic_residuals,
        regime_gates=regime_gates,
    )
    strict = _build_state_features_spec_v1(
        state,
        belief=belief,
        individuals=individuals,
        pedigree=pedigree,
        genes=genes,
        myopic_policy=myopic_policy,
        myopic_values=myopic_values,
        myopic_residuals=myopic_residuals,
        regime_gates=regime_gates,
    )
    features = dict(strict)
    bridge_score = float(strict.get("bridge_pressure_score", 0.0) or 0.0)
    breadth_score = float(strict.get("breadth_pressure_score", 0.0) or 0.0)
    reward_score = float(strict.get("reward_pressure_score", 0.0) or 0.0)
    cost_score = float(strict.get("test_cost_pressure_score", 0.0) or 0.0)
    features.update(
        {
            "strict_bridge_depth_uncertainty": float(strict.get("bridge_depth_mass", 0.0) or 0.0),
            "strict_descendant_bridge_uncertainty": float(strict.get("descendant_bridge_mass", 0.0) or 0.0),
            "strict_sibling_breadth_uncertainty": float(strict.get("sibling_breadth", 0.0) or 0.0),
            "strict_collateral_block_count": float(strict.get("collateral_block_count", 0.0) or 0.0),
            "strict_frontier_carrier_variance": float(strict.get("frontier_carrier_variance", 0.0) or 0.0),
            "soft_bridge_dominated_bridge_depth": bridge_score * float(strict.get("bridge_depth_mass", 0.0) or 0.0),
            "soft_breadth_dominated_sibling_breadth": breadth_score * float(strict.get("sibling_breadth", 0.0) or 0.0),
            "soft_reward_pressure_bridge_depth": reward_score * float(strict.get("bridge_depth_mass", 0.0) or 0.0),
            "soft_reward_pressure_collateral_block_count": reward_score
            * float(strict.get("collateral_block_count", 0.0) or 0.0),
            "soft_test_cost_pressure_frontier_carrier_variance": cost_score
            * float(strict.get("frontier_carrier_variance", 0.0) or 0.0),
            "carrier_depth_mass_honest": float(proxy.get("bridge_depth_mass", 0.0) or 0.0),
            "child_count_weighted_carrier_mass_honest": float(proxy.get("descendant_bridge_mass", 0.0) or 0.0),
            "parent_pair_sibling_count_honest": float(proxy.get("sibling_breadth", 0.0) or 0.0),
            "parent_pair_collateral_proxy_count_honest": float(proxy.get("collateral_block_count", 0.0) or 0.0),
            "all_untested_carrier_variance_honest": float(proxy.get("frontier_carrier_variance", 0.0) or 0.0),
            "parent_pair_breadth_active_sibling_count_honest": float(
                proxy.get("breadth_dominated_sibling_breadth", 0.0) or 0.0
            ),
            "child_count_bridge_dominated_carrier_depth_honest": float(
                proxy.get("bridge_dominated_bridge_depth_mass", 0.0) or 0.0
            ),
            "collateral_active_parent_pair_block_count_honest": float(
                proxy.get("collateral_active_collateral_block_count", 0.0) or 0.0
            ),
            "reward_pressure_carrier_depth_mass_honest": float(
                proxy.get("reward_pressure_high_bridge_depth_mass", 0.0) or 0.0
            ),
            "reward_pressure_parent_pair_collateral_count_honest": float(
                proxy.get("reward_pressure_high_collateral_block_count", 0.0) or 0.0
            ),
            "test_cost_pressure_all_untested_carrier_variance_honest": float(
                proxy.get("test_cost_pressure_high_frontier_carrier_variance", 0.0) or 0.0
            ),
            "fb1r_frontier_mode_information_relevance": 1.0,
            "fb1r_mass_mode_uncertainty_plus_honest_carrier": 1.0,
            "fb1r_gate_mode_ungated_plus_soft": 1.0,
        }
    )
    if genes:
        for gene in tuple(genes):
            prefix = f"gene_{gene}"
            strict_depth = float(strict.get(f"{prefix}_bridge_depth_mass", 0.0) or 0.0)
            strict_bridge = float(strict.get(f"{prefix}_descendant_bridge_mass", 0.0) or 0.0)
            strict_var = float(strict.get(f"{prefix}_frontier_carrier_variance", 0.0) or 0.0)
            proxy_depth = float(proxy.get(f"{prefix}_bridge_depth_mass", 0.0) or 0.0)
            proxy_bridge = float(proxy.get(f"{prefix}_descendant_bridge_mass", 0.0) or 0.0)
            proxy_var = float(proxy.get(f"{prefix}_frontier_carrier_variance", 0.0) or 0.0)
            features[f"{prefix}_strict_bridge_depth_uncertainty"] = strict_depth
            features[f"{prefix}_strict_descendant_bridge_uncertainty"] = strict_bridge
            features[f"{prefix}_strict_frontier_carrier_variance"] = strict_var
            features[f"{prefix}_carrier_depth_mass_honest"] = proxy_depth
            features[f"{prefix}_child_count_weighted_carrier_mass_honest"] = proxy_bridge
            features[f"{prefix}_all_untested_carrier_variance_honest"] = proxy_var
            features[f"allele_asymmetry_high_{prefix}_carrier_depth_mass_honest"] = float(
                proxy.get(f"allele_asymmetry_high_{prefix}_bridge_depth_mass", 0.0) or 0.0
            )
            features[f"reward_pressure_{prefix}_child_count_weighted_carrier_mass_honest"] = float(
                proxy.get(f"reward_pressure_high_{prefix}_descendant_bridge_mass", 0.0) or 0.0
            )
            features[f"test_cost_pressure_{prefix}_all_untested_carrier_variance_honest"] = float(
                proxy.get(f"test_cost_pressure_high_{prefix}_frontier_carrier_variance", 0.0) or 0.0
            )
            features[f"soft_reward_pressure_{prefix}_descendant_bridge_uncertainty"] = (
                reward_score * strict_bridge
            )
            features[f"soft_test_cost_pressure_{prefix}_frontier_carrier_variance"] = (
                cost_score * strict_var
            )
    return features


def build_state_features(
    state,
    *,
    belief,
    individuals,
    pedigree=None,
    genes=None,
    myopic_policy=None,
    myopic_values=None,
    myopic_residuals=None,
    regime_gates=None,
    feature_semantics=FEATURE_SEMANTICS_PROXY_V0,
):
    """Small, deterministic feature bank for myopic-ADP and gauged-regime experiments."""

    semantics = resolve_feature_semantics(feature_semantics)
    if semantics == FEATURE_SEMANTICS_PROXY_V0:
        return _build_state_features_proxy_v0(
            state,
            belief=belief,
            individuals=individuals,
            pedigree=pedigree,
            genes=genes,
            myopic_policy=myopic_policy,
            myopic_values=myopic_values,
            myopic_residuals=myopic_residuals,
            regime_gates=regime_gates,
        )
    if semantics == FEATURE_SEMANTICS_SPEC_V1:
        return _build_state_features_spec_v1(
            state,
            belief=belief,
            individuals=individuals,
            pedigree=pedigree,
            genes=genes,
            myopic_policy=myopic_policy,
            myopic_values=myopic_values,
            myopic_residuals=myopic_residuals,
            regime_gates=regime_gates,
        )
    if semantics == FEATURE_SEMANTICS_ABCD_HAND:
        return _build_state_features_fb1r_calib(
            state,
            belief=belief,
            individuals=individuals,
            pedigree=pedigree,
            genes=genes,
            myopic_policy=myopic_policy,
            myopic_values=myopic_values,
            myopic_residuals=myopic_residuals,
            regime_gates=regime_gates,
        )
    return _build_state_features_fb1r_calib(
        state,
        belief=belief,
        individuals=individuals,
        pedigree=pedigree,
        genes=genes,
        myopic_policy=myopic_policy,
        myopic_values=myopic_values,
        myopic_residuals=myopic_residuals,
        regime_gates=regime_gates,
    )


def regime_residual_feature_values(state, star, *, belief, individuals, pedigree=None, genes=None):
    if not isinstance(star, Mapping) or not star.get("enabled"):
        return {}
    feature_semantics = star.get("feature_semantics")
    if feature_semantics is None or str(feature_semantics).strip() == "":
        raise ValueError("regime_residual_star feature_semantics is required for reconstruction.")
    resolved_semantics = resolve_feature_semantics(feature_semantics)
    feature_bank = star.get("feature_bank")
    if feature_bank is not None and str(feature_bank).strip() != "":
        bank_semantics = feature_semantics_for_bank(feature_bank)
        if bank_semantics != resolved_semantics:
            raise ValueError(
                "regime_residual_star feature_bank and feature_semantics disagree: "
                f"{feature_bank!r} implies {bank_semantics!r}, got {resolved_semantics!r}."
            )
    raw = build_state_features(
        state,
        belief=belief,
        individuals=individuals,
        pedigree=pedigree,
        genes=genes,
        regime_gates=star.get("regime_gates"),
        feature_semantics=resolved_semantics,
    )
    root_values = star.get("feature_root_values", {})
    scales = star.get("feature_scales", {})
    selected = star.get("selected_features", ())
    values = {}
    for name in selected:
        scale = float(scales.get(name, 0.0) or 0.0)
        if abs(scale) < 1e-12:
            continue
        values[name] = (float(raw.get(name, 0.0) or 0.0) - float(root_values.get(name, 0.0) or 0.0)) / scale
    return values


def regime_residual_term_value(state, star, *, belief, individuals, pedigree=None, genes=None):
    if not isinstance(star, Mapping) or not star.get("enabled"):
        return 0.0
    coeffs = star.get("coefficients", {})
    if not isinstance(coeffs, Mapping) or not coeffs:
        return 0.0
    values = regime_residual_feature_values(
        state,
        star,
        belief=belief,
        individuals=individuals,
        pedigree=pedigree,
        genes=genes,
    )
    return float(sum(float(coeffs.get(name, 0.0) or 0.0) * float(values.get(name, 0.0) or 0.0) for name in coeffs))


def _vector_norm(vector):
    return math.sqrt(sum(float(value) * float(value) for value in vector))


def _dot(left, right):
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _orthonormalize(vectors, *, tol=1e-10):
    basis = []
    skipped = 0
    for vector in vectors:
        residual = [float(value) for value in vector]
        for unit in basis:
            coeff = _dot(residual, unit)
            residual = [value - coeff * unit_value for value, unit_value in zip(residual, unit)]
        norm = _vector_norm(residual)
        if norm <= tol:
            skipped += 1
            continue
        basis.append([value / norm for value in residual])
    return basis, skipped


def residualize_signature(vector, basis):
    residual = [float(value) for value in vector]
    for unit in basis:
        coeff = _dot(residual, unit)
        residual = [value - coeff * unit_value for value, unit_value in zip(residual, unit)]
    return residual


def select_signature_features(
    candidate_vectors,
    *,
    legacy_vectors=(),
    top_k=5,
    min_ratio=0.20,
    incremental_tol=1e-8,
):
    legacy_basis, legacy_skipped = _orthonormalize(legacy_vectors)
    selected = []
    selected_vectors = []
    diagnostics = {
        "legacy_signature_rank": int(len(legacy_basis)),
        "legacy_signature_zero_or_dependent_count": int(legacy_skipped),
        "candidate_raw_norms": {},
        "candidate_residual_norms": {},
        "candidate_residual_ratios": {},
        "candidate_incremental_norms": {},
        "selection_order": [],
    }
    basis = list(legacy_basis)
    remaining = sorted(candidate_vectors)
    while remaining and len(selected) < int(top_k):
        scored = []
        for name in remaining:
            vector = [float(value) for value in candidate_vectors[name]]
            raw_norm = _vector_norm(vector)
            residual = residualize_signature(vector, basis)
            residual_norm = _vector_norm(residual)
            ratio = residual_norm / (raw_norm + 1e-12)
            diagnostics["candidate_raw_norms"][name] = float(raw_norm)
            diagnostics["candidate_residual_norms"][name] = float(residual_norm)
            diagnostics["candidate_residual_ratios"][name] = float(ratio)
            diagnostics["candidate_incremental_norms"][name] = float(residual_norm)
            if ratio + 1e-15 < float(min_ratio):
                continue
            if residual_norm <= float(incremental_tol):
                continue
            scored.append((ratio, residual_norm, name, residual))
        if not scored:
            break
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        ratio, residual_norm, name, residual = scored[0]
        norm = _vector_norm(residual)
        if norm <= float(incremental_tol):
            break
        unit = [value / norm for value in residual]
        basis.append(unit)
        selected.append(name)
        selected_vectors.append(candidate_vectors[name])
        diagnostics["selection_order"].append(
            {
                "feature": name,
                "residual_ratio": float(ratio),
                "incremental_norm": float(residual_norm),
            }
        )
        remaining = [candidate for candidate in remaining if candidate != name]
    diagnostics["selected_signature_rank"] = int(len(basis) - len(legacy_basis))
    return selected, selected_vectors, diagnostics


def _weighted_dot(left, right, weights):
    return sum(float(w) * float(a) * float(b) for a, b, w in zip(left, right, weights))


def _weighted_norm(vector, weights):
    return math.sqrt(max(0.0, _weighted_dot(vector, vector, weights)))


def _weighted_orthonormalize(vectors, weights, *, tol=1e-10):
    basis = []
    skipped = 0
    for vector in vectors:
        residual = [float(value) for value in vector]
        for unit in basis:
            coeff = _weighted_dot(residual, unit, weights)
            residual = [value - coeff * unit_value for value, unit_value in zip(residual, unit)]
        norm = _weighted_norm(residual, weights)
        if norm <= tol:
            skipped += 1
            continue
        basis.append([value / norm for value in residual])
    return basis, skipped


def residualize_weighted_signature(vector, basis, weights):
    residual = [float(value) for value in vector]
    for unit in basis:
        coeff = _weighted_dot(residual, unit, weights)
        residual = [value - coeff * unit_value for value, unit_value in zip(residual, unit)]
    return residual


def select_weighted_signature_features(
    candidate_vectors,
    *,
    weights,
    legacy_vectors=(),
    selected_seed_vectors=(),
    exclude_features=(),
    top_k=5,
    min_ratio=0.10,
    incremental_tol=1e-8,
):
    clean_weights = [max(0.0, float(value)) for value in weights]
    if not clean_weights or sum(clean_weights) <= 0.0:
        clean_weights = [1.0 for _ in range(len(next(iter(candidate_vectors.values()), [])))]
    legacy_basis, legacy_skipped = _weighted_orthonormalize(legacy_vectors, clean_weights)
    seed_basis, seed_skipped = _weighted_orthonormalize(selected_seed_vectors, clean_weights)
    basis = list(legacy_basis)
    for unit in seed_basis:
        residual = residualize_weighted_signature(unit, basis, clean_weights)
        norm = _weighted_norm(residual, clean_weights)
        if norm > 1e-10:
            basis.append([value / norm for value in residual])
        else:
            seed_skipped += 1
    excluded = set(exclude_features or ())
    selected = []
    selected_vectors = []
    diagnostics = {
        "weighted": True,
        "legacy_signature_rank": int(len(legacy_basis)),
        "seed_signature_rank": int(len(basis) - len(legacy_basis)),
        "legacy_signature_zero_or_dependent_count": int(legacy_skipped),
        "seed_signature_zero_or_dependent_count": int(seed_skipped),
        "candidate_raw_norms": {},
        "candidate_residual_norms": {},
        "candidate_residual_ratios": {},
        "candidate_incremental_norms": {},
        "selection_order": [],
        "excluded_features": sorted(excluded),
    }
    remaining = sorted(name for name in candidate_vectors if name not in excluded)
    while remaining and len(selected) < int(top_k):
        scored = []
        for name in remaining:
            vector = [float(value) for value in candidate_vectors[name]]
            raw_norm = _weighted_norm(vector, clean_weights)
            residual = residualize_weighted_signature(vector, basis, clean_weights)
            residual_norm = _weighted_norm(residual, clean_weights)
            ratio = residual_norm / (raw_norm + 1e-12)
            diagnostics["candidate_raw_norms"][name] = float(raw_norm)
            diagnostics["candidate_residual_norms"][name] = float(residual_norm)
            diagnostics["candidate_residual_ratios"][name] = float(ratio)
            diagnostics["candidate_incremental_norms"][name] = float(residual_norm)
            if ratio + 1e-15 < float(min_ratio):
                continue
            if residual_norm <= float(incremental_tol):
                continue
            scored.append((ratio, residual_norm, name, residual))
        if not scored:
            break
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        ratio, residual_norm, name, residual = scored[0]
        norm = _weighted_norm(residual, clean_weights)
        if norm <= float(incremental_tol):
            break
        unit = [value / norm for value in residual]
        basis.append(unit)
        selected.append(name)
        selected_vectors.append(candidate_vectors[name])
        diagnostics["selection_order"].append(
            {
                "feature": name,
                "residual_ratio": float(ratio),
                "incremental_norm": float(residual_norm),
            }
        )
        remaining = [candidate for candidate in remaining if candidate != name]
    diagnostics["selected_signature_rank"] = int(len(basis) - len(legacy_basis))
    diagnostics["selected_new_signature_rank"] = int(len(selected))
    return selected, selected_vectors, diagnostics


def myopic_adp_term_value(state, star, *, belief, individuals, pedigree=None, genes=None):
    if not isinstance(star, Mapping) or not star.get("enabled"):
        return 0.0
    term = 0.0
    key = state_key(state)
    mode = star.get("mode")
    if mode == "control_variate":
        values = star.get("myopic_values", {})
        if isinstance(values, Mapping):
            term += float(values.get(state, values.get(key, 0.0)) or 0.0)
    coeffs = star.get("coefficients", {})
    if isinstance(coeffs, Mapping) and coeffs:
        features = build_state_features(
            state,
            belief=belief,
            individuals=individuals,
            pedigree=pedigree,
            genes=genes,
            myopic_policy=star.get("myopic_policy"),
            myopic_values=star.get("myopic_values"),
            myopic_residuals=star.get("myopic_residuals"),
        )
        for name, coef in coeffs.items():
            term += float(coef) * float(features.get(name, 0.0))
    return float(term)


def serializable_summary(star):
    if not isinstance(star, Mapping):
        return None
    return {
        "enabled": bool(star.get("enabled")),
        "mode": star.get("mode"),
        "feature_bank": star.get("feature_bank"),
        "feature_semantics": star.get("feature_semantics"),
        "feature_names": list(star.get("feature_names", ())),
        "selected_features": list(star.get("selected_features", ())),
        "coefficient_count": len(star.get("coefficients", {}) or {}),
        "myopic_state_count": len(star.get("myopic_policy", {}) or {}),
        "myopic_value_count": len(star.get("myopic_values", {}) or {}),
        "root_myopic_value": star.get("root_myopic_value"),
        "diagnostics": star.get("diagnostics", {}),
    }
