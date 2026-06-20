def compute_phi(S, p_vals, z_vals, W, gen_states):
    """
    Compute Φ(s) given
      • S  = set of already‑tested individuals
      • p_vals, z_vals = Gurobi tupledicts,  key (i,g)
      • W[i][g]  = weight for individual i, genotype g
    """
    phi = 0.0
    for i in W:
        if i in S:                        # tested ⇒ use z
            for g in gen_states:
                phi += z_vals[i, g] * W[i][g]
        else:                             # untested ⇒ use p
            for g in gen_states:
                phi += p_vals[i, g] * W[i][g]
    return phi


def _project_outcome(value):
    """Project possibly tuple-valued outcomes onto the first component."""
    if isinstance(value, tuple):
        if not value:
            return 0
        return value[0]
    return value


# --- Probability-only symmetry (for posterior caching) ---
def canonicalize_state_prob(state, role_groups=None, gen_states=(0,1,2)):
    """
    Canonical key for *probability* caching: aggregate exchangeable roles by
    genotype counts; keep explicit (person,g) for non-role individuals.
    """
    if not role_groups:
        return ("RAW", tuple(sorted(state)))
    sd = dict(state)
    role_counts, covered = [], set()
    for role, members in role_groups.items():
        tested = [_project_outcome(sd[i]) for i in members if i in sd]
        if tested:
            counts = tuple(tested.count(g) for g in gen_states)
            role_counts.append((role, counts))
            covered.update(members)
    role_counts.sort(key=lambda x: x[0])
    explicit = tuple(sorted((i, sd[i]) for i in sd if i not in covered))
    return ("ROLECOUNTS_ONLY", tuple(role_counts), explicit)

def build_canonical_evidence(state, role_groups, gen_states=(0,1,2)):
    """
    Construct a deterministic evidence dict consistent with the raw state.
    We simply carry forward explicit observed (person,g) entries; for untested
    members no assignments are made (BN handles marginals).
    """
    sd = dict(state)
    canon = {}
    # Keep explicit observations for anyone already tested
    for i, g in sd.items():
        canon[i] = g
    return canon

def remap_posteriors_from_canonical(p_s_canon, state, role_groups, gen_states=(0,1,2)):
    """
    Rebuild per-person marginals for the concrete state:
      - tested people: one-hot at observed genotype
      - untested role members: share the same marginal (taken from any untested member)
      - non-role people: use canonical marginals as-is
    """
    sd = dict(state)
    p_s = {}
    # Non-role set
    nonrole = set(sd.keys())
    for grp in role_groups.values():
        nonrole -= set(grp)

    # Non-role / others
    for i in p_s_canon.keys():
        if any(i in grp for grp in role_groups.values()):
            continue  # handled below as role member
        if i in sd:
            g = _project_outcome(sd[i])
            p_s[i] = {gg: (1.0 if gg == g else 0.0) for gg in gen_states}
        else:
            p_s[i] = dict(p_s_canon[i])

    # Role members
    for role, members in role_groups.items():
        members = sorted(members)
        # representative untested marginal (if any untested exists)
        rep = next((j for j in members if j not in sd and j in p_s_canon), None)
        rep_marg = dict(p_s_canon[rep]) if rep is not None else None
        for i in members:
            if i in sd:
                g = _project_outcome(sd[i])
                p_s[i] = {gg: (1.0 if gg == g else 0.0) for gg in gen_states}
            else:
                # fall back to its own canonical marginal if no representative
                p_s[i] = dict(rep_marg) if rep_marg is not None else dict(p_s_canon[i])
    return p_s

# --- Value-side canonicalization (optional, controlled by value_canon_mode) ---
def canonicalize_state(state, role_groups=None, gen_states=(0,1,2), param_sig_fn=None):
    """
    Canonical key for Φ when merging is opted-in.
    If param_sig_fn is provided, people are grouped by (role, param_sig_fn(i)).
    """
    if not role_groups:
        return ("RAW", tuple(sorted(state)))
    sd = dict(state)
    covered = set()
    role_counts = []
    for role, members in role_groups.items():
        # Cohort-bucket by parameter signature if provided
        buckets = {}
        for i in members:
            sig = param_sig_fn(i) if param_sig_fn else None
            buckets.setdefault(sig, []).append(i)
        for sig, bucket in sorted(buckets.items(), key=lambda kv: str(kv[0])):
            tested = [_project_outcome(sd[i]) for i in bucket if i in sd]
            if tested:
                counts = tuple(tested.count(g) for g in gen_states)
                role_counts.append((role, sig, counts))
                covered.update(bucket)
    role_counts.sort(key=lambda x: (x[0], str(x[1])))
    explicit = tuple(sorted((i, sd[i]) for i in sd if i not in covered))
    tag = "ROLECOUNTS+PARAMS" if param_sig_fn else "ROLECOUNTS"
    return (tag, tuple(role_counts), explicit)

# --- Exchangeability validator (priors, CPDs, rewards) ---
def _get_prior_vec(p0, person, gen_states):
    try:
        return [float(p0[person][g]) for g in gen_states]
    except Exception:
        try:
            return [float(p0[(person, g)]) for g in gen_states]
        except Exception:
            return None

def _almost_equal_vec(a, b, tol=1e-9):
    if a is None or b is None or len(a) != len(b): return False
    return all(abs(x-y) <= tol for x,y in zip(a,b))

def _extract_parents(pedigree, person):
    if pedigree is None: return None
    if isinstance(pedigree, dict) and person in pedigree:
        val = pedigree[person]
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return tuple(val)
    if isinstance(pedigree, dict) and 'parents' in pedigree and isinstance(pedigree['parents'], dict):
        val = pedigree['parents'].get(person)
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return tuple(val)
    return None

def _cpd_signature(child_cpds, person):
    if not child_cpds or person not in child_cpds:
        return None
    cpd = child_cpds[person]
    try:
        items = []
        for k, v in cpd.items():
            if isinstance(v, dict):
                row = tuple((gg, float(v[gg])) for gg in sorted(v.keys()))
            else:
                row = tuple(float(x) for x in v)
            items.append((k, row))
        return tuple(sorted(items, key=lambda x: str(x[0])))
    except Exception:
        return None

def _reward_vec(i, a, b, c, delta):
    try:
        return (float(a[i]), float(b[i]), float(c[i]), float(delta[i]))
    except Exception:
        return None

def reward_signature_fn(a, b, c, delta, rounding=8):
    """Return a function i -> signature tuple for reward params (rounded)."""
    def _sig(i):
        rv = _reward_vec(i, a,b,c,delta)
        return None if rv is None else tuple(None if x is None else round(float(x), rounding) for x in rv)
    return _sig

def validate_role_groups(I, role_groups, gen_states, p0=None, child_cpds=None, pedigree=None,
                         a=None, b=None, c=None, delta=None, tol=1e-9):
    """
    Validate that members within each role_group are exchangeable for the BN
    (priors, CPDs/parents) and optionally homogeneous in reward params.
    Returns {'ok': [...], 'warnings': [...], 'errors': [...]}.
    """
    report = {'ok': [], 'warnings': [], 'errors': []}
    if not role_groups:
        report['warnings'].append('No role_groups provided; nothing to validate.')
        return report
    Iset = set(I)
    for role, members in role_groups.items():
        members = list(members)
        miss = [m for m in members if m not in Iset]
        if miss:
            report['errors'].append(f"Role '{role}': unknown individuals: {miss}")
            continue
        # Priors
        base = None; pri_warn = False
        for m in members:
            vec = _get_prior_vec(p0, m, gen_states) if p0 is not None else None
            if base is None and vec is not None: base = vec
            elif vec is not None and not _almost_equal_vec(base, vec, tol): pri_warn = True
        if pri_warn:
            report['warnings'].append(f"Role '{role}': differing prior genotype distributions.")
        # Parents
        par_vals = {_extract_parents(pedigree, m) for m in members}
        if len([p for p in par_vals if p is not None]) > 1:
            report['warnings'].append(f"Role '{role}': different parent sets in pedigree.")
        # CPDs
        sig_vals = {_cpd_signature(child_cpds, m) for m in members if child_cpds is not None}
        sig_vals = {s for s in sig_vals if s is not None}
        if len(sig_vals) > 1:
            report['warnings'].append(f"Role '{role}': different CPD tables.")
        # Rewards (optional)
        if all(x is not None for x in (a,b,c,delta)):
            rvals = {_reward_vec(m, a,b,c,delta) for m in members}
            rvals = {r for r in rvals if r is not None}
            if len(rvals) > 1:
                report['warnings'].append(
                    f"Role '{role}': different reward params (a,b,c,delta). "
                    "Safe for probability cache; do NOT merge Φ across these members unless using 'cohort' with identical params."
                )
        if (not pri_warn) and (len({p for p in par_vals if p is not None}) <= 1) and (len(sig_vals) <= 1):
            report['ok'].append(role)
    return report

def discover_role_groups(I, gen_states, p0=None, child_cpds=None, pedigree=None,
                         mode="probability", a=None, b=None, c=None, delta=None,
                         min_group=2, rounding=9):
    """
    Auto-build role_groups via hashing:
      probability: (parents, priors, CPD)
      cohort:      (parents, priors, CPD, reward signature)
    Returns dict[str, set] with groups of size >= min_group.
    """
    keys = {}
    for i in I:
        k_par = _extract_parents(pedigree, i)
        # normalize parent order
        if isinstance(k_par, (list, tuple)): k_par = tuple(sorted(k_par))
        k_pri = _get_prior_vec(p0, i, gen_states) if p0 is not None else None
        if k_pri is not None: k_pri = tuple(round(float(x), rounding) for x in k_pri)
        k_cpd = _cpd_signature(child_cpds, i)
        if mode == "cohort":
            rv = _reward_vec(i, a,b,c,delta)
            k_rwd = None if rv is None else tuple(round(float(x), rounding) for x in rv)
            key = ("K", k_par, k_pri, k_cpd, k_rwd)
        else:
            key = ("K", k_par, k_pri, k_cpd)
        keys.setdefault(key, set()).add(i)
    groups = [members for members in keys.values() if len(members) >= min_group]
    role_groups = {f"ROLE_{idx}": set(sorted(members)) for idx, members in enumerate(sorted(groups, key=lambda s: tuple(sorted(s))))}
    return role_groups
