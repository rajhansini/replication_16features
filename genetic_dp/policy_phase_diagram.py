from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, MutableMapping, Sequence


VALUE_EPS_ABS = 1e-5
VALUE_EPS_REL = 1e-3
SIGNIFICANT_GAP_ABS = 1e-3
SIGNIFICANT_GAP_REL = 0.02
TOP_K = 3

BENCHMARK_SUITES = {
    "DEBUG_8",
    "DISCOVERY_GRID",
    "COVERAGE_COMPLETION",
    "PHASE_BENCH_24",
    "PHASE_BENCH_40",
}

EXACT_STATUSES = {
    "ok",
    "skipped_too_large",
    "skipped_run_exact_disabled",
    "failed",
}

PHASE_CLASSES = {
    "STOP_DOMINANT",
    "MYOPIC_NEAR_OPTIMAL",
    "NONMYOPIC_PRESENT_MYOPIC_WRONG",
    "NONMYOPIC_PRESENT_MYOPIC_SAME_ACTION",
    "ADP_PROPOSES_OPTIMAL",
    "ADP_POLICY_IMPROVES",
    "ADP_CERTIFICATE_ONLY",
    "HARD_NONMYOPIC_UNSOLVED",
    "UNCLASSIFIED",
}

BENCHMARK_CLEAN_ENV = {
    "EXACT_DP_SOLVER": "dual",
    "ENABLE_TUPLE_ROWGEN": "1",
    "ENABLE_PER_GENE_PHI": "1",
    "EXHAUSTIVE_BELLMAN": "1",
    "EXHAUSTIVE_STRICT": "1",
    "MAX_STATES_PER_ITER": "110000",
    "MAX_CUTS_PER_ITER": "1500000",
    "EXHAUSTIVE_WALLTIME_LIMIT_SEC": "7200",
    "EXHAUSTIVE_NO_PROGRESS_LIMIT_SEC": "600",
    "EXHAUSTIVE_HEARTBEAT_EVERY_SEC": "30",
    "EXHAUSTIVE_HEARTBEAT_EVERY_ITERS": "1",
    "GUROBI_SEED": "0",
    "PULP_SOLVER": "gurobi",
    "THETA_MODE": "stage",
    "ENABLE_EDGE_FEATURES": "0",
    "DISABLE_TRUNCATED_TUPLE_STRENGTHENING": "1",
}

REQUIRED_ROW_FIELDS = {
    "case_id",
    "benchmark_suite",
    "family_id",
    "topology_tags",
    "parameter_tags",
    "initial_state_id",
    "exact_status",
    "exact_state_count",
    "exact_runtime_sec",
    "V_star_root",
    "V_stop_root",
    "V_myopic_root",
    "V_legacy_policy_root",
    "V_grr2_fb2_policy_root",
    "V_grr2d_fb2_policy_root",
    "V_rollout_grr_candidates_root",
    "a_star_root",
    "a_stop_root",
    "a_myopic_root",
    "a_grr2_root",
    "a_grr2d_root",
    "grr2_topk_actions",
    "grr2d_topk_actions",
    "expected_tests_star",
    "expected_tests_myopic",
    "expected_tests_grr2",
    "expected_tests_rollout",
    "gap_myopic_abs",
    "gap_myopic_rel",
    "gap_stop_abs",
    "gap_stop_rel",
    "gap_nonmyopic_abs",
    "gap_nonmyopic_rel",
    "myopic_first_action_wrong",
    "grr2_first_action_wrong",
    "grr2_topk_covers_optimal",
    "grr2d_topk_covers_optimal",
    "primary_phase_class",
    "phase_tags",
    "certificate_gap1_grr2",
    "certificate_ratio3_grr2",
    "certificate_gap1_grr2d",
    "certificate_ratio3_grr2d",
    "notes",
}


@dataclass(frozen=True)
class ClassificationThresholds:
    value_eps_abs: float = VALUE_EPS_ABS
    value_eps_rel: float = VALUE_EPS_REL
    significant_gap_abs: float = SIGNIFICANT_GAP_ABS
    significant_gap_rel: float = SIGNIFICANT_GAP_REL


def finite_float(value: Any) -> float | None:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(value_f):
        return None
    return value_f


def clean_gap(value: float, *, eps: float = 1e-10) -> float:
    return 0.0 if abs(value) <= eps else float(value)


def action_to_string(action: Any) -> str | None:
    if action is None:
        return None
    if isinstance(action, str):
        token = action.strip()
        if not token:
            return None
        if token.lower() == "stop":
            return "STOP"
        if token.upper().startswith("TEST("):
            return token
        return token
    if isinstance(action, (list, tuple)) and action:
        kind = str(action[0]).strip().lower()
        if kind == "stop":
            return "STOP"
        if kind == "test":
            who = action[1] if len(action) > 1 else None
            return f"TEST({who})"
    return str(action)


def actions_equal(left: Any, right: Any) -> bool:
    return action_to_string(left) == action_to_string(right)


def metric_denominator(row: Mapping[str, Any]) -> float | None:
    values = [
        abs(value)
        for value in (
            finite_float(row.get("V_star_root")),
            finite_float(row.get("V_stop_root")),
            finite_float(row.get("V_myopic_root")),
            1e-8,
        )
        if value is not None
    ]
    if not values:
        return None
    return max(values)


def compute_policy_phase_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    if updated.get("exact_status") != "ok":
        for key in (
            "gap_myopic_abs",
            "gap_myopic_rel",
            "gap_stop_abs",
            "gap_stop_rel",
            "gap_nonmyopic_abs",
            "gap_nonmyopic_rel",
        ):
            updated.setdefault(key, None)
        updated.setdefault("myopic_first_action_wrong", None)
        updated.setdefault("grr2_first_action_wrong", None)
        updated.setdefault("grr2_topk_covers_optimal", None)
        updated.setdefault("grr2d_topk_covers_optimal", None)
        return updated

    v_star = finite_float(updated.get("V_star_root"))
    v_stop = finite_float(updated.get("V_stop_root"))
    v_myopic = finite_float(updated.get("V_myopic_root"))
    denom = metric_denominator(updated)
    if v_star is None or v_stop is None or v_myopic is None or denom is None:
        return updated

    gap_myopic_abs = clean_gap(v_star - v_myopic)
    gap_stop_abs = clean_gap(v_star - v_stop)
    gap_nonmyopic_abs = clean_gap(v_star - max(v_myopic, v_stop))
    updated["gap_myopic_abs"] = gap_myopic_abs
    updated["gap_stop_abs"] = gap_stop_abs
    updated["gap_nonmyopic_abs"] = gap_nonmyopic_abs
    updated["gap_myopic_rel"] = clean_gap(gap_myopic_abs / denom)
    updated["gap_stop_rel"] = clean_gap(gap_stop_abs / denom)
    updated["gap_nonmyopic_rel"] = clean_gap(gap_nonmyopic_abs / denom)
    updated["myopic_first_action_wrong"] = not actions_equal(
        updated.get("a_myopic_root"),
        updated.get("a_star_root"),
    )
    if updated.get("a_grr2_root") is not None:
        updated["grr2_first_action_wrong"] = not actions_equal(
            updated.get("a_grr2_root"),
            updated.get("a_star_root"),
        )
    else:
        updated.setdefault("grr2_first_action_wrong", None)
    updated["grr2_topk_covers_optimal"] = action_in_topk(
        updated.get("a_star_root"),
        updated.get("grr2_topk_actions"),
    )
    updated["grr2d_topk_covers_optimal"] = action_in_topk(
        updated.get("a_star_root"),
        updated.get("grr2d_topk_actions"),
    )
    return updated


def action_in_topk(action: Any, topk: Any) -> bool | None:
    if topk is None:
        return None
    if not isinstance(topk, Sequence) or isinstance(topk, (str, bytes)):
        return None
    action_s = action_to_string(action)
    return action_s in {action_to_string(item) for item in topk}


def candidate_policy_improves(row: Mapping[str, Any], thresholds: ClassificationThresholds) -> bool:
    v_stop = finite_float(row.get("V_stop_root"))
    v_myopic = finite_float(row.get("V_myopic_root"))
    if v_stop is None or v_myopic is None:
        return False
    baseline = max(v_stop, v_myopic)
    for key in (
        "V_grr2_fb2_policy_root",
        "V_grr2d_fb2_policy_root",
        "V_legacy_policy_root",
    ):
        value = finite_float(row.get(key))
        if value is not None and value > baseline + thresholds.value_eps_abs:
            return True
    rollout_value = finite_float(row.get("V_rollout_grr_candidates_root"))
    if (
        rollout_value is not None
        and row.get("rollout_status") == "ok"
        and row.get("rollout_incumbent_action_present") is True
        and rollout_value > baseline + thresholds.value_eps_abs
    ):
        return True
    return False


def certificate_only(row: Mapping[str, Any], thresholds: ClassificationThresholds) -> bool:
    if candidate_policy_improves(row, thresholds):
        return False
    if row.get("certificate_improves") is True:
        return True
    for key in (
        "certificate_delta_gap1_grr2",
        "certificate_delta_ratio3_grr2",
        "certificate_delta_gap1_grr2d",
        "certificate_delta_ratio3_grr2d",
    ):
        value = finite_float(row.get(key))
        if value is not None and value < -thresholds.value_eps_abs:
            return True
    return False


def classify_policy_phase(
    row: Mapping[str, Any],
    thresholds: ClassificationThresholds | None = None,
) -> tuple[str, list[str]]:
    thresholds = thresholds or ClassificationThresholds()
    row = compute_policy_phase_metrics(row)
    if row.get("exact_status") != "ok":
        return "UNCLASSIFIED", ["UNCLASSIFIED"]

    gap_stop_abs = finite_float(row.get("gap_stop_abs"))
    gap_stop_rel = finite_float(row.get("gap_stop_rel"))
    gap_myopic_abs = finite_float(row.get("gap_myopic_abs"))
    gap_myopic_rel = finite_float(row.get("gap_myopic_rel"))
    gap_nonmyopic_rel = finite_float(row.get("gap_nonmyopic_rel"))
    myopic_wrong = row.get("myopic_first_action_wrong") is True
    grr2_covers = row.get("grr2_topk_covers_optimal") is True
    grr2d_covers = row.get("grr2d_topk_covers_optimal") is True

    tags: list[str] = []

    if (
        (gap_stop_abs is not None and gap_stop_abs <= thresholds.value_eps_abs)
        or (gap_stop_rel is not None and gap_stop_rel <= thresholds.value_eps_rel)
    ):
        tags.append("STOP_DOMINANT")

    if (
        (gap_myopic_abs is not None and gap_myopic_abs <= thresholds.value_eps_abs)
        or (gap_myopic_rel is not None and gap_myopic_rel <= thresholds.value_eps_rel)
    ):
        tags.append("MYOPIC_NEAR_OPTIMAL")

    significant_myopic = (
        gap_myopic_abs is not None
        and gap_myopic_rel is not None
        and gap_myopic_abs >= thresholds.significant_gap_abs
        and gap_myopic_rel >= thresholds.significant_gap_rel
    )
    if significant_myopic and myopic_wrong:
        tags.append("NONMYOPIC_PRESENT_MYOPIC_WRONG")
    elif significant_myopic and row.get("myopic_first_action_wrong") is False:
        tags.append("NONMYOPIC_PRESENT_MYOPIC_SAME_ACTION")

    if myopic_wrong and (grr2_covers or grr2d_covers):
        tags.append("ADP_PROPOSES_OPTIMAL")

    if candidate_policy_improves(row, thresholds):
        tags.append("ADP_POLICY_IMPROVES")

    if certificate_only(row, thresholds):
        tags.append("ADP_CERTIFICATE_ONLY")

    topk_recorded = row.get("grr2_topk_actions") is not None or row.get("grr2d_topk_actions") is not None
    topk_covers = grr2_covers or grr2d_covers
    if (
        gap_nonmyopic_rel is not None
        and gap_nonmyopic_rel >= thresholds.significant_gap_rel
        and topk_recorded
        and not topk_covers
    ):
        tags.append("HARD_NONMYOPIC_UNSOLVED")

    if not tags:
        tags.append("UNCLASSIFIED")

    primary_priority = [
        "STOP_DOMINANT",
        "MYOPIC_NEAR_OPTIMAL",
        "ADP_POLICY_IMPROVES",
        "ADP_PROPOSES_OPTIMAL",
        "HARD_NONMYOPIC_UNSOLVED",
        "NONMYOPIC_PRESENT_MYOPIC_WRONG",
        "NONMYOPIC_PRESENT_MYOPIC_SAME_ACTION",
        "ADP_CERTIFICATE_ONLY",
        "UNCLASSIFIED",
    ]
    primary = next(item for item in primary_priority if item in tags)
    return primary, tags


def apply_classification(row: Mapping[str, Any]) -> dict[str, Any]:
    updated = compute_policy_phase_metrics(row)
    primary, tags = classify_policy_phase(updated)
    updated["primary_phase_class"] = primary
    updated["phase_tags"] = tags
    return updated


def _entry_payload(entry: Any) -> Any:
    return entry[0] if isinstance(entry, tuple) else entry


def _state_outcomes(
    *,
    state: frozenset,
    person: str,
    belief: Mapping[frozenset, Any],
    gen_states: Sequence[Any],
) -> list[tuple[Any, float]]:
    entry = belief[state]
    payload = _entry_payload(entry)
    if hasattr(payload, "has_tuple_pmfs") and payload.has_tuple_pmfs():
        return [
            (outcome, float(prob))
            for outcome, prob in payload.get_tuple_pmfs().get(person, {}).items()
            if float(prob) > 0.0
        ]
    marginals = payload.marginals if hasattr(payload, "marginals") else payload
    return [
        (outcome, float(marginals.get(person, {}).get(outcome, 0.0)))
        for outcome in gen_states
        if float(marginals.get(person, {}).get(outcome, 0.0)) > 0.0
    ]


def expected_tests_under_policy(
    policy: Mapping[frozenset, Any],
    *,
    belief: Mapping[frozenset, Any],
    individuals: Sequence[str],
    gen_states: Sequence[Any],
    root_state: frozenset = frozenset(),
) -> float:
    memo: dict[frozenset, float] = {}

    def rec(state: frozenset) -> float:
        if state in memo:
            return memo[state]
        if len(state) >= len(individuals):
            memo[state] = 0.0
            return 0.0
        action = policy.get(state)
        if action is None:
            memo[state] = 0.0
            return 0.0
        action_s = action_to_string(action)
        if action_s == "STOP" or not action_s or not action_s.startswith("TEST("):
            memo[state] = 0.0
            return 0.0
        person = action[1] if isinstance(action, (tuple, list)) and len(action) > 1 else action_s[5:-1]
        total = 1.0
        expected = 0.0
        for outcome, prob in _state_outcomes(
            state=state,
            person=str(person),
            belief=belief,
            gen_states=gen_states,
        ):
            evidence = dict(state)
            evidence[str(person)] = outcome
            succ = frozenset(evidence.items())
            if len(succ) >= len(individuals):
                continue
            expected += prob * rec(succ)
        memo[state] = total + expected
        return memo[state]

    return float(rec(root_state))


def _score_records_are_ranked(records: Any) -> bool:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return False
    previous_rank = 0
    previous_score: float | None = None
    for item in records:
        if not isinstance(item, Mapping):
            return False
        rank = item.get("rank")
        score = finite_float(item.get("score"))
        if not isinstance(rank, int) or rank <= previous_rank or score is None:
            return False
        if previous_score is not None and score > previous_score + 1e-10:
            return False
        previous_rank = rank
        previous_score = score
    return bool(records)


def validate_policy_phase_row(
    row: Mapping[str, Any],
    *,
    tolerance: float = 1e-7,
    require_grr: bool = False,
) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_ROW_FIELDS - set(row))
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")

    suite = row.get("benchmark_suite")
    if suite is not None and suite not in BENCHMARK_SUITES:
        errors.append(f"invalid benchmark_suite={suite!r}")

    status = row.get("exact_status")
    if status is not None and status not in EXACT_STATUSES:
        errors.append(f"invalid exact_status={status!r}")
    if status != "ok" and not row.get("skip_reason") and not row.get("error"):
        errors.append("skipped/failed rows must include skip_reason or error")

    primary = row.get("primary_phase_class")
    if primary is not None and primary not in PHASE_CLASSES:
        errors.append(f"invalid primary_phase_class={primary!r}")

    if status == "ok":
        recomputed = apply_classification(row)
        for key in (
            "gap_myopic_abs",
            "gap_myopic_rel",
            "gap_stop_abs",
            "gap_stop_rel",
            "gap_nonmyopic_abs",
            "gap_nonmyopic_rel",
        ):
            expected = finite_float(recomputed.get(key))
            actual = finite_float(row.get(key))
            if expected is None or actual is None:
                errors.append(f"{key} missing for exact_status=ok")
            elif abs(expected - actual) > tolerance:
                errors.append(f"{key}={actual!r} does not match recomputed {expected!r}")
            if actual is not None and actual < -tolerance:
                errors.append(f"{key} is negative: {actual}")
        if row.get("primary_phase_class") != recomputed.get("primary_phase_class"):
            errors.append(
                "primary_phase_class does not match rules: "
                f"{row.get('primary_phase_class')!r} vs {recomputed.get('primary_phase_class')!r}"
            )
        expected_tags = list(recomputed.get("phase_tags") or [])
        actual_tags = list(row.get("phase_tags") or [])
        if actual_tags != expected_tags:
            errors.append(f"phase_tags do not match rules: {actual_tags!r} vs {expected_tags!r}")

        env = row.get("benchmark_clean_env")
        if isinstance(env, Mapping):
            for key, expected in BENCHMARK_CLEAN_ENV.items():
                actual = env.get(key)
                if actual is not None and str(actual) != str(expected):
                    errors.append(f"benchmark_clean_env[{key}]={actual!r} != {expected!r}")

    grr_present = (
        require_grr
        or row.get("grr_status") is not None
        or row.get("grr2_topk_actions") is not None
        or row.get("grr2d_topk_actions") is not None
    )
    if status == "ok" and grr_present:
        for key in ("grr_status", "grr2_status", "grr2d_status", "rollout_status"):
            if row.get(key) != "ok":
                errors.append(f"{key} must be 'ok' for GRR-complete rows")
        for key in (
            "a_grr2_root",
            "a_grr2d_root",
            "V_grr2_fb2_policy_root",
            "V_grr2d_fb2_policy_root",
            "V_rollout_grr_candidates_root",
            "expected_tests_grr2",
            "expected_tests_rollout",
            "certificate_gap1_grr2",
            "certificate_gap1_grr2d",
        ):
            if row.get(key) is None:
                errors.append(f"{key} missing for GRR-complete row")
        top_k = int(row.get("top_k") or TOP_K)
        for key in ("grr2_topk_actions", "grr2d_topk_actions"):
            actions = row.get(key)
            if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)) or not actions:
                errors.append(f"{key} must be a non-empty action list")
            elif len(actions) > top_k:
                errors.append(f"{key} length {len(actions)} exceeds top_k={top_k}")
        for key in ("grr2_root_action_scores", "grr2d_root_action_scores"):
            if not _score_records_are_ranked(row.get(key)):
                errors.append(f"{key} must be non-empty records ranked by descending score")
        for key in (
            "certificate_gap1_grr2",
            "certificate_ratio3_grr2",
            "certificate_gap1_grr2d",
            "certificate_ratio3_grr2d",
        ):
            value = finite_float(row.get(key))
            if value is not None and value < -tolerance:
                errors.append(f"{key} is negative: {value}")
        if row.get("rollout_incumbent_action_present") is not True:
            errors.append("rollout_incumbent_action_present must be true for GRR-complete rows")
        candidates = row.get("rollout_candidate_actions")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or not candidates:
            errors.append("rollout_candidate_actions must be a non-empty list for GRR-complete rows")

    return errors


def validate_policy_phase_artifact(payload: Mapping[str, Any]) -> list[str]:
    rows = payload.get("cases")
    if not isinstance(rows, list):
        return ["artifact must contain a cases list"]
    errors: list[str] = []
    seen: set[str] = set()
    require_grr = payload.get("run_grr") is True
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"cases[{index}] is not an object")
            continue
        case_id = str(row.get("case_id", f"index:{index}"))
        if case_id in seen:
            errors.append(f"duplicate case_id={case_id!r}")
        seen.add(case_id)
        for error in validate_policy_phase_row(row, require_grr=require_grr):
            errors.append(f"{case_id}: {error}")
    return errors


def summarize_policy_phase_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    row_count = len(rows)
    exact_ok = [row for row in rows if row.get("exact_status") == "ok"]
    phase_counts = Counter(str(row.get("primary_phase_class", "UNCLASSIFIED")) for row in rows)
    grr2_topk_recorded = sum(1 for row in exact_ok if row.get("grr2_topk_actions") is not None)
    grr2d_topk_recorded = sum(1 for row in exact_ok if row.get("grr2d_topk_actions") is not None)
    grr2_topk_covers = sum(1 for row in exact_ok if row.get("grr2_topk_covers_optimal") is True)
    grr2d_topk_covers = sum(1 for row in exact_ok if row.get("grr2d_topk_covers_optimal") is True)
    grr_ok = sum(1 for row in exact_ok if row.get("grr_status") == "ok")
    topology_counts: dict[str, Counter[str]] = defaultdict(Counter)
    parameter_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        phase = str(row.get("primary_phase_class", "UNCLASSIFIED"))
        for tag in row.get("topology_tags") or []:
            topology_counts[str(tag)][phase] += 1
        for tag in row.get("parameter_tags") or []:
            parameter_counts[str(tag)][phase] += 1
    return {
        "row_count": row_count,
        "exact_ok_count": len(exact_ok),
        "grr_ok_count": grr_ok,
        "grr2_topk_recorded_count": grr2_topk_recorded,
        "grr2d_topk_recorded_count": grr2d_topk_recorded,
        "grr2_topk_covers_count": grr2_topk_covers,
        "grr2d_topk_covers_count": grr2d_topk_covers,
        "phase_counts": dict(sorted(phase_counts.items())),
        "topology_phase_counts": {
            tag: dict(sorted(counts.items()))
            for tag, counts in sorted(topology_counts.items())
        },
        "parameter_phase_counts": {
            tag: dict(sorted(counts.items()))
            for tag, counts in sorted(parameter_counts.items())
        },
    }


def _fmt(value: Any, digits: int = 6) -> str:
    value_f = finite_float(value)
    if value_f is None:
        return ""
    return f"{value_f:.{digits}f}"


def _policy_delta(row: Mapping[str, Any], value_key: str) -> float | None:
    stored_key = {
        "V_legacy_policy_root": "delta_policy_value_legacy",
        "V_grr2_fb2_policy_root": "delta_policy_value_grr2",
        "V_grr2d_fb2_policy_root": "delta_policy_value_grr2d",
    }.get(value_key)
    if stored_key:
        stored = finite_float(row.get(stored_key))
        if stored is not None:
            return stored
    value = finite_float(row.get(value_key))
    v_stop = finite_float(row.get("V_stop_root"))
    v_myopic = finite_float(row.get("V_myopic_root"))
    if value is None or v_stop is None or v_myopic is None:
        return None
    return value - max(v_stop, v_myopic)


def _best_candidate_policy_delta(row: Mapping[str, Any]) -> float | None:
    stored = finite_float(row.get("delta_policy_value_best_candidate"))
    if stored is not None:
        return stored
    deltas = [
        _policy_delta(row, key)
        for key in (
            "V_legacy_policy_root",
            "V_grr2_fb2_policy_root",
            "V_grr2d_fb2_policy_root",
            "V_rollout_grr_candidates_root",
        )
    ]
    deltas = [value for value in deltas if value is not None]
    return max(deltas) if deltas else None


def build_policy_phase_markdown(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str,
    source_path: str | None = None,
) -> str:
    summary = summarize_policy_phase_rows(rows)
    lines = [
        f"# {title}",
        "",
    ]
    if source_path:
        lines.append(f"- Source artifact: `{source_path}`")
    lines.extend(
        [
            f"- Rows: `{summary['row_count']}`",
            f"- Exact-complete rows: `{summary['exact_ok_count']}`",
            f"- GRR-complete rows: `{summary.get('grr_ok_count', 0)}`",
            "",
            "## Overall Phase Counts",
            "",
            "| Phase class | Count | Percent |",
            "| --- | ---: | ---: |",
        ]
    )
    total = max(1, int(summary["row_count"]))
    for phase in sorted(PHASE_CLASSES):
        count = int(summary["phase_counts"].get(phase, 0))
        if count:
            lines.append(f"| `{phase}` | {count} | {100.0 * count / total:.1f}% |")

    lines.extend(
        [
            "",
            "## GRR Top-K Coverage",
            "",
            "| Metric | Count | Denominator |",
            "| --- | ---: | ---: |",
            f"| GRR2 top-K recorded | {summary.get('grr2_topk_recorded_count', 0)} | {summary['exact_ok_count']} |",
            f"| GRR2 top-K covers exact optimal root action | {summary.get('grr2_topk_covers_count', 0)} | {summary.get('grr2_topk_recorded_count', 0)} |",
            f"| GRR2D top-K recorded | {summary.get('grr2d_topk_recorded_count', 0)} | {summary['exact_ok_count']} |",
            f"| GRR2D top-K covers exact optimal root action | {summary.get('grr2d_topk_covers_count', 0)} | {summary.get('grr2d_topk_recorded_count', 0)} |",
        ]
    )

    lines.extend(["", "## Phase Counts By Topology", "", "| Topology tag | STOP | MYOPIC | NONMYOPIC | ADP_TOPK | HARD |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for tag, counts in summary["topology_phase_counts"].items():
        nonmyopic = counts.get("NONMYOPIC_PRESENT_MYOPIC_WRONG", 0) + counts.get(
            "NONMYOPIC_PRESENT_MYOPIC_SAME_ACTION",
            0,
        )
        lines.append(
            f"| {tag} | {counts.get('STOP_DOMINANT', 0)} | {counts.get('MYOPIC_NEAR_OPTIMAL', 0)} | "
            f"{nonmyopic} | {counts.get('ADP_PROPOSES_OPTIMAL', 0)} | {counts.get('HARD_NONMYOPIC_UNSOLVED', 0)} |"
        )

    lines.extend(["", "## Phase Counts By Parameter", "", "| Parameter tag | STOP | MYOPIC | NONMYOPIC | ADP_TOPK | HARD |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for tag, counts in summary["parameter_phase_counts"].items():
        nonmyopic = counts.get("NONMYOPIC_PRESENT_MYOPIC_WRONG", 0) + counts.get(
            "NONMYOPIC_PRESENT_MYOPIC_SAME_ACTION",
            0,
        )
        lines.append(
            f"| {tag} | {counts.get('STOP_DOMINANT', 0)} | {counts.get('MYOPIC_NEAR_OPTIMAL', 0)} | "
            f"{nonmyopic} | {counts.get('ADP_PROPOSES_OPTIMAL', 0)} | {counts.get('HARD_NONMYOPIC_UNSOLVED', 0)} |"
        )

    exact_rows = [row for row in rows if row.get("exact_status") == "ok"]
    ranked = sorted(
        exact_rows,
        key=lambda row: finite_float(row.get("gap_nonmyopic_rel")) or 0.0,
        reverse=True,
    )[:10]
    lines.extend(
        [
            "",
            "## Top Non-Myopic Opportunities",
            "",
            "| Case | Family | Params | V* | J_myopic | Gap rel | a* | a_myopic | GRR top-K covers? |",
            "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in ranked:
        lines.append(
            "| {case} | {family} | {params} | {vstar} | {vmyopic} | {gaprel} | {astar} | {amyopic} | {covers} |".format(
                case=row.get("case_id", ""),
                family=row.get("family_id", ""),
                params=", ".join(str(tag) for tag in (row.get("parameter_tags") or [])),
                vstar=_fmt(row.get("V_star_root")),
                vmyopic=_fmt(row.get("V_myopic_root")),
                gaprel=_fmt(row.get("gap_nonmyopic_rel")),
                astar=row.get("a_star_root") or "",
                amyopic=row.get("a_myopic_root") or "",
                covers=row.get("grr2_topk_covers_optimal"),
            )
        )

    lines.extend(
        [
            "",
            "## Certificate-Policy Correlation",
            "",
            "| Case | Delta ratio3 GRR2 | Delta policy GRR2 | Delta policy best candidate | Class |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in exact_rows[:20]:
        lines.append(
            "| {case} | {dr3} | {dpv} | {best} | {phase} |".format(
                case=row.get("case_id", ""),
                dr3=_fmt(row.get("certificate_delta_ratio3_grr2")),
                dpv=_fmt(_policy_delta(row, "V_grr2_fb2_policy_root")),
                best=_fmt(_best_candidate_policy_delta(row)),
                phase=row.get("primary_phase_class", ""),
            )
        )

    return "\n".join(lines) + "\n"


def select_phase_benchmark_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_size: int = 24,
) -> list[dict[str, Any]]:
    targets = {
        "STOP_DOMINANT": 3,
        "MYOPIC_NEAR_OPTIMAL": 3,
        "NONMYOPIC_PRESENT_MYOPIC_WRONG": 4,
        "NONMYOPIC_PRESENT_MYOPIC_SAME_ACTION": 3,
        "ADP_PROPOSES_OPTIMAL": 4,
        "ADP_POLICY_IMPROVES": 4,
        "HARD_NONMYOPIC_UNSOLVED": 2,
        "UNCLASSIFIED": 1,
    }
    selected: list[dict[str, Any]] = []
    used: set[str] = set()

    def sort_key(row: Mapping[str, Any]) -> tuple[float, str]:
        return (-(finite_float(row.get("gap_nonmyopic_rel")) or 0.0), str(row.get("case_id", "")))

    for phase, count in targets.items():
        candidates = [
            row
            for row in rows
            if row.get("exact_status") == "ok"
            and row.get("primary_phase_class") == phase
            and row.get("case_id") not in used
        ]
        for row in sorted(candidates, key=sort_key)[:count]:
            row_copy = dict(row)
            selected.append(row_copy)
            used.add(str(row.get("case_id")))

    if len(selected) < target_size:
        remaining = [
            row
            for row in rows
            if row.get("exact_status") == "ok" and str(row.get("case_id")) not in used
        ]
        for row in sorted(remaining, key=sort_key):
            if len(selected) >= target_size:
                break
            selected.append(dict(row))
            used.add(str(row.get("case_id")))

    return selected[:target_size]
