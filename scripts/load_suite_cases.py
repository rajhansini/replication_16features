#!/usr/bin/env python3
"""Benchmark-family loader for the ABCD-16 replication.

Builds the `Setting`/`SuiteCase` objects (family topology + reward preset +
allele frequencies + costs) for the three suites — `original8`, `local40`, and
`phase6_54` (102 families total) — from the bundled settings manifests under
`documentation/` and `artifacts/`.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from scripts.run_multigene_ratio45_new_settings import Setting

# Manifest sources, resolved relative to the cwd (the package root). Copied to
# the same relative locations the original loader expects.
ORIGINAL8_SOURCE = Path("documentation/original8_settings_20260427.json")
LOCAL40_SOURCE = Path("documentation/local40_setting_grid_20260515.json")
PHASE6_54_SOURCE = Path(
    "artifacts/extended_lowhigh_certificate_repair/06_extended_phase_diagram/extended_54row_manifest.csv"
)

SUITE_ORDER = ("original8", "local40", "phase6_54")  # full102 = 8 + 40 + 54


@dataclass(frozen=True)
class SuiteCase:
    suite: str
    row_index: int
    row_id: str
    setting: Setting
    source_path: str
    source_note: str


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _json_loads_maybe(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return json.loads(stripped)
    return value


def _as_float(value: Any, default: float) -> float:
    parsed = finite_float(value)
    return default if parsed is None else float(parsed)


def _setting_from_mapping(row: Mapping[str, Any], *, default_name: str) -> Setting:
    name = str(
        row.get("setting_name")
        or row.get("setting_id")
        or row.get("row_id")
        or row.get("planned_row_id")
        or row.get("name")
        or default_name
    )
    family = str(row.get("family") or row.get("topology") or row.get("topology_name") or "Extended")
    preset_raw = row.get("preset") or row.get("coefficient_preset_name") or row.get("coefficient_setting") or "Base"
    preset = str(preset_raw)
    if preset not in {"Base", "Aggressive"}:
        preset = "Base"
    allele_freqs_raw = _json_loads_maybe(row.get("allele_freqs") or {"GeneA": 0.02, "GeneB": 0.15})
    if not isinstance(allele_freqs_raw, Mapping):
        raise ValueError(f"{name} allele_freqs is not a mapping: {allele_freqs_raw!r}")
    return Setting(
        name=name,
        family=family,
        preset=preset,
        allele_freqs={str(k): float(v) for k, v in dict(allele_freqs_raw).items()},
        a_scale=_as_float(row.get("a_scale"), 1.0),
        b_scale=_as_float(row.get("b_scale"), 1.0),
        delta_shift=_as_float(row.get("delta_shift"), 0.0),
        fixed_cost=_as_float(row.get("fixed_cost"), 0.01),
        variable_cost=_as_float(row.get("variable_cost"), 0.02),
    )


def load_cases(suite: str) -> list[SuiteCase]:
    if suite == "original8":
        payload = read_json(ORIGINAL8_SOURCE)
        rows = list(payload.get("settings") or [])
        source = ORIGINAL8_SOURCE
        note = "canonical Original-8 setting manifest"
    elif suite == "local40":
        payload = read_json(LOCAL40_SOURCE)
        rows = list(payload.get("rows") or [])
        source = LOCAL40_SOURCE
        note = "local-40 setting rows"
    elif suite == "phase6_54":
        rows = read_csv(PHASE6_54_SOURCE)
        source = PHASE6_54_SOURCE
        note = "phase-6 extended 54-row setting manifest"
    else:
        raise ValueError(f"unsupported suite for this replication package: {suite!r}")

    cases: list[SuiteCase] = []
    for idx, row in enumerate(rows, start=1):
        setting = _setting_from_mapping(row, default_name=f"{suite}_row_{idx:03d}")
        row_id = str(
            row.get("row_id")
            or row.get("setting_id")
            or row.get("setting_name")
            or row.get("planned_row_id")
            or setting.name
        )
        cases.append(
            SuiteCase(
                suite=suite,
                row_index=idx,
                row_id=row_id,
                setting=setting,
                source_path=str(source),
                source_note=note,
            )
        )
    return cases
