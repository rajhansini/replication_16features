"""
Runs after step5 GNN finishes. Reads MLP, GNN, and Gurobi ADP results,
writes a clean comparison table to output/final_comparison.txt.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

mlp = json.loads((ROOT / "ground-up-experiments/step4_all_families/results/partial_eval.json").read_text())
gnn_raw = json.loads((ROOT / "ground-up-experiments/step5_gnn/results/results.json").read_text())
adp_raw = json.loads((ROOT / "output/original8_combined.json").read_text())

adp_map = {r["row_id"]: r["reproduced"]["ratio2"] for r in adp_raw["rows"]}

# Build GNN map — key format is Family__Preset__allele_freq
# Map back to canonical row_id
def gnn_to_row_id(key):
    family, preset, freq = key.split("__")
    regime = "LowHigh" if float(freq) < 0.05 else "MediumEven"
    return f"{family}_{regime}_{preset}"

gnn_map = {gnn_to_row_id(k): v["ratio2"] for k, v in gnn_raw["families"].items()}

ORDER = [
    "Extended_LowHigh_Base",
    "Extended_LowHigh_Aggressive",
    "Extended_MediumEven_Base",
    "Extended_MediumEven_Aggressive",
    "ThreeGeneration_LowHigh_Base",
    "ThreeGeneration_LowHigh_Aggressive",
    "ThreeGeneration_MediumEven_Base",
    "ThreeGeneration_MediumEven_Aggressive",
]

lines = []
lines.append("ratio2 = (V* - L_policy) / (V* - V_stop)  [0=optimal, 1=worst]")
lines.append("ADP  = fresh Gurobi, per-stage theta, 16 ABCD features")
lines.append("MLP  = flat MLP supervised on V*(s), corrected GeneB params")
lines.append("GNN  = pedigree GNN supervised on V*(s), corrected GeneB params")
lines.append("")
lines.append(f"{'Family':<45} {'Split':>5}  {'ADP r2':>8}  {'MLP r2':>8}  {'GNN r2':>8}  {'MLP beats ADP':>14}  {'GNN beats ADP':>14}")
lines.append("=" * 110)

for key in ORDER:
    m      = mlp.get(key, {})
    split  = m.get("split", "—")
    mlp_r2 = m.get("ratio2_net")
    adp_r2 = adp_map.get(key)
    gnn_r2 = gnn_map.get(key)

    def fmt(v): return f"{v:.4f}" if v is not None else "  —   "
    def cmp(net, adp):
        if net is None or adp is None: return "  —"
        if net < adp: return f"yes  {adp/net:.1f}x"
        return f"no  ({net/adp:.1f}x worse)"

    lines.append(
        f"{key:<45} {split:>5}  {fmt(adp_r2):>8}  {fmt(mlp_r2):>8}  {fmt(gnn_r2):>8}"
        f"  {cmp(mlp_r2, adp_r2):>14}  {cmp(gnn_r2, adp_r2):>14}"
    )

lines.append("=" * 110)

out = "\n".join(lines) + "\n"
out_path = ROOT / "output" / "final_comparison.txt"
out_path.write_text(out)
print(out)
print(f"\nSaved to {out_path}")
