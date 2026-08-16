"""Step 9 — Training: Q-value GNN over 3-gene pedigrees (multi-topology).

The model predicts Q*(s, i) for each state s and person i:
    Q*(s, i) = r_test(i, s) + E[V*(s')] for untested persons.

Action selection at eval time: argmax_i Q(s, i) over untested persons.
This replaces the old V*(s) formulation where actions were chosen externally.

Node features (10 per person):
    [P(g=0)_GeneA .. P(g=2)_GeneC, is_tested]

Cost vector (11 scalars, global per config):
    [a, b, delta per gene] x 3 genes + [fixed_cost, variable_cost]
    Broadcast to each node before the head MLP.

Run:
    python step9_gnn_3gene/run.py --device cuda --epochs 500
"""
from __future__ import annotations

import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE         = Path(__file__).resolve().parent
EXPERIMENTS  = HERE.parent
PROJECT_ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENTS))

from shared.model    import PedigreeGNN
from shared.data_gen import FAMILY_CASES
from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree
from genetic_dp.exact_dp.utils import GENOTYPE_STATES, lift_tuple_posteriors_to_genes
from genetic_dp.models.belief  import InferenceResult
from genetic_dp.models.reward  import r_reward_test

GENES         = ("GeneA", "GeneB", "GeneC")
NODE_FEAT_DIM = 3 * len(GENES) + 1   # 10
COST_DIM      = 3 * len(GENES) + 2   # 11
CACHE_DIR     = HERE / "results" / "cache"

FAMILIES = ["Trio", "Nuclear", "ThreeGeneration"]

ALLELE_REGIMES = [
    "LowHigh", "MediumEven", "LowLow", "HighHigh", "MixedA", "MixedB",
]

TEST_KEYS = {
    "ThreeGeneration_MediumEven_Aggressive_3gene",
    "ThreeGeneration_HighHigh_Aggressive_3gene",
    "Nuclear_MediumEven_Aggressive_3gene",
    "Nuclear_HighHigh_Aggressive_3gene",
    "Trio_MediumEven_Aggressive_3gene",
    "Trio_HighHigh_Aggressive_3gene",
}

ALL_KEYS = [
    f"{family}_{regime}_{preset}_3gene"
    for family in FAMILIES
    for regime  in ALLELE_REGIMES
    for preset  in ["Base", "Aggressive"]
]

TRAIN_KEYS = [k for k in ALL_KEYS if k not in TEST_KEYS]


# ─── helpers ─────────────────────────────────────────────────────────────────

def config_to_cost_vec(config, genes) -> np.ndarray:
    """(a, b, delta) per gene + fixed + variable. dim = 3*n_genes + 2 = 11."""
    vec = []
    for g in genes:
        vec.append(min(config.a_gene[g].values(), key=abs))
        vec.append(min(config.b_gene[g].values(), key=abs))
        vec.append(list(config.delta_gene[g].values())[0])
    vec.append(config.fixed_cost)
    vec.append(config.variable_cost)
    return np.array(vec, dtype=np.float32)


def build_edge_index(family_label: str, individuals: list) -> np.ndarray:
    pedigree = generate_deterministic_pedigree(FAMILY_CASES[family_label])
    idx = {p: i for i, p in enumerate(individuals)}
    srcs, dsts = [], []
    for child in pedigree.get_offspring():
        for parent in pedigree.get_parents(child):
            if parent in idx and child in idx:
                srcs.append(idx[parent])
                dsts.append(idx[child])
    return np.array([srcs, dsts], dtype=np.int64) if srcs else np.zeros((2, 0), dtype=np.int64)


def extract_node_features(ds: dict, individuals: list) -> np.ndarray:
    """Returns (N_states, N_people, 10)."""
    X      = ds["X"]
    states = ds["states"]
    n_gene_feats = 3 * len(ds.get("genes", GENES))
    n_people     = len(individuals)
    person_idx   = {p: i for i, p in enumerate(individuals)}
    N   = len(states)
    nf  = X.reshape(N, n_people, n_gene_feats).copy()
    col = np.zeros((N, n_people, 1), dtype=np.float32)
    for i, state in enumerate(states):
        for person, _ in state:
            col[i, person_idx[person], 0] = 1.0
    return np.concatenate([nf, col], axis=-1)   # (N, n_people, 10)


def compute_q_labels(ds: dict, individuals: list, genes: tuple) -> np.ndarray:
    """
    Compute Q*(s, i) for every state s and person i.

    Q*(s, i) = r_test(i, s) + E_{g ~ P(g|s,i)}[V*(s | {(i,g)})]
             for untested persons; NaN for tested persons.

    Computed from existing exact DP values (ds["Y"]) — no re-running DP.
    Returns (N_states, N_people) float32 array.
    """
    states   = ds["states"]
    Y        = ds["Y"]
    belief   = ds["belief"]
    config   = ds["config"]
    N        = len(states)
    P        = len(individuals)
    pid      = {p: j for j, p in enumerate(individuals)}

    v_star = {s: float(Y[i]) for i, s in enumerate(states)}

    q = np.full((N, P), np.nan, dtype=np.float32)

    for i_s, s in enumerate(states):
        tested = {person for person, _ in s}
        if len(tested) == P:
            continue  # fully tested — no actions possible

        entry = belief[s]
        if isinstance(entry, InferenceResult):
            per_gene   = entry.get_per_gene_probs()
            tuple_pmfs = entry.get_tuple_pmfs()
        else:
            per_gene   = lift_tuple_posteriors_to_genes(entry, genes, GENOTYPE_STATES)
            tuple_pmfs = None

        for person in individuals:
            if person in tested:
                continue
            j = pid[person]

            r_i = float(r_reward_test(
                person, None, config.a, config.b, config.c, config.delta,
                config.fixed_cost, config.variable_cost,
                per_gene_probs=per_gene,
                a_gene=config.a_gene, c_gene=config.c_gene,
                delta_gene=config.delta_gene,
            ))

            pmf = (tuple_pmfs or {}).get(person, {})
            if not pmf and not isinstance(entry, InferenceResult):
                pmf = entry.get(person, {})

            exp_val = sum(
                prob * v_star.get(frozenset(s | {(person, g)}), 0.0)
                for g, prob in pmf.items() if prob > 1e-12
            )

            q[i_s, j] = r_i + exp_val

    return q


# ─── multi-topology training ──────────────────────────────────────────────────

def train_gnn(groups: list[dict], model: PedigreeGNN,
              epochs: int = 500, start_epoch: int = 1,
              lr: float = 1e-3, val_frac: float = 0.2,
              batch_size: int = 512, device: str = "cpu",
              print_every: int = 50, ckpt_path=None, ckpt_every: int = 50,
              history=None) -> tuple[PedigreeGNN, dict]:
    """
    Train on Q*(s, i) labels with masked MSE loss.
    Only untested persons contribute to the loss (NaN entries are masked out).

    groups: list of dicts, each with:
        "nf" : (N, N_people, 10) GPU tensor
        "gf" : (N, COST_DIM) GPU tensor
        "q"  : (N, N_people) GPU tensor  — Q* labels (NaN for tested persons)
        "ei" : (2, E) GPU tensor
    """
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if history is None:
        history = {"train_loss": [], "val_loss": []}

    train_groups, val_groups = [], []
    for g in groups:
        N   = g["nf"].shape[0]
        n_v = max(1, int(N * val_frac))
        n_t = N - n_v
        train_groups.append({"nf": g["nf"][:n_t], "gf": g["gf"][:n_t],
                              "q":  g["q"][:n_t],  "ei": g["ei"]})
        val_groups.append(  {"nf": g["nf"][n_t:], "gf": g["gf"][n_t:],
                              "q":  g["q"][n_t:],  "ei": g["ei"]})

    def masked_mse(q_pred, q_true):
        mask = ~torch.isnan(q_true)
        return ((q_pred[mask] - q_true[mask]) ** 2).mean()

    n_train = sum(g["nf"].shape[0] for g in train_groups)
    n_val   = sum(g["nf"].shape[0] for g in val_groups)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        for g in train_groups:
            M    = g["nf"].shape[0]
            perm = torch.randperm(M, device=device)
            for start in range(0, M, batch_size):
                sl = perm[start: start + batch_size]
                optimizer.zero_grad()
                q_pred = model.forward_batch(g["nf"][sl], g["ei"], g["gf"][sl])
                loss   = masked_mse(q_pred, g["q"][sl])
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(sl)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for g in val_groups:
                M = g["nf"].shape[0]
                for start in range(0, M, batch_size):
                    end = min(start + batch_size, M)
                    q_pred = model.forward_batch(g["nf"][start:end], g["ei"], g["gf"][start:end])
                    val_loss += masked_mse(q_pred, g["q"][start:end]).item() * (end - start)
        val_loss /= n_val

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % print_every == 0:
            print(f"  epoch {epoch:4d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}", flush=True)

        if ckpt_path is not None and epoch % ckpt_every == 0:
            tmp = Path(str(ckpt_path) + ".tmp")
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "history": history}, tmp)
            tmp.replace(Path(ckpt_path))

    return model, history


# ─── main ────────────────────────────────────────────────────────────────────

def main(device: str = "cpu", target_epochs: int = 500, fresh: bool = False):
    results_dir = HERE / "results"
    ckpt_path   = results_dir / "gnn3_ckpt.pt"
    results_dir.mkdir(exist_ok=True)

    log_f = open(results_dir / "train.log", "a")
    log_f.write(f"\n{'='*60}\n[TRAIN] {datetime.now().isoformat()}\n{'='*60}\n")
    def log(msg=""):
        print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()

    log(f"node_feat_dim={NODE_FEAT_DIM}  cost_dim={COST_DIM}  device={device}  epochs={target_epochs}")
    log(f"train configs: {len(TRAIN_KEYS)}  test configs: {len(TEST_KEYS)}")
    log("Output: Q*(s, i) per person per state (action-conditioned)")

    # ── 1. Verify all train pkls exist ────────────────────────────────────────
    log("\n[1] Cache check (train configs):")
    for key in TRAIN_KEYS:
        pkl = CACHE_DIR / f"{key}.pkl"
        if not pkl.exists():
            raise FileNotFoundError(f"Missing: {pkl}\nRun build_datasets.py first.")
        log(f"    {key}  [OK]")

    # ── 2. Load data, compute Q* labels, group by family topology ─────────────
    log("\n[2] Loading training data and computing Q* labels:")
    family_groups: dict[str, dict] = {}

    for key in TRAIN_KEYS:
        family = key.split("_")[0]
        t0 = time.time()
        with open(CACHE_DIR / f"{key}.pkl", "rb") as f:
            ds = pickle.load(f)

        individuals = ds["individuals"]

        if family not in family_groups:
            ei = build_edge_index(family, individuals)
            family_groups[family] = {
                "nf_parts": [], "gf_parts": [], "q_parts": [],
                "ei": ei, "individuals": individuals,
            }

        nf       = extract_node_features(ds, individuals)
        cost_vec = config_to_cost_vec(ds["config"], GENES)
        gf       = np.tile(cost_vec, (len(ds["states"]), 1))

        log(f"    {key}: {len(ds['states']):,} states — computing Q* labels...")
        q = compute_q_labels(ds, individuals, GENES)
        log(f"      done in {time.time()-t0:.1f}s")

        family_groups[family]["nf_parts"].append(nf)
        family_groups[family]["gf_parts"].append(gf)
        family_groups[family]["q_parts"].append(q)
        del ds

    groups = []
    for family, g in family_groups.items():
        nf_all = np.concatenate(g["nf_parts"], axis=0)
        gf_all = np.concatenate(g["gf_parts"], axis=0)
        q_all  = np.concatenate(g["q_parts"],  axis=0)
        n_valid = int((~np.isnan(q_all)).sum())
        log(f"    [{family}] {len(q_all):,} states  {n_valid:,} valid Q* entries")
        groups.append({
            "key": family,
            "nf":  torch.FloatTensor(nf_all).to(device),
            "gf":  torch.FloatTensor(gf_all).to(device),
            "q":   torch.FloatTensor(q_all).to(device),
            "ei":  torch.LongTensor(g["ei"]).to(device),
        })
        del nf_all, gf_all, q_all

    log(f"    Total training states: {sum(g['nf'].shape[0] for g in groups):,}")

    # ── 2c. Pipeline verification (write to verify.log — check manually) ──────
    vlog_path = results_dir / "verify.log"
    vlog_f    = open(vlog_path, "w")
    def vlog(msg=""):
        print(msg, flush=True); vlog_f.write(msg + "\n"); vlog_f.flush()

    vlog(f"{'='*70}")
    vlog(f"PIPELINE VERIFICATION  —  {datetime.now().isoformat()}")
    vlog(f"{'='*70}")

    _vkey  = TRAIN_KEYS[0]
    vlog(f"\nUsing config: {_vkey}")
    with open(CACHE_DIR / f"{_vkey}.pkl", "rb") as f:
        _vds = pickle.load(f)

    _vind    = _vds["individuals"]
    _vstates = _vds["states"]
    _vY      = _vds["Y"]
    _vbelief = _vds["belief"]
    _vconfig = _vds["config"]
    _vgenes  = _vds.get("genes", GENES)
    _vstidx  = {s: i for i, s in enumerate(_vstates)}
    _vstar   = {s: float(_vY[i]) for i, s in enumerate(_vstates)}

    # [A] DP value sanity
    vlog(f"\n[A] DP VALUE SANITY")
    vlog(f"    Individuals:          {_vind}")
    vlog(f"    Total states:         {len(_vstates):,}")
    vlog(f"    V_root (ds field):    {_vds['V_root']:.8f}")
    vlog(f"    V_stop_root (ds):     {_vds['V_stop_root']:.8f}")

    _vroot = frozenset()
    _v_root_dp = _vstar.get(_vroot, float("nan"))
    _match = abs(_v_root_dp - _vds["V_root"]) < 1e-6
    vlog(f"    V*(empty state):      {_v_root_dp:.8f}  {'OK — matches V_root' if _match else 'MISMATCH!'}")

    _full   = [s for s in _vstates if len(s) == len(_vind)]
    _nz     = sum(1 for s in _full if abs(_vstar[s]) > 1e-6)
    vlog(f"    Fully-tested states:  {len(_full)}")
    vlog(f"    V*(fully-tested) != 0:{_nz}  (should be 0)")
    for _s in _full[:3]:
        vlog(f"      V*({dict(_s)}) = {_vstar[_s]:.8f}")

    # [B] Q* formula check
    vlog(f"\n[B] Q* FORMULA CHECK  —  Q*(s,i) = r_test(i,s) + E[V*(s')]")
    _partial = [s for s in _vstates if 0 < len(s) < len(_vind)][:5]
    for _s in _partial:
        vlog(f"\n    State: {dict(_s)}")
        _tested = {p for p, _ in _s}
        _entry  = _vbelief[_s]
        if isinstance(_entry, InferenceResult):
            _pg  = _entry.get_per_gene_probs()
            _pmfs = _entry.get_tuple_pmfs()
        else:
            _pg   = lift_tuple_posteriors_to_genes(_entry, _vgenes, GENOTYPE_STATES)
            _pmfs = None
        for _p in _vind:
            if _p in _tested:
                vlog(f"      {_p}: TESTED → NaN"); continue
            _ri = float(r_reward_test(
                _p, None, _vconfig.a, _vconfig.b, _vconfig.c, _vconfig.delta,
                _vconfig.fixed_cost, _vconfig.variable_cost,
                per_gene_probs=_pg, a_gene=_vconfig.a_gene,
                c_gene=_vconfig.c_gene, delta_gene=_vconfig.delta_gene,
            ))
            _pmf = (_pmfs or {}).get(_p, {})
            _ev  = sum(prob * _vstar.get(frozenset(_s | {(_p, g)}), 0.0)
                       for g, prob in _pmf.items() if prob > 1e-12)
            vlog(f"      {_p}: r_test={_ri:.6f}  E[V*]={_ev:.6f}  Q*={_ri+_ev:.6f}")

    # [C] Node feature sanity
    vlog(f"\n[C] NODE FEATURE SANITY")
    _nf_v = extract_node_features(_vds, _vind)
    vlog(f"    Shape: {_nf_v.shape}  (expected ({len(_vstates)}, {len(_vind)}, {NODE_FEAT_DIM}))")
    _n_bad_prob, _n_bad_flag = 0, 0
    for _i, _sv in enumerate(_vstates[:100]):
        for _j in range(len(_vind)):
            for _gi in range(len(_vgenes)):
                if abs(_nf_v[_i, _j, _gi*3:_gi*3+3].sum() - 1.0) > 1e-3:
                    _n_bad_prob += 1
            _exp_flag = 1.0 if _vind[_j] in {pp for pp, _ in _sv} else 0.0
            if abs(_nf_v[_i, _j, -1] - _exp_flag) > 0.5:
                _n_bad_flag += 1
    vlog(f"    Prob cols not summing to 1 (first 100 states): {_n_bad_prob}  (should be 0)")
    vlog(f"    is_tested flag wrong (first 100 states):        {_n_bad_flag}  (should be 0)")
    vlog(f"    Root-state node features:")
    for _j, _pp in enumerate(_vind):
        vlog(f"      {_pp}: probs={_nf_v[0,_j,:NODE_FEAT_DIM-1].tolist()}  is_tested={_nf_v[0,_j,-1]:.0f}")

    # [D] Model output shape
    vlog(f"\n[D] MODEL OUTPUT SHAPE CHECK  (must be (B, N_people), not (B,))")
    _fam_v  = _vkey.split("_")[0]
    _ei_v   = torch.LongTensor(build_edge_index(_fam_v, _vind))
    _mtest  = PedigreeGNN(node_feat_dim=NODE_FEAT_DIM, hidden_dim=32, n_rounds=2, global_feat_dim=COST_DIM)
    _mtest.eval()
    _cv_v   = config_to_cost_vec(_vconfig, GENES)
    _nft_v  = torch.FloatTensor(_nf_v[:4])
    _gft_v  = torch.FloatTensor(np.tile(_cv_v, (4, 1)))
    with torch.no_grad():
        _out_v = _mtest.forward_batch(_nft_v, _ei_v, _gft_v)
    vlog(f"    Output shape: {tuple(_out_v.shape)}  (expected (4, {len(_vind)}))")
    vlog(f"    Sample Q values per state (untrained model):")
    for _b in range(4):
        vlog(f"      state {_b}: { {_vind[_j]: round(float(_out_v[_b,_j]),4) for _j in range(len(_vind))} }")

    # [E] Q* label range and NaN mask
    vlog(f"\n[E] Q* LABEL MASK CHECK")
    _ql_v = compute_q_labels(_vds, _vind, _vgenes)
    _nan_cnt   = int(np.isnan(_ql_v).sum())
    _valid_cnt = int((~np.isnan(_ql_v)).sum())
    vlog(f"    Shape: {_ql_v.shape}  (expected ({len(_vstates)}, {len(_vind)}))")
    vlog(f"    NaN entries   (tested persons):   {_nan_cnt}")
    vlog(f"    Valid entries (untested persons):  {_valid_cnt}")
    vlog(f"    Q* value range: [{float(np.nanmin(_ql_v)):.6f},  {float(np.nanmax(_ql_v)):.6f}]")

    vlog(f"\n{'='*70}")
    vlog(f"VERIFICATION DONE. Read verify.log for full output.")
    vlog(f"{'='*70}\n")
    vlog_f.close()
    log(f"    Verification written → {vlog_path}")
    del _vds, _nf_v, _ql_v

    # ── 3. Train ──────────────────────────────────────────────────────────────
    log(f"\n[3] Training GNN:")
    model = PedigreeGNN(node_feat_dim=NODE_FEAT_DIM, hidden_dim=32, n_rounds=2, global_feat_dim=COST_DIM)

    start_epoch = 1
    history     = {"train_loss": [], "val_loss": []}
    if not fresh and ckpt_path.exists():
        ckpt        = torch.load(ckpt_path, map_location=device)
        start_epoch = ckpt["epoch"] + 1
        model.load_state_dict(ckpt["model_state"])
        history     = ckpt.get("history", history)
        log(f"    Resumed from checkpoint epoch {ckpt['epoch']}/{target_epochs}")
    else:
        log(f"    Starting fresh (0/{target_epochs})")

    if start_epoch <= target_epochs:
        t0 = time.time()
        model, history = train_gnn(
            groups, model,
            epochs=target_epochs, start_epoch=start_epoch,
            lr=1e-3, batch_size=512, device=device,
            print_every=50, ckpt_path=ckpt_path, ckpt_every=50,
            history=history,
        )
        log(f"    Done in {time.time()-t0:.1f}s")
    else:
        log(f"    Already at {start_epoch-1}/{target_epochs} — skipping.")

    torch.save(model.state_dict(), results_dir / "gnn3_model.pt")
    log(f"\n[DONE] model saved → gnn3_model.pt")
    if history["train_loss"]:
        log(f"       train_loss={history['train_loss'][-1]:.6f}  val_loss={history['val_loss'][-1]:.6f}")
    log("Run eval.py next.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--fresh",  action="store_true")
    args = p.parse_args()
    main(device=args.device, target_epochs=args.epochs, fresh=args.fresh)
