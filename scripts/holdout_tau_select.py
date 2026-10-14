"""Holdout-validation threshold selection on GPQA-Diamond.

Procedure (single committed split):
  1. Split Diamond's n questions into a validation subset and an eval subset.
  2. For each tau, evaluate its K checkpoints (seeds) on the VALIDATION subset
     and average -> pick tau* = argmax validation average.
  3. Evaluate tau*'s K checkpoints on the EVAL subset; report per-seed
     accuracies + their mean / SD / SE.

Threshold is chosen only on validation, so the eval-subset number is leak-free.
Single judges (no tau to select) are reported directly on the same eval subset.

The paper uses a 50/50 split (the default). Pass several --val-fracs to see
how the selection depends on the split size.

Usage:
  python scripts/holdout_tau_select.py            # 50/50 split, split seed 0
"""
from __future__ import annotations
import argparse, json, os
import numpy as np

EVAL_DIR = os.path.join(os.environ.get("RAR_STAGE", f"/workspace/home/{os.environ.get('USER', '')}/deep_rl_stage"), "eval")
SEEDS = ["", "_s1", "_s2", "_s3", "_s4"]


def _load(fmt) -> np.ndarray:
    """(K_seeds, n_prompts) array of per-prompt rewards."""
    return np.array([json.load(open(os.path.join(EVAL_DIR, fmt(s))))["rewards"] for s in SEEDS])


def load_ensemble(taus):
    return {t: _load(lambda s, t=t: f"gpqa_mt8192_jvr_sci_dsmini_softlikert_drop{t}{s}.json") for t in taus}


def load_judges(judges):
    return {j: _load(lambda s, j=j: f"gpqa_mt8192_jvr_sci_{j}{s}.json") for j in judges}


def holdout_select(ENS, taus, val_frac, split_seed, n):
    """One committed split: select tau on val, return tau* + eval per-seed."""
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    nv = int(round(n * val_frac))
    vi, ei = perm[:nv], perm[nv:]
    val_avg = {t: ENS[t][:, vi].mean() for t in taus}          # 5-seed avg on val
    tstar = max(val_avg, key=val_avg.get)
    eval_per_seed = ENS[tstar][:, ei].mean(axis=1)             # per-seed on eval
    return tstar, val_avg, vi, ei, eval_per_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-fracs", type=float, nargs="+",
                    default=[0.50])
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--taus", nargs="+", default=["030", "040", "045", "050"])
    ap.add_argument("--judges", nargs="+", default=["mini", "oss", "qw", "ds", "gem"])
    args = ap.parse_args()

    ENS = load_ensemble(args.taus)
    JUD = load_judges(args.judges)
    n = ENS[args.taus[0]].shape[1]

    print(f"GPQA-Diamond holdout tau-selection (split-seed={args.split_seed}, n={n}, K={len(SEEDS)} seeds)\n")
    hdr = f"{'val/eval':>9} {'n_val':>5} {'n_eval':>6}  {'tau*':>7}  {'eval mean':>9} {'±SD':>6} {'±SE':>6}   " + " ".join(f"{j:>5}" for j in args.judges)
    print(hdr); print("-"*len(hdr))
    for vf in args.val_fracs:
        tstar, val_avg, vi, ei, eps = holdout_select(ENS, args.taus, vf, args.split_seed, n)
        m, sd = eps.mean(), eps.std(ddof=1); se = sd/np.sqrt(len(eps))
        jud = {j: JUD[j][:, ei].mean() for j in args.judges}
        nv = len(vi)
        print(f"{int(round(vf*100)):>4}/{100-int(round(vf*100)):<4} {nv:>5} {n-nv:>6}  drop{tstar:>3}  "
              f"{m:>9.4f} {sd:>6.3f} {se:>6.3f}   " + " ".join(f"{jud[j]:.3f}" for j in args.judges))

    # Detail for the largest val frac (most reliable selection)
    vf = max(args.val_fracs)
    tstar, val_avg, vi, ei, eps = holdout_select(ENS, args.taus, vf, args.split_seed, n)
    print(f"\nDetail @ {int(round(vf*100))}% val — validation averages (5-seed):")
    for t in args.taus:
        print(f"   drop{t}: {val_avg[t]:.4f}{'  <- selected' if t==tstar else ''}")
    print(f"   eval-subset per-seed for drop{tstar}: " + ", ".join(f"{x:.3f}" for x in eps))


if __name__ == "__main__":
    main()
