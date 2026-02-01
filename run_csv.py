# run_csv.py
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from abar import ABAR, ABARConfig


def normalize_01(x: pd.Series) -> pd.Series:
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mn) / (mx - mn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--G", type=int, default=20)
    ap.add_argument("--beta1", type=float, default=0.2)
    ap.add_argument("--beta2", type=float, default=0.2)
    ap.add_argument("--beta3", type=float, default=0.6)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--lam", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out_csv", type=str, default="abar_csv_logs.csv")
    ap.add_argument("--qos_col", type=str, default="qos")
    ap.add_argument("--av_col", type=str, default="av_id")
    ap.add_argument("--replica_col", type=str, default="replica_id")
    ap.add_argument("--failprob_col", type=str, default="fail_prob")  # if exists
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    for col in [args.av_col, args.replica_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if args.qos_col not in df.columns:
        raise ValueError(f"Missing QoS column '{args.qos_col}'. Provide it normalized or supply raw and normalize yourself.")

    # Ensure qos in [0,1]
    qos_raw = df[args.qos_col].astype(float)
    if (qos_raw.min() < 0) or (qos_raw.max() > 1):
        qos = normalize_01(qos_raw)
    else:
        qos = qos_raw

    df = df.copy()
    df["_qos01"] = qos

    av_ids = sorted(df[args.av_col].astype(str).unique().tolist())

    replica_ids_by_av = {}
    qos_by_replica = {}
    for av in av_ids:
        sub = df[df[args.av_col].astype(str) == str(av)]
        rids = sub[args.replica_col].astype(str).unique().tolist()
        replica_ids_by_av[str(av)] = rids
        # QoS per replica: take mean if repeated rows
        for rid in rids:
            qos_by_replica[rid] = float(sub[sub[args.replica_col].astype(str) == rid]["_qos01"].mean())

    cfg = ABARConfig(
        beta1=args.beta1, beta2=args.beta2, beta3=args.beta3,
        gamma=args.gamma, lam=args.lam, G=args.G
    )
    abar = ABAR(cfg, av_ids, replica_ids_by_av, qos_by_replica, force_explore_unseen=True)

    rng = np.random.default_rng(args.seed)

    # fail generation:
    # If you have a 'fail_prob' column, we simulate a per-audit fail from it.
    # Otherwise, if you have 'fail' column (0/1), we will sample from empirical mean per replica.
    fail_prob = {}
    if args.failprob_col in df.columns:
        for rid, grp in df.groupby(args.replica_col):
            fail_prob[str(rid)] = float(grp[args.failprob_col].astype(float).mean())
    elif "fail" in df.columns:
        for rid, grp in df.groupby(args.replica_col):
            fail_prob[str(rid)] = float(grp["fail"].astype(float).mean())
    else:
        # fallback: rare failures
        for av in av_ids:
            for rid in replica_ids_by_av[av]:
                fail_prob[rid] = 0.02

    logs = []
    cum_aud, cum_det = 0, 0

    for t in range(1, args.rounds + 1):
        selected = abar.select(t)

        round_aud, round_det, reward_sum = 0, 0, 0.0
        for av, rids in selected.items():
            for rid in rids:
                p = float(fail_prob.get(rid, 0.02))
                failed = (rng.random() < p)
                R = abar.update_after_audit(rid, t, failed=failed)
                reward_sum += R
                round_aud += 1
                round_det += int(failed)

        abar.update_tsl(selected)

        cum_aud += round_aud
        cum_det += round_det

        logs.append({
            "t": t,
            "audits": round_aud,
            "detections": round_det,
            "detection_rate": (round_det / round_aud) if round_aud else 0.0,
            "reward_sum": reward_sum,
            "cum_audits": cum_aud,
            "cum_detections": cum_det,
            "cum_detection_rate": (cum_det / cum_aud) if cum_aud else 0.0
        })

    out = pd.DataFrame(logs)
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved logs to: {args.out_csv}")
    print(f"Final cumulative detection rate = {out['cum_detection_rate'].iloc[-1]:.4f}")


if __name__ == "__main__":
    main()
