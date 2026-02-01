# simulate.py
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from abar import ABAR, ABARConfig


def build_synthetic(
    m_av: int,
    replicas_per_av: int,
    seed: int = 7,
):
    rng = np.random.default_rng(seed)
    av_ids = [f"AV{i+1}" for i in range(m_av)]
    replica_ids_by_av = {}
    qos_by_replica = {}

    for av in av_ids:
        rids = []
        for j in range(replicas_per_av):
            rid = f"{av}_R{j+1}"
            rids.append(rid)
            # QoS in [0,1] (higher is "better / more reliable")
            qos_by_replica[rid] = float(np.clip(rng.normal(0.7, 0.15), 0.0, 1.0))
        replica_ids_by_av[av] = rids

    # latent corruption probability p_ij in [0,1]
    # make a small fraction more corrupt
    all_rids = [rid for av in av_ids for rid in replica_ids_by_av[av]]
    p = {}
    for rid in all_rids:
        base = rng.beta(1.0, 25.0)  # mostly small
        if rng.random() < 0.08:
            base = rng.uniform(0.15, 0.45)  # more risky
        p[rid] = float(base)

    return av_ids, replica_ids_by_av, qos_by_replica, p


def run_sim(
    rounds: int,
    cfg: ABARConfig,
    m_av: int,
    replicas_per_av: int,
    seed: int,
    out_csv: str,
):
    av_ids, replica_ids_by_av, qos_by_replica, p = build_synthetic(m_av, replicas_per_av, seed=seed)
    abar = ABAR(cfg, av_ids, replica_ids_by_av, qos_by_replica, force_explore_unseen=True)

    rng = np.random.default_rng(seed + 123)

    rows = []
    total_detect = 0
    total_audits = 0

    for t in range(1, rounds + 1):
        selected = abar.select(t)

        round_audits = 0
        round_detect = 0
        round_reward_sum = 0.0

        for av, rids in selected.items():
            for rid in rids:
                # simulate audit outcome: fail with prob p[rid]
                failed = (rng.random() < p[rid])
                R = abar.update_after_audit(rid, t, failed=failed)
                round_reward_sum += R
                round_audits += 1
                if failed:
                    round_detect += 1

        abar.update_tsl(selected)

        total_detect += round_detect
        total_audits += round_audits

        rows.append(
            {
                "t": t,
                "audits": round_audits,
                "detections": round_detect,
                "detection_rate": (round_detect / round_audits) if round_audits > 0 else 0.0,
                "reward_sum": round_reward_sum,
                "cum_audits": total_audits,
                "cum_detections": total_detect,
                "cum_detection_rate": (total_detect / total_audits) if total_audits > 0 else 0.0,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Saved logs to: {out_csv}")
    print(f"Final cumulative detection rate = {df['cum_detection_rate'].iloc[-1]:.4f}")
    print(f"Total audits = {total_audits}, total detections = {total_detect}")

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--m_av", type=int, default=3)
    ap.add_argument("--replicas_per_av", type=int, default=30)
    ap.add_argument("--G", type=int, default=20)
    ap.add_argument("--beta1", type=float, default=0.2)
    ap.add_argument("--beta2", type=float, default=0.2)
    ap.add_argument("--beta3", type=float, default=0.6)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--lam", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out_csv", type=str, default="abar_sim_logs.csv")
    args = ap.parse_args()

    cfg = ABARConfig(
        beta1=args.beta1,
        beta2=args.beta2,
        beta3=args.beta3,
        gamma=args.gamma,
        lam=args.lam,
        G=args.G,
    )
    run_sim(
        rounds=args.rounds,
        cfg=cfg,
        m_av=args.m_av,
        replicas_per_av=args.replicas_per_av,
        seed=args.seed,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":
    main()
