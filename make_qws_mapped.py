# make_qws_mapped.py
import argparse
import numpy as np
import pandas as pd

# QWS v2 commonly includes these QoS columns (names vary slightly across releases)
POSITIVE = [
    "Availability", "Throughput", "Successability", "Reliability",
    "Compliance", "Best Practices", "Documentation"
]
NEGATIVE = ["Response Time", "Latency"]  # lower is better

def minmax(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mn) / (mx - mn)

def build_qos(df: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    # Normalize positives
    comps = {}
    for c in POSITIVE:
        if c in df.columns:
            comps[c] = minmax(df[c])
    # Normalize and invert negatives
    for c in NEGATIVE:
        if c in df.columns:
            comps[c] = 1.0 - minmax(df[c])

    if len(comps) == 0:
        raise ValueError("No recognized QoS columns found. Check your QWS header names.")

    # Default: equal weights over available components
    cols = list(comps.keys())
    if weights is None:
        w = {c: 1.0 / len(cols) for c in cols}
    else:
        # Use provided weights but only for available columns
        w = {c: float(weights[c]) for c in cols if c in weights}
        s = sum(w.values())
        if s <= 0:
            raise ValueError("Weights sum to zero.")
        w = {c: v / s for c, v in w.items()}

    qos = np.zeros(len(df), dtype=float)
    for c in cols:
        qos += w[c] * comps[c].to_numpy()

    # Ensure [0,1]
    qos = np.clip(qos, 0.0, 1.0)
    return pd.Series(qos, index=df.index, name="qos")

def build_fail_prob(qos: pd.Series, p_min=0.01, p_max=0.20, alpha=2.0) -> pd.Series:
    fp = p_min + (p_max - p_min) * np.power((1.0 - qos.to_numpy()), alpha)
    fp = np.clip(fp, 0.0, 1.0)
    return pd.Series(fp, index=qos.index, name="fail_prob")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qws_csv", required=True, help="Raw QWS v2 CSV")
    ap.add_argument("--out_csv", default="your_qws_mapped.csv")
    ap.add_argument("--seed", type=int, default=7)

    # Attack model params
    ap.add_argument("--p_min", type=float, default=0.01)
    ap.add_argument("--p_max", type=float, default=0.20)
    ap.add_argument("--alpha", type=float, default=2.0)

    # Synthetic MVMS mapping (only if you don't have AV IDs in the dataset)
    ap.add_argument("--num_av", type=int, default=3, help="How many AVs to simulate")
    args = ap.parse_args()

    df = pd.read_csv(args.qws_csv)

    # Identify a stable id column (fall back to row index)
    if "WSDL Address" in df.columns:
        base_id = df["WSDL Address"].astype(str)
    elif "Service Name" in df.columns:
        base_id = df["Service Name"].astype(str)
    else:
        base_id = pd.Series([f"svc_{i}" for i in range(len(df))])

    df = df.copy()
    df["service_id"] = base_id

    # Compute qos ∈ [0,1]
    df["qos"] = build_qos(df)

    # Create fail_prob from attack model
    df["fail_prob"] = build_fail_prob(df["qos"], p_min=args.p_min, p_max=args.p_max, alpha=args.alpha)

    # Create MVMS ids (if no AV label exists)
    rng = np.random.default_rng(args.seed)
    av_ids = [f"AV{i+1}" for i in range(args.num_av)]
    df["av_id"] = rng.choice(av_ids, size=len(df), replace=True)

    # Treat each service as one "replica" (or you can expand to multiple replicas per service)
    df["replica_id"] = df["service_id"].astype(str)

    out = df[["service_id", "av_id", "replica_id", "qos", "fail_prob"]].copy()
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {args.out_csv} with {len(out)} rows.")

if __name__ == "__main__":
    main()
