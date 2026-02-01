# abar.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class ABARConfig:
    beta1: float = 0.2
    beta2: float = 0.2
    beta3: float = 0.6
    gamma: float = 0.3
    lam: float = 0.75
    # global budget (max verifications per round across all AVs)
    G: int = 20


@dataclass
class ReplicaState:
    # counts
    P: int = 0  # pass count
    N: int = 0  # total audits
    # derived
    SR: float = 1.0
    TSL: int = 0
    # learning value
    ubar: float = 0.0


class ABAR:
    """
    ABAR: Adaptive Bandit-Based Anomaly Ranking for replica selection.
    Implements:
      - anomaly score A(d,t) using SR, QoS, TSL
      - context multiplier C(d,t) (QoS + TSL part)
      - reward R(t) = SR * C if fail else 0
      - ubar running mean update
      - UCB index I(t) = lam*ubar + (1-lam)*A + gamma*sqrt(log t / (1+N))

    Notes:
      * TSL is unbounded here (Option A).
      * For first-time exploration, we optionally force I=+inf when N==0.
    """

    def __init__(
        self,
        cfg: ABARConfig,
        av_ids: List[str],
        replica_ids_by_av: Dict[str, List[str]],
        qos_by_replica: Dict[str, float],
        force_explore_unseen: bool = True,
    ):
        self.cfg = cfg
        self.av_ids = list(av_ids)
        self.replica_ids_by_av = {k: list(v) for k, v in replica_ids_by_av.items()}
        self.qos = dict(qos_by_replica)
        self.force_explore_unseen = force_explore_unseen

        # state per replica
        self.state: Dict[str, ReplicaState] = {}
        for av in self.av_ids:
            for rid in self.replica_ids_by_av[av]:
                self.state[rid] = ReplicaState()

    def anomaly_score(self, rid: str, t: int) -> float:
        s = self.state[rid]
        qos = self.qos[rid]
        # A(d,t) = b1(1-SR) + b2(1-QoS) + b3*TSL
        return (
            self.cfg.beta1 * (1.0 - s.SR)
            + self.cfg.beta2 * (1.0 - qos)
            + self.cfg.beta3 * float(s.TSL)
        )

    def context_multiplier(self, rid: str, t: int) -> float:
        # C(d,t) as in your method: weighted combination of (1-QoS) and TSL
        # C = (b2/(b2+b3))*(1-QoS) + (b3/(b2+b3))*TSL
        b2, b3 = self.cfg.beta2, self.cfg.beta3
        denom = (b2 + b3) if (b2 + b3) > 0 else 1.0
        qos = self.qos[rid]
        tsl = float(self.state[rid].TSL)
        return (b2 / denom) * (1.0 - qos) + (b3 / denom) * tsl

    def ucb_index(self, rid: str, t: int) -> float:
        s = self.state[rid]
        if self.force_explore_unseen and s.N == 0:
            return float("inf")
        A = self.anomaly_score(rid, t)
        bonus = self.cfg.gamma * np.sqrt(np.log(max(t, 2)) / (1.0 + s.N))
        return self.cfg.lam * s.ubar + (1.0 - self.cfg.lam) * A + bonus

    def allocate_budgets(self, I_by_av: Dict[str, List[Tuple[str, float]]]) -> Dict[str, int]:
        """
        Allocate per-AV budget b_i(t) = floor(w_i(t)*G),
        w_i(t)=barI_i(t)/sum_j barI_j(t), where barI_i(t) is avg score of top-ranked replicas.
        Here we compute barI_i(t) as mean of top-min(k, G) indices from that AV.
        """
        G = int(self.cfg.G)

        barI = {}
        for av, items in I_by_av.items():
            # items already include all replicas; sort descending
            items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
            topk = items_sorted[: max(1, min(len(items_sorted), G))]
            barI[av] = float(np.mean([v for _, v in topk])) if topk else 0.0

        denom = sum(max(0.0, v) for v in barI.values())
        if denom <= 0:
            # fallback: equal split
            base = G // max(1, len(self.av_ids))
            b = {av: base for av in self.av_ids}
            # distribute remainder
            rem = G - sum(b.values())
            for av in self.av_ids[:rem]:
                b[av] += 1
            return b

        w = {av: max(0.0, barI[av]) / denom for av in self.av_ids}
        b = {av: int(np.floor(w[av] * G)) for av in self.av_ids}

        # fix rounding to ensure sum <= G and use leftover
        used = sum(b.values())
        leftover = max(0, G - used)
        if leftover > 0:
            # give leftovers to AVs with largest fractional parts
            fracs = sorted(
                [(av, (w[av] * G) - np.floor(w[av] * G)) for av in self.av_ids],
                key=lambda x: x[1],
                reverse=True,
            )
            for av, _ in fracs[:leftover]:
                b[av] += 1

        # if overshoot (rare), trim
        while sum(b.values()) > G:
            # remove from AV with largest b
            av_max = max(b.items(), key=lambda x: x[1])[0]
            if b[av_max] > 0:
                b[av_max] -= 1
            else:
                break
        return b

    def select(self, t: int) -> Dict[str, List[str]]:
        """
        Returns selected replicas per AV for round t.
        """
        # compute indices per AV
        I_by_av: Dict[str, List[Tuple[str, float]]] = {}
        for av in self.av_ids:
            I_by_av[av] = []
            for rid in self.replica_ids_by_av[av]:
                I_by_av[av].append((rid, self.ucb_index(rid, t)))

        budgets = self.allocate_budgets(I_by_av)

        selected: Dict[str, List[str]] = {}
        for av in self.av_ids:
            items_sorted = sorted(I_by_av[av], key=lambda x: x[1], reverse=True)
            bi = min(budgets.get(av, 0), len(items_sorted))
            selected[av] = [rid for rid, _ in items_sorted[:bi]]
        return selected

    def update_after_audit(self, rid: str, t: int, failed: bool) -> float:
        """
        Update replica state after audit outcome.
        Returns the learning reward R_ij(t).
        """
        st = self.state[rid]
        # compute reward first using current SR and context
        if failed:
            R = st.SR * self.context_multiplier(rid, t)
        else:
            R = 0.0

        # update counts
        st.N += 1
        if not failed:
            st.P += 1
        st.SR = st.P / st.N if st.N > 0 else 1.0

        # update running mean ubar
        # ubar_{t+1} = (ubar_t*(N-1) + R)/N
        st.ubar = ((st.ubar * (st.N - 1)) + R) / st.N

        return float(R)

    def update_tsl(self, selected_per_av: Dict[str, List[str]]) -> None:
        """
        After round finishes: selected -> TSL=0, others -> +1
        """
        selected_set = set()
        for av, rids in selected_per_av.items():
            selected_set.update(rids)

        for rid, st in self.state.items():
            if rid in selected_set:
                st.TSL = 0
            else:
                st.TSL += 1
