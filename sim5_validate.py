#!/usr/bin/env python3
"""Validation of round-3 finalists on a fresh seed / larger N (paired)."""
import random
from collections import Counter
import sim4

CANDS = {
    "r2 (baseline)": dict(sim4.DEFAULT),
    "risk_w1 guard=450": dict(sim4.DEFAULT, repeat_key="risk_w1", guard=450),
    "risk_w1 guard=500": dict(sim4.DEFAULT, repeat_key="risk_w1", guard=500),
    "cost lam=0.002 guard=450": dict(sim4.DEFAULT, repeat_key="cost", lam=0.002, guard=450),
    "cost lam=0.002 guard=500": dict(sim4.DEFAULT, repeat_key="cost", lam=0.002, guard=500),
    "cost lam=0.002 guard=550": dict(sim4.DEFAULT, repeat_key="cost", lam=0.002, guard=550),
    "cost lam=0.002 guard=600": dict(sim4.DEFAULT, repeat_key="cost", lam=0.002, guard=600),
    "cost lam=0.002 guard=650": dict(sim4.DEFAULT, repeat_key="cost", lam=0.002, guard=650),
    "cost lam=0.0025 guard=550": dict(sim4.DEFAULT, repeat_key="cost", lam=0.0025, guard=550),
    "cost lam=0.003 guard=550": dict(sim4.DEFAULT, repeat_key="cost", lam=0.003, guard=550),
    "risk_w1 guard=550": dict(sim4.DEFAULT, repeat_key="risk_w1", guard=550),
    "risk_w1 guard=600": dict(sim4.DEFAULT, repeat_key="risk_w1", guard=600),
}


def main():
    n = 3000
    rng = random.Random(777)
    instances = [sim4.draw_instance(rng) for _ in range(n)]
    prac = [sim4.load_practice(t) for t in (0, 1, 2)]
    for name, spec in CANDS.items():
        p = [sim4.run_instance(x, spec) for x in prac]
        syn = [sim4.run_instance(x, spec) for x in instances]
        calls = [r["calls"] for r in syn]
        hist = Counter(calls)
        sc99 = sum(1 for r in syn if r["score"] == 99)
        losses = sorted(r["loss"] for r in syn)
        print(f"{name:28s} prac {p[0]['calls']}c/q{p[0]['quality']} "
              f"{p[1]['calls']}c/q{p[1]['quality']} {p[2]['calls']}c/q{p[2]['quality']} | "
              f"syn mean={sum(calls)/len(calls):.2f} "
              f"P<=21={sum(1 for c in calls if c <= 21)/len(calls):.3f} "
              f"P<=22={sum(1 for c in calls if c <= 22)/len(calls):.3f} "
              f"hist20={hist.get(20,0)} 21={hist.get(21,0)} 22={hist.get(22,0)} "
              f"23={hist.get(23,0)} 24+={sum(v for k, v in hist.items() if k >= 24)} "
              f"score99={sc99}/{n} loss mean={sum(losses)/len(losses):.0f} "
              f"p99={losses[99*n//100]} max={losses[-1]}")


if __name__ == "__main__":
    main()
