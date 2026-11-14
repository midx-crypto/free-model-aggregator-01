#!/usr/bin/env python3
"""Offline simulator for free-model-aggregator-01 routing strategies (v2).

Replicates the env's provider behaviour and the scorer's quality formula,
using the practice instances' public data. Strategies are RUNTIME rules:
they see only the brief (ratings/workload) plus the outcomes of calls they
made, including the failure MESSAGE kind (budget/outage/scheduled) — which
the env's error body carries and r1 discards.

Usage: python3 sim2.py [strategy ...]
"""
import json, sys

K = "/home/hatch/workspace/makefun/free-model-aggregator-01-kit/free-model-aggregator-01-kit"
RATINGS = [
    ("llama-3.1-8b-instant", "groq", 1210, 131072, True, False, True),
    ("gpt-oss-120b", "groq", 1352, 131072, True, False, True),
    ("qwen3.8-27b", "groq", 1434, 131072, True, False, True),
    ("gpt-oss-20b", "groq", 1318, 131072, True, False, False),
    ("llama-4-scout-17b", "cerebras", 1250, 131072, True, False, False),
    ("qwen-3.8-27b", "cerebras", 1434, 131072, True, False, True),
    ("gpt-oss-120b", "cerebras", 1352, 131072, True, False, True),
    ("gemini-3.8-flash", "gemini", 1458, 1048576, True, True, True),
    ("gemma-4-31b", "gemini", 1452, 131072, True, True, False),
    ("gemma-4-26b", "gemini", 1410, 32768, False, True, False),
    ("glm-4.7-flash", "zai", 1470, 204800, True, False, True),
    ("glm-4.6v-flash", "zai", 1432, 204800, True, True, False),
    ("z-ai/glm-5.3-flash:free", "openrouter", 1475, 204800, True, False, True),
    ("deepseek/deepseek-v4.1:free", "openrouter", 1462, 163840, True, False, True),
    ("minimax/m3:free", "openrouter", 1436, 196608, True, False, True),
    ("deepseek-v3.2", "sambanova", 1465, 163840, True, False, True),
    ("gemma-4-31b", "sambanova", 1452, 131072, False, True, False),
    ("mistral-nemo", "pollinations", 1180, 32768, True, False, False),
    ("openai-fast", "pollinations", 1390, 32768, True, False, False),
    ("glm-4.7-flash", "cloudflare", 1470, 204800, True, False, True),
    ("llama-3.3-70b", "cloudflare", 1330, 131072, True, False, False),
    ("nemotron-3.5-super-vl", "cloudflare", 1414, 131072, False, True, False),
    ("qwen3.5-397b", "ovh", 1440, 262144, True, False, False),
    ("llama-3.3-70b", "ovh", 1330, 131072, True, False, False),
    ("kilo-fast", "kilol", 1140, 131072, True, False, False),
    ("kilo-auto/free", "kilol", 1420, 131072, True, False, False),
]
WIRE = {"groq": "openai", "cerebras": "openai", "gemini": "gemini", "zai": "anthropic",
        "openrouter": "openai", "sambanova": "openai", "pollinations": "openai",
        "cloudflare": "openai", "ovh": "openai", "kilol": "openai"}
BEST = {"r01": 1458, "r02": 1458, "r03": 1475, "r04": 1475, "r05": 1475, "r06": 1458,
        "r07": 1475, "r08": 1475, "r09": 1475, "r10": 1475, "r11": 1475, "r12": 1475,
        "r13": 1475, "r14": 1475, "r15": 1475, "r16": 1475, "r17": 1475, "r18": 1475,
        "r19": 1475, "r20": 1475}
WORKLOAD = [
    ("r01", "openai", 2, {"tools": True, "vision": True}),
    ("r02", "openai", 2, {"tools": True, "vision": True}),
    ("r03", "openai", 2, {"tools": True}),
    ("r04", "anthropic", 2, {"tools": True}),
    ("r05", "anthropic", 2, {"tools": True}),
    ("r06", "openai", 2, {"vision": True}),
    ("r07", "openai", 1, {"json": True}),
    ("r08", "anthropic", 1, {"json": True}),
    ("r09", "anthropic", 1, {"json": True}),
    ("r10", "openai", 1, {"min_context": 200000}),
    ("r11", "openai", 1, {"min_context": 200000}),
    ("r12", "anthropic", 1, {"min_context": 200000}),
    ("r13", "openai", 1, {}),
    ("r14", "openai", 1, {}),
    ("r15", "openai", 1, {}),
    ("r16", "anthropic", 1, {}),
    ("r17", "anthropic", 1, {}),
    ("r18", "openai", 1, {}),
    ("r19", "openai", 1, {}),
    ("r20", "anthropic", 1, {}),
]


def load_trial(t):
    p = json.load(open(f"{K}/practice/trial-{t}/input/env/providers.json"))
    return {pr["id"]: {"budget": pr["budget"], "fail": set(pr["fail_calls"]),
                       "outage": pr["outage_from"]} for pr in p["providers"]}


def meets(row, req):
    if req.get("tools") and not row[4]:
        return False
    if req.get("vision") and not row[5]:
        return False
    if req.get("json") and not row[6]:
        return False
    if "min_context" in req and row[3] < req["min_context"]:
        return False
    return True


def candidates(client_api, req, json_mode):
    out = []
    for prov in sorted(set(r[1] for r in RATINGS)):
        if client_api == "openai" and json_mode and WIRE[prov] == "anthropic":
            continue
        rows = [r for r in RATINGS if r[1] == prov and meets(r, req)]
        if not rows:
            continue
        rows.sort(key=lambda r: -r[2])
        out.append((prov, rows[0]))
    def key(c):
        prov, row = c
        same = 0 if WIRE[prov] == client_api else 1
        return (-row[2], same, prov, row[0])
    out.sort(key=key)
    return out


class State:
    """Runtime-visible state: per-provider calls made, consecutive failures,
    dead flags (budget/outage learned from the error message), successes."""

    def __init__(self, provs):
        self.provs = provs
        self.calls = {p: 0 for p in provs}
        self.consec = {p: 0 for p in provs}
        self.dead = {p: False for p in provs}
        self.used = {p: 0 for p in provs}

    def call(self, prov):
        self.calls[prov] += 1
        n = self.calls[prov]
        e = self.provs[prov]
        if n > e["budget"]:
            return (False, "budget")
        if e["outage"] is not None and n >= e["outage"]:
            return (False, "outage")
        if n in e["fail"]:
            return (False, "scheduled")
        return (True, None)


def run_trial(t, order_fn):
    """order_fn(cands, st, rid) -> ordered list of (prov,row) to try.
    After each attempt the strategy's state is updated by the harness here
    (consec/dead/used), exactly as a runtime implementation would."""
    st = State(load_trial(t))
    served = {}
    calls = 0
    for rid, api, weight, req in WORKLOAD:
        cands = candidates(api, req, req.get("json", False))
        order = order_fn(cands, st, rid)
        got = None
        for prov, row in order:
            if st.dead[prov]:
                continue
            calls += 1
            ok, kind = st.call(prov)
            if ok:
                st.consec[prov] = 0
                st.used[prov] += 1
                got = row
                break
            else:
                st.consec[prov] += 1
                if kind in ("budget", "outage"):
                    st.dead[prov] = True
        served[rid] = got
    bound = sum(w * BEST[rid] for rid, _, w, _ in WORKLOAD)
    earned = sum(w * min(served[rid][2], BEST[rid]) for rid, _, w, _ in WORKLOAD
                 if served[rid] is not None)
    quality = (100 * earned) // bound
    return {"trial": t, "calls": calls, "quality": quality,
            "served": sum(1 for v in served.values() if v is not None),
            "used": {k: v for k, v in st.used.items() if v},
            "dead": [k for k, v in st.dead.items() if v]}


# --- strategies: order_fn(cands, st, rid) ---

def s_r1(cands, st, rid):
    fresh = [c for c in cands if st.consec[c[0]] < 2]
    rested = [c for c in cands if st.consec[c[0]] >= 2]
    return fresh + rested


def s_r1_msg(cands, st, rid):
    """r1 + message-aware dead blacklisting (budget/outage kills the provider).
    The harness marks dead providers; this ordering is the same as r1 but
    dead ones are skipped by the harness."""
    return s_r1(cands, st, rid)


def s_cap2(cands, st, rid):
    """Soft cap: prefer providers with <2 calls used; others only as fallback.
    Within each tier keep r1's rating order; rest-after-2 as in r1."""
    def tier(c):
        if st.consec[c[0]] >= 2:
            return 2
        return 0 if st.calls[c[0]] < 2 else 1
    return sorted(cands, key=lambda c: tier(c))


def s_cap3(cands, st, rid):
    def tier(c):
        if st.consec[c[0]] >= 2:
            return 2
        return 0 if st.calls[c[0]] < 3 else 1
    return sorted(cands, key=lambda c: tier(c))


def s_cap1(cands, st, rid):
    def tier(c):
        if st.consec[c[0]] >= 2:
            return 2
        return 0 if st.calls[c[0]] < 1 else 1
    return sorted(cands, key=lambda c: tier(c))


def s_penalty(pen):
    def fn(cands, st, rid):
        def key(c):
            prov, row = c
            return (-(row[2] - pen * st.calls[prov]),
                    0 if WIRE[prov] == ("anthropic" if False else "openai") else 0)
        # keep r1 tie-breaks after the penalised rating
        def key2(c):
            prov, row = c
            return (-(row[2] - pen * st.calls[prov]), prov, row[0])
        return sorted(cands, key=key2)
    return fn


def s_rest1(cands, st, rid):
    """Rest after ONE consecutive failure (spread), + message dead."""
    fresh = [c for c in cands if st.consec[c[0]] == 0]
    rested = [c for c in cands if st.consec[c[0]] > 0]
    return fresh + rested


STRATS = {
    "r1": s_r1,
    "r1_msg": s_r1_msg,
    "cap1": s_cap1,
    "cap2": s_cap2,
    "cap3": s_cap3,
    "pen10": s_penalty(10),
    "pen20": s_penalty(20),
    "pen30": s_penalty(30),
    "pen50": s_penalty(50),
    "rest1": s_rest1,
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(STRATS)
    for name in names:
        fn = STRATS[name]
        line = []
        for t in [0, 1, 2]:
            r = run_trial(t, fn)
            line.append(f"t{t}: {r['calls']}c q{r['quality']} s{r['served']}")
        print(f"{name:8s} " + " | ".join(line))
