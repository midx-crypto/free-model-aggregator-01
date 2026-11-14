#!/usr/bin/env python3
"""Offline simulator for free-model-aggregator-01 routing strategies.

Replicates the env's provider behaviour (budget / fail_calls / outage_from)
and the scorer's quality formula, using the practice instances' public data.
Used to design and measure routing strategies BEFORE running grade-local.

Usage: python3 sim.py [strategy] [trial ...]
Strategies: r1, spread, spread2, oracle
"""
import json, sys

K = "/home/hatch/workspace/makefun/free-model-aggregator-01-kit/free-model-aggregator-01-kit"

# Ratings from the sealed table (prompt.txt / scorer PROVIDERS), free models only.
RATINGS = [
    # model, provider, rating, ctx, tools, vision, json
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
    provs = {}
    for pr in p["providers"]:
        provs[pr["id"]] = {"budget": pr["budget"], "fail": set(pr["fail_calls"]),
                           "outage": pr["outage_from"]}
    return provs


def meets(row, req):
    _, _, _, _, tools, vision, js = row
    if req.get("tools") and not tools:
        return False
    if req.get("vision") and not vision:
        return False
    if req.get("json") and not js:
        return False
    if "min_context" in req and row[3] < req["min_context"]:
        return False
    return True


def candidates(client_api, req, json_mode):
    """One best model per provider, sorted like r1: rating desc, same-wire, provider id, model id."""
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


class Env:
    def __init__(self, provs):
        self.provs = provs
        self.calls = {p: 0 for p in provs}

    def call(self, prov):
        """Returns True on success. Mirrors server.mjs refusal logic."""
        self.calls[prov] += 1
        n = self.calls[prov]
        e = self.provs[prov]
        if n > e["budget"]:
            return False
        if e["outage"] is not None and n >= e["outage"]:
            return False
        if n in e["fail"]:
            return False
        return True


def run_trial(t, strategy, verbose=False):
    provs = load_trial(t)
    env = Env(provs)
    consec = {p: 0 for p in provs}
    total_fail = {p: 0 for p in provs}
    used = {p: 0 for p in provs}
    served = {}
    calls = 0
    for rid, api, weight, req in WORKLOAD:
        json_mode = req.get("json", False)
        cands = candidates(api, req, json_mode)
        got = None
        order = strategy(cands, consec, env, rid)
        for prov, row in order:
            calls += 1
            ok = env.call(prov)
            if ok:
                consec[prov] = 0
                used[prov] += 1
                got = row
                break
            else:
                consec[prov] += 1
                total_fail[prov] += 1
                if verbose:
                    print(f"  {rid}: FAIL {prov}/{row[0]} (call #{env.calls[prov]})")
        served[rid] = got
    # quality
    bound = sum(w * BEST[rid] for rid, _, w, _ in WORKLOAD)
    earned = 0
    for rid, _, w, _ in WORKLOAD:
        row = served[rid]
        if row is not None:
            earned += w * min(row[2], BEST[rid])
    quality = (100 * earned) // bound
    served_n = sum(1 for v in served.values() if v is not None)
    # failover: requests whose first call failed but were served
    return {"trial": t, "calls": calls, "quality": quality, "served": served_n,
            "used": used, "fails": total_fail,
            "served_models": {rid: (s[0] if s else None) for rid, s in served.items()}}


def strat_r1(cands, consec, env, rid):
    fresh = [c for c in cands if consec[c[0]] < 2]
    rested = [c for c in cands if consec[c[0]] >= 2]
    return fresh + rested


def strat_spread(cands, consec, env, rid):
    """Rest a provider after ONE failure (never retry a failed provider in the same
    request's fresh pass); also skip providers that already failed once this trial
    in the fresh pass, trying them only as a last resort."""
    fresh = [c for c in cands if consec[c[0]] == 0]
    tried = [c for c in cands if consec[c[0]] > 0]
    return fresh + tried


def strat_spread2(cands, consec, env, rid):
    """Like spread, but a provider that failed once is skipped entirely unless
    no fresh candidate exists at all (same as spread, but rest threshold 1
    applies across the whole trial: once failed, deprioritise permanently)."""
    return strat_spread(cands, consec, env, rid)


if __name__ == "__main__":
    strats = {"r1": strat_r1, "spread": strat_spread, "spread2": strat_spread2}
    names = sys.argv[1:] or ["r1"]
    trials = [0, 1, 2]
    for name in names:
        fn = strats[name]
        print(f"### strategy {name}")
        for t in trials:
            r = run_trial(t, fn)
            print(f"  trial-{t}: calls={r['calls']} quality={r['quality']} served={r['served']}/20")
            print(f"    used={ {k: v for k, v in r['used'].items() if v} }")
            print(f"    fails={ {k: v for k, v in r['fails'].items() if v} }")
