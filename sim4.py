#!/usr/bin/env python3
"""Round-3 strategy sweep for free-model-aggregator-01.

Exact generator replication (draw.mjs): budget U[calls_min, calls_max];
fail count U{1,2}; fail indices distinct U[2, budget-1]; ONE designated
provider is forced to include fail index 2 (by REPLACING one index, count
unchanged -- sim3.py's .add() could make it 3, which is wrong); ONE provider
gets outage_from U[2, budget-1].

Strategies are runtime rules: they see only the brief + outcomes of calls
they made (failure kind included), plus priors from the generator's fixed
per-provider budget ranges (public kit data, same for every draw).

Key structural facts used:
- call #1 of a provider ALWAYS succeeds (fail indices >= 2, outage >= 2,
  budget >= 4); call #2+ may hit a scheduled fail / outage / budget.
- a provider has at most 2 scheduled fail indices: after 2 OBSERVED
  scheduled failures its fail schedule is exhausted -> its next call can
  only fail on budget/outage ("safe").
- after 1 observed scheduled failure at index n, P(next index fails) is
  small (~0.5/(budget-3)): retrying the same provider is usually right.
- expected fail prob of a repeat call grows as budget shrinks: big-budget
  providers (kilol 12-16, pollinations 10-14, openrouter 10-12, groq 9-12)
  are cheap repeat targets; small-budget ones (zai/sambanova 4-5,
  cerebras/ovh 5-6) are expensive.
- score 99 needs quality >= 98 -> total weighted rating loss <= 764
  (bound 38248, earned >= 37484). Loss budget is a real constraint.
"""
import json, random, sys

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
CALLS_MINMAX = {"groq": (9, 12), "cerebras": (5, 6), "gemini": (6, 8), "zai": (4, 5),
                "openrouter": (10, 12), "sambanova": (4, 5), "pollinations": (10, 14),
                "cloudflare": (7, 8), "ovh": (5, 6), "kilol": (12, 16)}
MIDBUDGET = {p: (lo + hi) / 2 for p, (lo, hi) in CALLS_MINMAX.items()}
IDS = list(CALLS_MINMAX)
BEST = {"r01": 1458, "r02": 1458, "r03": 1475, "r04": 1475, "r05": 1475, "r06": 1458,
        "r07": 1475, "r08": 1475, "r09": 1475, "r10": 1475, "r11": 1475, "r12": 1475,
        "r13": 1475, "r14": 1475, "r15": 1475, "r16": 1475, "r17": 1475, "r18": 1475,
        "r19": 1475, "r20": 1475}
BOUND = sum(w * BEST[rid] for rid, _, w, _ in
            [("r01", "openai", 2, {}), ("r02", "openai", 2, {}), ("r03", "openai", 2, {}),
             ("r04", "anthropic", 2, {}), ("r05", "anthropic", 2, {}), ("r06", "openai", 2, {})] +
            [(f"r{i:02d}", "openai", 1, {}) for i in range(7, 21)])
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
LOSS_CAP_HARD = 764  # quality >= 98 (score 99) needs weighted loss <= 764


def draw_instance(rng):
    """Exact replication of draw.mjs drawInstance."""
    provs = {}
    for pid in IDS:
        lo, hi = CALLS_MINMAX[pid]
        budget = rng.randint(lo, hi)
        count = rng.randint(1, 2)
        fails = set(rng.sample(range(2, budget), count))
        provs[pid] = {"budget": budget, "fail": fails, "outage": None}
    designated = rng.choice(IDS)
    if 2 not in provs[designated]["fail"]:
        # generator REPLACES one index (keeps count), it does not add.
        f = sorted(provs[designated]["fail"])
        provs[designated]["fail"] = set(sorted([2] + f[1:]))
    down = rng.choice(IDS)
    b = provs[down]["budget"]
    provs[down]["outage"] = rng.randint(2, b - 1)
    return provs


def load_practice(t):
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
    def __init__(self, provs):
        self.provs = provs
        self.calls = {p: 0 for p in provs}      # counted calls (incl. failures)
        self.consec = {p: 0 for p in provs}     # consecutive failed calls
        self.dead = {p: False for p in provs}
        self.sched = {p: 0 for p in provs}      # observed scheduled failures
        self.loss = 0                          # cumulative weighted rating loss

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


def risk_next(prov, st):
    """Approx P(next call to prov fails on its schedule), from priors only.
    sched>=2 -> fail schedule exhausted (only budget/outage left)."""
    if st.sched[prov] >= 2:
        return 0.02
    n = st.calls[prov] + 1  # index of the next call
    if n <= 1:
        return 0.0
    if st.sched[prov] == 1:
        # one fail index spent; the other (if count==2) is uniform over the
        # remaining positions: P ~ 0.5 / (positions left after n)
        return 0.5 / max(2.0, MIDBUDGET[prov] - n)
    return 1.5 / max(2.0, MIDBUDGET[prov] - n + 1)


# --- strategy spec ---
# fresh: 'w1' (r2: fresh preference only for weight-1), 'all', 'none'
# fresh_guard: fresh candidate only jumps the queue if its rating is within
#   this of the best candidate's rating (None = any fresh first)
# retry: extra attempts on the SAME provider after a scheduled failure
# rest: consecutive failures before demotion (999 = no rest)
# safe2: providers with sched>=2 are promoted (schedule exhausted)
# repeat_key: 'rating' | 'risk' | 'risk_w1' (risk for w1, rating for w2)
#   | 'cost' (expected-extra-calls + lam * weighted loss, repeats only)
#   | 'cost_w1'
# lam: calls per unit of weighted rating loss in 'cost' modes
# guard: once cumulative weighted loss reaches this, ordering collapses to
#   pure best-rating (fresh preference and risk ordering off) for the rest of
#   the trial -- insurance for the quality tail. None = off.
# window: repeat (non-fresh) candidates with rating below best-window are
#   demoted behind in-window repeats. None = off.
DEFAULT = dict(fresh="w1", fresh_guard=None, retry=2, rest=2, safe2=False,
               repeat_key="rating", lam=0.001, loss_cap=None, cap_scope="repeat",
               guard=None, window=None)


def make_order(spec):
    def order(cands, st, rid, weight):
        best_rating = cands[0][1][2] if cands else 0
        fresh_ok = spec["fresh"] == "all" or (spec["fresh"] == "w1" and weight == 1)

        def breaching(c):
            if spec["loss_cap"] is None:
                return 0
            prov, row = c
            is_repeat = st.calls[prov] > 0
            if spec["cap_scope"] == "repeat" and not is_repeat:
                return 0
            return 1 if st.loss + weight * (BEST[rid] - row[2]) > spec["loss_cap"] else 0

        def key(c):
            prov, row = c
            # Guard mode: loss already high -> pure best-rating ordering.
            if spec["guard"] is not None and st.loss >= spec["guard"]:
                return (0, 0, 0, 0, 0, 0.0, -row[2], prov, row[0])
            # rest demotion (safe2 providers are exempt: their fails are spent)
            rest = 0
            if st.consec[prov] >= spec["rest"]:
                rest = 0 if (spec["safe2"] and st.sched[prov] >= 2) else 1
            fresh = 0
            if fresh_ok and st.calls[prov] == 0:
                if spec["fresh_guard"] is None or row[2] >= best_rating - spec["fresh_guard"]:
                    fresh = 0
                else:
                    fresh = 1
            else:
                fresh = 1
            safe = 0 if (spec["safe2"] and st.sched[prov] >= 2) else 1
            # rating window for repeat (non-fresh) candidates
            win = 0
            if spec["window"] is not None and st.calls[prov] > 0:
                win = 0 if row[2] >= best_rating - spec["window"] else 1
            rk = spec["repeat_key"]
            use_risk = rk in ("risk", "cost") or (rk in ("risk_w1", "cost_w1") and weight == 1)
            if rk.startswith("cost") and use_risk:
                p = risk_next(prov, st)
                risk = p / max(0.05, 1 - p) + spec["lam"] * weight * (BEST[rid] - row[2])
            elif use_risk:
                risk = risk_next(prov, st)
            else:
                risk = 0.0
            return (breaching(c), rest, fresh, safe, win, risk, -row[2], prov, row[0])
        return sorted(cands, key=key)
    return order


def run_instance(provs, spec, trace=None):
    order = make_order(spec)
    st = State(provs)
    served = {}
    calls = 0
    for rid, api, weight, req in WORKLOAD:
        cands = candidates(api, req, req.get("json", False))
        ordered = order(cands, st, rid, weight)
        got = None
        for prov, row in ordered:
            if st.dead[prov]:
                continue
            attempts = 0
            while attempts <= spec["retry"]:
                calls += 1
                ok, kind = st.call(prov)
                if ok:
                    st.consec[prov] = 0
                    st.loss += weight * (BEST[rid] - row[2])
                    got = row
                    if trace is not None:
                        trace.append((rid, prov, row[0], st.calls[prov], "ok",
                                      calls, st.loss))
                    break
                st.consec[prov] += 1
                if kind == "scheduled":
                    st.sched[prov] += 1
                if kind in ("budget", "outage"):
                    st.dead[prov] = True
                    if trace is not None:
                        trace.append((rid, prov, row[0], st.calls[prov], kind,
                                      calls, st.loss))
                    break
                if trace is not None:
                    trace.append((rid, prov, row[0], st.calls[prov], kind,
                                  calls, st.loss))
                if kind != "scheduled":
                    break
                attempts += 1
            if got is not None:
                break
        served[rid] = got
    earned = sum(w * min(served[rid][2], BEST[rid]) for rid, _, w, _ in WORKLOAD
                 if served[rid] is not None)
    quality = (100 * earned) // BOUND
    served_n = sum(1 for v in served.values() if v is not None)
    score = None
    if served_n == 20:
        score = round((60 * quality + 15 * 100 + 10 * 100 + 10 * 100 + 5 * 100) / 100)
    return {"calls": calls, "quality": quality, "served": served_n,
            "score": score, "loss": st.loss}


def evaluate(name, spec, n_syn=1000, seed=12345):
    rng = random.Random(seed)
    prac = [run_instance(load_practice(t), spec) for t in [0, 1, 2]]
    syn = [run_instance(draw_instance(rng), spec) for _ in range(n_syn)]
    syn_calls = [r["calls"] for r in syn]
    syn_sorted = sorted(syn_calls)
    sc99 = sum(1 for r in syn if r["score"] == 99)
    unserved = sum(1 for r in syn if r["served"] < 20)
    q98 = sum(1 for r in syn if r["quality"] < 98)
    mean_loss = sum(r["loss"] for r in syn) / len(syn)
    losses = sorted(r["loss"] for r in syn)
    print(f"{name:28s} prac {prac[0]['calls']}c/q{prac[0]['quality']} "
          f"{prac[1]['calls']}c/q{prac[1]['quality']} "
          f"{prac[2]['calls']}c/q{prac[2]['quality']} | "
          f"syn mean={sum(syn_calls)/len(syn_calls):.2f} "
          f"med={syn_sorted[len(syn_sorted)//2]} "
          f"p10={syn_sorted[len(syn_sorted)//10]} "
          f"p90={syn_sorted[9*len(syn_sorted)//10]} "
          f"max={max(syn_calls)} unserved={unserved} "
          f"q<98={q98} score99={sc99}/{len(syn)} "
          f"loss mean={mean_loss:.0f} p95={losses[95*len(losses)//100]} "
          f"p99={losses[99*len(losses)//100]} max={losses[-1]}")
    return syn


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep2"
    r2 = dict(DEFAULT)
    if mode == "calib":
        print("== calibration (r2 faithful) ==")
        evaluate("r2", r2)
    elif mode == "sweep3":
        print("== risk_w1 + guard (loss mode-switch) ==")
        for guard in (300, 400, 500):
            evaluate(f"r2 risk_w1 guard={guard}",
                     dict(r2, repeat_key="risk_w1", guard=guard))
        print("== risk_w1 + rating window (repeats) ==")
        for win in (60, 85):
            evaluate(f"r2 risk_w1 win={win}",
                     dict(r2, repeat_key="risk_w1", window=win))
            evaluate(f"r2 risk_w1 win={win} guard=400",
                     dict(r2, repeat_key="risk_w1", window=win, guard=400))
        print("== cost + guard ==")
        for lam, guard in ((0.002, 400), (0.002, 500), (0.001, 400), (0.002, 300)):
            evaluate(f"r2 cost lam={lam} guard={guard}",
                     dict(r2, repeat_key="cost", lam=lam, guard=guard))
        print("== cost_w1 + guard ==")
        for lam, guard in ((0.002, 400), (0.001, 400)):
            evaluate(f"r2 cost_w1 lam={lam} guard={guard}",
                     dict(r2, repeat_key="cost_w1", lam=lam, guard=guard))
