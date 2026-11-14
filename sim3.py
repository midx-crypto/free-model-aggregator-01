#!/usr/bin/env python3
"""Strategy evaluation for free-model-aggregator-01 across practice + synthetic
instances drawn with the generator's own distributions (draw.mjs):
  budget ~ U[calls_min, calls_max]; fail count ~ U{1,2}; fail indices ~ U[2, budget-1]
  (distinct); one designated provider forced to include 2; one provider gets
  outage_from ~ U[2, budget-1].
The workload and ratings are fixed across instances. Strategies are RUNTIME
rules: they see only the brief + the outcomes (incl. failure message kind)
of calls they made.
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
IDS = list(CALLS_MINMAX)
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


def draw_instance(rng):
    provs = {}
    for pid in IDS:
        lo, hi = CALLS_MINMAX[pid]
        budget = rng.randint(lo, hi)
        count = rng.randint(1, 2)
        fails = set(rng.sample(range(2, budget), count))
        provs[pid] = {"budget": budget, "fail": fails, "outage": None}
    designated = rng.choice(IDS)
    provs[designated]["fail"].add(2)
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


def run_instance(provs, order_fn):
    st = State(provs)
    served = {}
    calls = 0
    for rid, api, weight, req in WORKLOAD:
        cands = candidates(api, req, req.get("json", False))
        order = order_fn(cands, st, rid, weight)
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
    served_n = sum(1 for v in served.values() if v is not None)
    failover_req = 0  # not tracked here; served==20 implies failover 100 if any
    return {"calls": calls, "quality": quality, "served": served_n}


def score_of(r):
    if r["served"] < 20:
        return None
    return round((60 * r["quality"] + 15 * 100 + 10 * 100 + 10 * 100 + 5 * 100) / 100)


# --- strategies ---

def s_r1_msg(cands, st, rid, weight):
    fresh = [c for c in cands if st.consec[c[0]] < 2]
    rested = [c for c in cands if st.consec[c[0]] >= 2]
    return fresh + rested


def make_tiered(cap, rest=2, guard=None):
    """Prefer providers with fewer calls used (tier = min(calls, cap)).
    guard: if set, a tier-0 candidate is only preferred when its rating is
    within `guard` of the best candidate's rating; otherwise rating order wins
    among the top tier. rest: consecutive failures before a provider is
    demoted to the last tier."""
    def fn(cands, st, rid, weight):
        best = cands[0][1][2] if cands else 0
        def tier(c):
            prov = c[0]
            if st.consec[prov] >= rest:
                return 99
            t = min(st.calls[prov], cap)
            if guard is not None and t == 0 and c[1][2] < best - guard:
                return 1  # not fresh enough to jump the queue on rating
            return t
        return sorted(cands, key=lambda c: (tier(c), -c[1][2], c[0]))
    return fn


def make_pen(pen, rest=2):
    def fn(cands, st, rid, weight):
        def key(c):
            prov, row = c
            if st.consec[prov] >= rest:
                return (1, 0)
            return (0, -(row[2] - pen * st.calls[prov]))
        return sorted(cands, key=key)
    return fn


STRATS = {
    "r1_msg": s_r1_msg,
    "cap1": make_tiered(1),
    "cap1_g10": make_tiered(1, guard=10),
    "cap1_g20": make_tiered(1, guard=20),
    "cap1_g30": make_tiered(1, guard=30),
    "cap2": make_tiered(2),
    "cap2_g20": make_tiered(2, guard=20),
    "cap3": make_tiered(3),
    "pen15": make_pen(15),
    "pen25": make_pen(25),
    "pen40": make_pen(40),
}


def evaluate(name, fn, n_syn=300, seed=12345):
    rng = random.Random(seed)
    prac = [run_instance(load_practice(t), fn) for t in [0, 1, 2]]
    syn = [run_instance(draw_instance(rng), fn) for _ in range(n_syn)]
    def summ(rs):
        calls = [r["calls"] for r in rs]
        qs = [r["quality"] for r in rs]
        sc = [score_of(r) for r in rs]
        return (f"mean_calls={sum(calls)/len(calls):.1f} "
                f"min={min(calls)} p25={sorted(calls)[len(calls)//4]} "
                f"med={sorted(calls)[len(calls)//2]} "
                f"q<98={sum(1 for q in qs if q < 98)}/{len(qs)} "
                f"score99={sum(1 for s in sc if s == 99)}/{len(sc)}")
    p0 = prac[0]
    print(f"{name:10s} prac t0={p0['calls']}c q{p0['quality']} | "
          f"t1={prac[1]['calls']}c q{prac[1]['quality']} | "
          f"t2={prac[2]['calls']}c q{prac[2]['quality']} | syn: {summ(syn)}")


if __name__ == "__main__":
    names = sys.argv[1:] or list(STRATS)
    for name in names:
        evaluate(name, STRATS[name])


def make_cap1_loss(cap_loss, cap=1):
    """cap1 ordering, but once cumulative weighted loss reaches cap_loss,
    revert to r1_msg ordering (best first) for the rest of the trial."""
    def fn(cands, st, rid, weight):
        # track loss in st via a side attribute
        loss = getattr(st, "loss", 0)
        if loss >= cap_loss:
            fresh = [c for c in cands if st.consec[c[0]] < 2]
            rested = [c for c in cands if st.consec[c[0]] >= 2]
            return fresh + rested
        def tier(c):
            prov = c[0]
            if st.consec[prov] >= 2:
                return 99
            return min(st.calls[prov], cap)
        return sorted(cands, key=lambda c: (tier(c), -c[1][2], c[0]))
    return fn


def make_cap1_w1(cap=1):
    """cap1 ordering only for weight-1 requests; weight-2 use r1_msg."""
    def fn(cands, st, rid, weight):
        if weight == 2:
            fresh = [c for c in cands if st.consec[c[0]] < 2]
            rested = [c for c in cands if st.consec[c[0]] >= 2]
            return fresh + rested
        def tier(c):
            prov = c[0]
            if st.consec[prov] >= 2:
                return 99
            return min(st.calls[prov], cap)
        return sorted(cands, key=lambda c: (tier(c), -c[1][2], c[0]))
    return fn


# patch run_instance to track loss for cap1_loss strategies
_orig_run = run_instance
def run_instance2(provs, order_fn):
    st = State(provs)
    served = {}
    calls = 0
    st.loss = 0
    for rid, api, weight, req in WORKLOAD:
        cands = candidates(api, req, req.get("json", False))
        order = order_fn(cands, st, rid, weight)
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
        if got is not None:
            st.loss += weight * max(0, BEST[rid] - got[2])
    bound = sum(w * BEST[rid] for rid, _, w, _ in WORKLOAD)
    earned = sum(w * min(served[rid][2], BEST[rid]) for rid, _, w, _ in WORKLOAD
                 if served[rid] is not None)
    quality = (100 * earned) // bound
    served_n = sum(1 for v in served.values() if v is not None)
    return {"calls": calls, "quality": quality, "served": served_n}


def evaluate2(name, fn, n_syn=300, seed=12345):
    rng = random.Random(seed)
    prac = [run_instance2(load_practice(t), fn) for t in [0, 1, 2]]
    syn = [run_instance2(draw_instance(rng), fn) for _ in range(n_syn)]
    def summ(rs):
        calls = [r["calls"] for r in rs]
        qs = [r["quality"] for r in rs]
        sc = [score_of(r) for r in rs]
        return (f"mean_calls={sum(calls)/len(calls):.1f} "
                f"min={min(calls)} p25={sorted(calls)[len(calls)//4]} "
                f"med={sorted(calls)[len(calls)//2]} "
                f"q<=97={sum(1 for q in qs if q <= 97)}/{len(qs)} "
                f"score99={sum(1 for s in sc if s == 99)}/{len(sc)}")
    p0 = prac[0]
    print(f"{name:14s} prac t0={p0['calls']}c q{p0['quality']} | "
          f"t1={prac[1]['calls']}c q{prac[1]['quality']} | "
          f"t2={prac[2]['calls']}c q{prac[2]['quality']} | syn: {summ(syn)}")


STRATS2 = {
    "cap1_loss200": make_cap1_loss(200),
    "cap1_loss400": make_cap1_loss(400),
    "cap1_loss600": make_cap1_loss(600),
    "cap1_w1": make_cap1_w1(),
    "cap1": make_tiered(1),
    "r1_msg": s_r1_msg,
}


def run_instance3(provs, order_fn, retry=1):
    """retry: extra attempts on the SAME provider after a scheduled failure
    before moving to the next candidate."""
    st = State(provs)
    served = {}
    calls = 0
    for rid, api, weight, req in WORKLOAD:
        cands = candidates(api, req, req.get("json", False))
        order = order_fn(cands, st, rid, weight)
        got = None
        for prov, row in order:
            if st.dead[prov]:
                continue
            attempts = 0
            while attempts <= retry:
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
                        break
                    attempts += 1  # scheduled: retry same provider
            if got is not None:
                break
        served[rid] = got
    bound = sum(w * BEST[rid] for rid, _, w, _ in WORKLOAD)
    earned = sum(w * min(served[rid][2], BEST[rid]) for rid, _, w, _ in WORKLOAD
                 if served[rid] is not None)
    quality = (100 * earned) // bound
    served_n = sum(1 for v in served.values() if v is not None)
    # score: served column is 100*served/20; failover 100 if served==20
    s = None
    if served_n == 20:
        s = round((60 * quality + 15 * 100 + 10 * 100 + 10 * 100 + 5 * 100) / 100)
    return {"calls": calls, "quality": quality, "served": served_n, "score": s}


def evaluate3(name, fn, retry, n_syn=300, seed=12345):
    rng = random.Random(seed)
    prac = [run_instance3(load_practice(t), fn, retry) for t in [0, 1, 2]]
    syn = [run_instance3(draw_instance(rng), fn, retry) for _ in range(n_syn)]
    def summ(rs):
        calls = [r["calls"] for r in rs]
        qs = [r["quality"] for r in rs]
        sc = [r["score"] for r in rs]
        unserved = sum(1 for r in rs if r["served"] < 20)
        return (f"mean_calls={sum(calls)/len(calls):.1f} "
                f"min={min(calls)} p25={sorted(calls)[len(calls)//4]} "
                f"med={sorted(calls)[len(calls)//2]} "
                f"unserved={unserved}/{len(rs)} "
                f"q<=97={sum(1 for q in qs if q <= 97)}/{len(qs)} "
                f"score99={sum(1 for s in sc if s == 99)}/{len(sc)}")
    p0 = prac[0]
    print(f"{name:12s} retry={retry} prac t0={p0['calls']}c q{p0['quality']} s{p0['score']} | "
          f"t1={prac[1]['calls']}c q{prac[1]['quality']} s{prac[1]['score']} | "
          f"t2={prac[2]['calls']}c q{prac[2]['quality']} s{prac[2]['score']} | syn: {summ(syn)}")
