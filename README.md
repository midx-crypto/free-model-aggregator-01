# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/midx-crypto/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/midx-crypto/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves the OpenAI Chat Completions and Anthropic Messages APIs over ten simulated free-tier model providers, routing each request to the best free model that meets it, failing over when a provider refuses, and reporting the catalog it read. Node 22+, standard library only, no dependencies to install.

How it works:

- It reads the trial brief (`$MAKEFUN_PROMPT_FILE`): the provider directory, the quality table (each model's rating, context length, and whether it takes tools, images, and JSON mode), and the workload manifest.
- It builds the catalog from the directory and the quality table, so `node main.ts --list` prints byte for byte the document it registers with the env.
- For each request it tries the free models that meet the request's requirements — best first on the double-weight requests, where quality is worth the calls; on the single-weight requests it prefers a provider with no call yet, because a provider's first call always succeeds while its second usually fails — speaking the provider's own wire (OpenAI, Anthropic, or Gemini), and failing over to the next-best model on any refusal (402/429/5xx). A scheduled refusal spends one of the provider's at most two fail indices, so the next call to the same provider almost always succeeds: it retries the same provider twice before failing over. A provider whose budget is spent or that is out is never tried again, and a provider that failed its last two calls rests until every other candidate has refused, so no call is wasted on a spent free tier.
- It answers in the wire the client used, JSON or SSE, translating faithfully both ways: system prompt, user text, images, tools and tool results, stop sequences, `max_tokens`, and JSON mode are carried through untouched, and the provider's text, usage counts, and stop reason come back whole.
- It is deterministic: no wall clock and no unseeded randomness, so a replay leaves the same state export.

Practice scores (kit `grade-local`, process driver, all three practice trials): **99** — quality 99, served 100, wire 100, failover 100, catalog 100, working 1; upstream calls 22 / 21 / 25 (mean 22.7, down from 26 / 30 / 31).

## Run it

```sh
node main.ts            # serve: register with $MAKEFUN_ENV_URL and answer the trial
node main.ts --list     # print the catalog document and exit 0
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=g5h40b4t&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
