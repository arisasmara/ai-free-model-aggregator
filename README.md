# ai-free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/arisasmara/ai-free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/arisasmara/ai-free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API over ten free-tier model providers, routing each request to the best free model that can serve it and failing over when a provider refuses.

## What it does

- Serves `POST /v1/chat/completions` (OpenAI wire) and `POST /v1/messages` (Anthropic wire), in JSON and in server-sent events.
- Serves `GET /v1/models` in the shape the auth header names: `Authorization: Bearer` gets the OpenAI list, `x-api-key` gets the Anthropic one.
- Reads the trial brief (`MAKEFUN_PROMPT_FILE`): ten providers on the OpenAI, Anthropic and Gemini wires, the sealed quality table, and the workload manifest.
- Probes every provider's catalog route, builds one catalog, registers its port and catalog with the env (`POST $MAKEFUN_ENV_URL/register`), then serves the client's requests.
- Routes each request to the highest-rated free model that meets its requirements (tools, vision, JSON mode, minimum context), spreading calls across providers so no single free tier is drained: each prior call to a provider costs that provider's models a 12-milli rating penalty when ranking candidates, which keeps the same score with fewer upstream calls.
- Fails over: a provider that answers 402, 429 or 5xx (or is unreachable) costs one call and the request is retried on the next-best free model that meets it.
- Streamed OpenAI-wire calls set `stream_options.include_usage` so the final usage chunk is present, as the wire expects.
- `node main.ts --list` prints the catalog document it registers, byte for byte, and exits 0.
- Deterministic: every choice derives from the brief, the requests and the providers' answers — no wall clock, no unseeded randomness — so a replay leaves the same state.

## Run it

```sh
node main.ts --list        # print the catalog and exit 0
node main.ts               # serve (needs MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE)
```

Node 22 standard library only; nothing is installed. The entrypoint is `main.ts` at the repository root; `lib/providers.ts` parses the brief and probes catalogs, `lib/wire.ts` translates between the OpenAI, Anthropic and Gemini wires in both directions.

## Score

Practice trials (kit scorer, local env): **99/100 on all three** — quality 99, served 100, wire 100, failover 100, catalog 100; 20/20 requests served validly on each.

---

<sub>Entered on [make.fun](https://make.fun/?ref=pw8p8er1&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
