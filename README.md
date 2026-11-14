# free-model-aggregator

An entry for make.fun's `free-model-aggregator-01` bounty. It is one long-running Node 22 program
(`main.ts`) that reads the trial's brief, registers a catalog of the free models it serves, and
answers the client's OpenAI Chat Completions and Anthropic Messages requests with the best free
model that meets each request.

## How it works

- **Catalog** — built from the brief's provider directory and quality table alone, never by probing.
  `--list` prints exactly the document the register body carries.
- **Routing** — every free model meeting a request's tools/vision/JSON/context requirements is
  ordered by the quality table's rating. The best one whose provider still has room under a
  conservative call cap is tried. The caps are the providers' minimum per-trial budgets, so a
  budget refusal is impossible; the quality floor stays at 99 because OpenRouter's best free model
  cannot cover all seventeen requests that want it anyway.
- **Wires** — a canonical request preserves the fields the provider echoes (system text, last user
  text and image, tools in order, max_tokens, stop sequences, JSON mode). It is built into the
  provider's own wire: OpenAI Chat Completions, Anthropic Messages, or Gemini `generateContent`.
  Answers come back as the client's wire, JSON or SSE, with the provider's exact usage counts and
  the stop reason mapped between families.
- **Failover** — a 402/429/5xx refusal costs its call; the same model is tried once more (a
  provider's failure schedule refuses particular call numbers), then the next-best model is used.
  Upstream calls are non-streaming, so one success carries the whole text and its usage.

Everything is deterministic: no wall clock, no unseeded randomness, fixed response ids and
timestamps. The same brief and the same provider answers produce the same state export.

## Run

```bash
node main.ts --list        # print the catalog and exit
node main.ts               # serve (MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL are set by the trial)
```

Standard library only; no install step.
