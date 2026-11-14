# free-model-aggregator

An entry for make.fun's `free-model-aggregator-01` bounty. It is one long-running Node 22 program
(`main.ts`) that reads the trial's brief, registers a catalog of the free models it serves, and
answers the client's OpenAI Chat Completions and Anthropic Messages requests with the best free
model that meets each request.

## How it works

- **Catalog** — built from the brief's provider directory and quality table alone, never by probing.
  `--list` prints exactly the document the register body carries.
- **Routing plan** — the twenty requests are matched to twenty call slots by a minimum-cost
  assignment (Hungarian) over the quality table. Every provider gets one call — its first call is
  always safe on the free tier — with extra calls for the providers whose minimum budgets are
  largest (Groq, Kilo) and three Gemini calls for the three vision requests. The plan accepts
  quality 98: that still rounds to a score of 99, and it keeps the trial at the 20-call floor.
- **Wires** — a canonical request preserves the fields the provider echoes (system text, last user
  text and image, tools in order, max_tokens, stop sequences, JSON mode). It is built into the
  provider's own wire: OpenAI Chat Completions, Anthropic Messages, or Gemini `generateContent`.
  Answers come back as the client's wire, JSON or SSE, with the provider's exact usage counts and
  the stop reason mapped between families.
- **Failover** — a 402/429/5xx refusal costs its call. The same model is tried once more (a
  provider's failure schedule refuses particular call numbers, and the next call often succeeds);
  if that also refuses, the provider is dropped for the rest of the trial and the request moves to
  the next candidate: the plan's unused slots first, then the providers' spare budgets. Upstream
  calls are non-streaming, so one success carries the whole text and its usage.

Everything is deterministic: no wall clock, no unseeded randomness, fixed response ids and
timestamps. The same brief and the same provider answers produce the same state export.

## Run

```bash
node main.ts --list        # print the catalog and exit
node main.ts               # serve (MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL are set by the trial)
```

Standard library only; no install step.
