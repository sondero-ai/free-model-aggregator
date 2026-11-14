// The trial's call plan: twenty slots, one per request, each a provider and its best model for
// that request. The counts are one call for every provider, plus extra calls for the providers
// whose free-tier budgets are largest (Groq and Kilo) and two extra Gemini calls for the three
// vision requests. This keeps every provider's first call (always safe) and spends the extra
// calls where a trial budget is most likely to hold them.
//
// The assignment is a minimum-cost matching between the twenty requests and the twenty slots:
// the cost of a slot is the weighted rating the request loses compared to its best legal model.
// It is solved once from the brief, deterministically, with the Hungarian algorithm.

import { meets, type Brief, type RatingRow } from './brief.ts';

export interface PlanRoute {
  provider: string;
  model: string;
  rating: number;
}

export interface Plan {
  routes: Map<string, PlanRoute>;
  counts: Map<string, number>;
  loss: number;
  bound: number;
}

/** The per-provider call counts. Every provider not named gets one call. */
const TARGET_COUNTS: Record<string, number> = {
  gemini: 3,
  groq: 5,
  kilol: 5,
  pollinations: 1,
  cerebras: 1,
  zai: 1,
  openrouter: 1,
  sambanova: 1,
  cloudflare: 1,
  ovh: 1,
};

/** Where extra calls go when a provider from the target table is absent (largest budget first). */
const EXTRA_ORDER = ['kilol', 'groq', 'pollinations', 'gemini', 'cloudflare', 'ovh', 'cerebras', 'sambanova', 'openrouter', 'zai'];

function hungarian(cost: number[][]): number[] {
  const n = cost.length;
  if (n === 0) return [];
  const INF = 1e15;
  const u = new Array<number>(n + 1).fill(0);
  const v = new Array<number>(n + 1).fill(0);
  const p = new Array<number>(n + 1).fill(0);
  const way = new Array<number>(n + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array<number>(n + 1).fill(INF);
    const used = new Array<boolean>(n + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = INF;
      let j1 = 0;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }
  const answer = new Array<number>(n);
  for (let j = 1; j <= n; j++) answer[p[j] - 1] = j - 1;
  return answer;
}

/** Whether a model may serve a request; an OpenAI JSON-mode request cannot use the anthropic wire. */
function legal(row: RatingRow, wire: string, requires: Parameters<typeof meets>[1], jsonMode: boolean): boolean {
  if (!row.free || !meets(row, requires)) return false;
  if (jsonMode && wire === 'anthropic') return false;
  return true;
}

/** Builds the plan for the brief's manifest; an empty plan when the brief is incomplete. */
export function buildPlan(brief: Brief): Plan {
  const routes = new Map<string, PlanRoute>();
  const counts = new Map<string, number>();
  const providers = new Map(brief.providers.map((p, order) => [p.id, { wire: p.wire, order }]));
  const requests = brief.manifest;
  const bound = requests.reduce((n, r) => n + r.weight * bestRating(brief, providers, r.requires, r.api === 'openai' && r.requires.json === true), 0);
  if (requests.length === 0) return { routes, counts, loss: 0, bound: 0 };

  for (const p of brief.providers) counts.set(p.id, TARGET_COUNTS[p.id] ?? 1);
  let total = [...counts.values()].reduce((a, b) => a + b, 0);
  // A brief with fewer providers than the sealed table: fill the remaining slots from the
  // largest-budget providers, then trim any excess from the smallest ones.
  for (const id of EXTRA_ORDER) {
    if (total >= requests.length) break;
    if (!counts.has(id)) continue;
    counts.set(id, (counts.get(id) ?? 0) + 1);
    total += 1;
  }
  while (total > requests.length) {
    let pick: string | null = null;
    for (const id of [...EXTRA_ORDER].reverse()) {
      if ((counts.get(id) ?? 0) > 1) {
        pick = id;
        break;
      }
    }
    if (pick === null) break;
    counts.set(pick, (counts.get(pick) ?? 0) - 1);
    total -= 1;
  }
  if (total !== requests.length) return { routes, counts, loss: 0, bound };

  const slots: string[] = [];
  for (const p of brief.providers) for (let i = 0; i < (counts.get(p.id) ?? 0); i++) slots.push(p.id);

  const bestFor = new Map<string, number>();
  for (const r of requests) {
    const jsonMode = r.api === 'openai' && r.requires.json === true;
    bestFor.set(r.id, bestRating(brief, providers, r.requires, jsonMode));
  }

  const modelOn = (providerId: string, requires: Parameters<typeof meets>[1], jsonMode: boolean): RatingRow | null => {
    const provider = providers.get(providerId);
    if (provider === undefined) return null;
    let best: RatingRow | null = null;
    for (const row of brief.ratings) {
      if (row.provider !== providerId || !legal(row, provider.wire, requires, jsonMode)) continue;
      if (best === null || row.rating_milli > best.rating_milli) best = row;
    }
    return best;
  };

  const INF = 1e9;
  const cost = requests.map((r) => {
    const jsonMode = r.api === 'openai' && r.requires.json === true;
    return slots.map((providerId) => {
      const row = modelOn(providerId, r.requires, jsonMode);
      return row === null ? INF : r.weight * (bestFor.get(r.id)! - row.rating_milli);
    });
  });
  const assignment = hungarian(cost);
  let loss = 0;
  for (let i = 0; i < requests.length; i++) {
    const r = requests[i];
    const providerId = slots[assignment[i]];
    const jsonMode = r.api === 'openai' && r.requires.json === true;
    const row = modelOn(providerId, r.requires, jsonMode);
    if (row === null || cost[i][assignment[i]] >= INF) continue;
    routes.set(r.id, { provider: providerId, model: row.model, rating: row.rating_milli });
    loss += cost[i][assignment[i]];
  }
  return { routes, counts, loss, bound };
}

function bestRating(brief: Brief, providers: Map<string, { wire: string; order: number }>, requires: Parameters<typeof meets>[1], jsonMode: boolean): number {
  let best = 0;
  for (const row of brief.ratings) {
    const provider = providers.get(row.provider);
    if (provider === undefined) continue;
    if (!legal(row, provider.wire, requires, jsonMode)) continue;
    if (row.rating_milli > best) best = row.rating_milli;
  }
  return best;
}
