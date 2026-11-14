// The brief's three documents: the provider directory, the quality table and the workload
// manifest. Each is one JSON object carrying a schema; the prose around them is ignored.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type Auth = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
export type Json = Record<string, unknown>;

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: Auth;
  base: string;
  key_env: string;
  key: string;
}

export interface RatingRow {
  provider: string;
  model: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface ManifestRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

export interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  manifest: ManifestRow[];
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface CatalogReport {
  providers: CatalogProvider[];
}

export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** The index of the `}` closing the object that opens at `start`, or -1. Strings are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') {
      inString = true;
    } else if (ch === '{') {
      depth += 1;
    } else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not a whole document at this brace: keep scanning.
    }
  }
  return out;
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base = stringOf(v.path) !== '' ? stringOf(v.path) : stringOf(v.base_url);
  return {
    id,
    name: stringOf(v.name) !== '' ? stringOf(v.name) : id,
    wire,
    auth,
    base,
    key_env: stringOf(v.key_env),
    key: stringOf(v.key),
  };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const provider = stringOf(v.provider);
  const model = stringOf(v.model);
  if (provider === '' || model === '') return null;
  const rating = numberOrNull(v.rating_milli);
  return {
    provider,
    model,
    rating_milli: rating === null ? 0 : rating,
    context_length: numberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function manifestOf(v: unknown): ManifestRow | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  const minContext = numberOrNull(requires.min_context);
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(minContext !== null ? { min_context: minContext } : {}),
    },
  };
}

/** Reads the brief at `path`; throws when it holds no provider directory. */
export function readBrief(path: string): Brief {
  if (path === '') throw new Error('MAKEFUN_PROMPT_FILE is not set');
  const text = readFileSync(path, 'utf8');
  const docs = documentsIn(text);
  const directory = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directory === null) throw new Error(`the brief at ${path} holds no ${PROVIDERS_SCHEMA} document`);
  const providers = (directory.providers as unknown[]).flatMap((raw) => {
    const p = providerOf(raw);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings = ratingsDoc === null ? [] : (ratingsDoc.ratings as unknown[]).flatMap((raw) => {
    const r = ratingOf(raw);
    return r === null ? [] : [r];
  });
  const workload = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const manifest = workload === null ? [] : (workload.manifest as unknown[]).flatMap((raw) => {
    const m = manifestOf(raw);
    return m === null ? [] : [m];
  });
  return { providers, ratings, manifest };
}

/**
 * The catalog `--list` prints and the register body carries: one entry per provider, its path as
 * base_url, and only the free models the quality table names. A paid model is never listed.
 */
export function buildCatalog(brief: Brief): CatalogReport {
  const providers = brief.providers.map((p) => ({
    id: p.id,
    base_url: p.base,
    key_env: p.key_env,
    wire: p.wire,
    models: brief.ratings
      .filter((r) => r.provider === p.id && r.free)
      .map((r) => ({ id: r.model, free: true, context_length: r.context_length })),
  }));
  return { providers };
}

/** Whether a model row meets one request's declared requirements. */
export function meets(row: RatingRow, requires: Requirement): boolean {
  if (requires.tools === true && !row.tools) return false;
  if (requires.vision === true && !row.vision) return false;
  if (requires.json === true && !row.json) return false;
  if (requires.min_context !== undefined && (row.context_length === null || row.context_length < requires.min_context)) return false;
  return true;
}
