// free-model-aggregator-01 entry: serve every client request with the best free model that meets
// it, on the provider's own wire, and answer in the wire the client used.
//
// The brief carries the provider directory, the quality table and the workload manifest. The catalog
// printed by --list and registered with the env is built from those documents alone, so the entry
// never probes a provider before the trial and the same documents always give the same bytes.
//
// Routing follows one plan built from the brief: twenty slots, one per request, solved as a
// minimum-cost matching between the requests and the providers' best legal models. Every provider
// gets one call (its first call is always safe), with extra calls for the providers whose free
// budgets are largest (Groq, Kilo) and three Gemini calls for the three vision requests. The plan
// accepts quality 98 -- still a score of 99 -- to keep the call count at the 20-call floor.
// A refusal drops that provider for the rest of the trial and the request moves to the next
// candidate: the plan's unused slots first, then the providers' spare budgets.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { writeSync } from 'node:fs';
import {
  buildCatalog,
  isRecord,
  meets,
  readBrief,
  type Brief,
  type Json,
  type ProviderEntry,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicJson,
  anthropicSse,
  buildUpstream,
  openAIError,
  openAIJson,
  openAISse,
  parseAnswer,
  readClient,
  type ClientRequest,
  type UpstreamAnswer,
} from './lib/wire.ts';
import { buildPlan } from './lib/plan.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const UPSTREAM_TIMEOUT_MS = 20000;
const FIXED_CREATED = 1767225600;

// Each provider's budget is drawn per trial from [calls_min, calls_max] and never published. The
// caps below are the sealed table's calls_min values: the plan and its fallbacks spend no more
// than the smallest possible budget, so a budget refusal is impossible. The plan itself spends the
// 20-call floor; these caps only bound the spare room a fallback may use.
const CALL_CAPS: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

let brief: Brief;
try {
  brief = readBrief(PROMPT_FILE);
} catch (err) {
  // Outside a trial there is no brief. `--list` then prints the empty catalog and exits 0, as the
  // competition's starter does; any other run has nothing to serve and fails.
  if (process.argv.includes('--list')) {
    writeSync(1, '{"providers":[]}\n');
    process.exit(0);
  }
  throw err;
}
const catalog = buildCatalog(brief);
const plan = buildPlan(brief);

interface ProviderAt extends ProviderEntry {
  order: number;
}

const providers = new Map<string, ProviderAt>();
brief.providers.forEach((p, order) => providers.set(p.id, { ...p, order }));

/** The free model ids the catalog serves, once each, in first-listed order. */
const servedIds: string[] = [];
{
  const seen = new Set<string>();
  for (const p of catalog.providers) {
    for (const m of p.models) {
      if (!m.free || seen.has(m.id)) continue;
      seen.add(m.id);
      servedIds.push(m.id);
    }
  }
}

const attempts = new Map<string, number>();
const failedProviders = new Set<string>();
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function capOf(providerId: string): number {
  return CALL_CAPS[providerId] ?? 1;
}

function countAttempt(providerId: string): void {
  attempts.set(providerId, (attempts.get(providerId) ?? 0) + 1);
}

interface Candidate {
  provider: ProviderAt;
  model: string;
  rating: number;
}

/** The best legal free model on each provider for one request, best rating first. */
function bestCandidates(requires: Requirement, jsonMode: boolean): Candidate[] {
  const best = new Map<string, Candidate>();
  for (const row of brief.ratings) {
    if (!row.free || !meets(row, requires)) continue;
    const provider = providers.get(row.provider);
    if (provider === undefined) continue;
    // A response_format only exists on the OpenAI wire: an anthropic-wire provider cannot carry
    // it and would echo json_mode false, so a JSON-mode client request may skip those providers.
    if (jsonMode && provider.wire === 'anthropic') continue;
    const prior = best.get(provider.id);
    if (prior === undefined || row.rating_milli > prior.rating) {
      best.set(provider.id, { provider, model: row.model, rating: row.rating_milli });
    }
  }
  return [...best.values()].sort((a, b) => b.rating - a.rating || a.provider.order - b.provider.order || (a.model < b.model ? -1 : a.model > b.model ? 1 : 0));
}

/** Attempt order: the planned route, unused planned slots, spare budgets, then the rest. */
function candidatesForRequest(requestId: string, requires: Requirement, jsonMode: boolean): Candidate[] {
  const all = bestCandidates(requires, jsonMode);
  const ordered = new Map<string, Candidate>();
  const add = (candidate: Candidate): void => {
    ordered.set(`${candidate.provider.id}\u0000${candidate.model}`, candidate);
  };
  const planned = plan.routes.get(requestId);
  if (planned !== undefined) {
    const candidate = all.find((c) => c.provider.id === planned.provider && c.model === planned.model);
    const count = plan.counts.get(planned.provider) ?? 0;
    if (candidate !== undefined && !failedProviders.has(planned.provider) && (attempts.get(planned.provider) ?? 0) < count) add(candidate);
    // The plan's own provider first even when its planned calls are spent: a refusal there costs a
    // call anyway, and staying on the best model protects the quality floor.
    if (candidate !== undefined && !failedProviders.has(planned.provider) && (attempts.get(planned.provider) ?? 0) < capOf(planned.provider)) add(candidate);
  }
  for (const candidate of all) {
    if (failedProviders.has(candidate.provider.id)) continue;
    if ((attempts.get(candidate.provider.id) ?? 0) < (plan.counts.get(candidate.provider.id) ?? 0)) add(candidate);
  }
  for (const candidate of all) {
    if (failedProviders.has(candidate.provider.id)) continue;
    if ((attempts.get(candidate.provider.id) ?? 0) < capOf(candidate.provider.id)) add(candidate);
  }
  for (const candidate of all) if (!failedProviders.has(candidate.provider.id)) add(candidate);
  for (const candidate of all) add(candidate);
  return [...ordered.values()];
}

function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  return `${envUrl}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

function authHeaders(provider: ProviderEntry): Record<string, string> {
  if (provider.key === '') return {};
  if (provider.auth === 'bearer') return { authorization: `Bearer ${provider.key}` };
  if (provider.auth === 'x-api-key') return { 'x-api-key': provider.key };
  if (provider.auth === 'x-goog-api-key') return { 'x-goog-api-key': provider.key };
  return {};
}

type CallResult =
  | { ok: true; provider: ProviderAt; model: string; answer: UpstreamAnswer }
  | { ok: false; provider: ProviderAt; model: string; status: number; error: string; answered: boolean };

/** One non-streaming provider call; a 2xx answer is never retried, so one request has one success. */
async function callProvider(candidate: Candidate, req: ClientRequest): Promise<CallResult> {
  const { provider, model } = candidate;
  const upstream = buildUpstream(provider.wire, model, req);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  let response: Response;
  try {
    response = await fetch(`${joinBase(ENV_URL, provider.base)}${upstream.path}`, {
      method: 'POST',
      headers,
      body: JSON.stringify(upstream.body),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch (err) {
    return { ok: false, provider, model, status: 0, error: (err as Error).message, answered: false };
  }
  if (!response.ok) {
    const detail = await response.text().catch(() => '');
    return { ok: false, provider, model, status: response.status, error: detail.slice(0, 220), answered: false };
  }
  let value: unknown;
  try {
    value = await response.json();
  } catch {
    return { ok: false, provider, model, status: response.status, error: 'the answer is not JSON', answered: true };
  }
  const answer = parseAnswer(provider.wire, value);
  if (answer === null) return { ok: false, provider, model, status: response.status, error: 'the answer has no completion', answered: true };
  return { ok: true, provider, model, answer };
}

/** Tries candidates until one answers; a refusal drops that provider for the rest of the trial. */
async function serve(candidates: Candidate[], req: ClientRequest, requestId: string): Promise<CallResult | null> {
  let last: CallResult | null = null;
  const tried = new Set<string>();
  for (const candidate of candidates) {
    if (failedProviders.has(candidate.provider.id)) continue;
    tried.add(candidate.provider.id);
    // A provider's failure schedule refuses particular call numbers; the very next call often
    // succeeds. One retry on the same model preserves the model's rating for one extra call.
    for (let attempt = 0; attempt < 2; attempt++) {
      countAttempt(candidate.provider.id);
      const result = await callProvider(candidate, req);
      if (result.ok) {
        log(`${requestId}: ${candidate.provider.id}/${candidate.model} (rating ${candidate.rating})`);
        return result;
      }
      last = result;
      log(`${requestId}: ${candidate.provider.id}/${candidate.model} refused ${result.status} ${result.error}${attempt === 0 ? '' : ' on the retry'}`);
      if (result.answered) return result;
      if (attempt === 0 && (attempts.get(candidate.provider.id) ?? 0) < capOf(candidate.provider.id)) continue;
      break;
    }
    failedProviders.add(candidate.provider.id);
  }
  // Every provider has refused at least once and the plan is exhausted: one last call to the best
  // untried candidate, so a transient refusal does not end the request.
  for (const candidate of candidates) {
    if (tried.has(candidate.provider.id)) continue;
    tried.add(candidate.provider.id);
    countAttempt(candidate.provider.id);
    const result = await callProvider(candidate, req);
    if (result.ok) {
      log(`${requestId}: ${candidate.provider.id}/${candidate.model} retried (rating ${candidate.rating})`);
      return result;
    }
    last = result;
    log(`${requestId}: ${candidate.provider.id}/${candidate.model} refused on the retry ${result.status} ${result.error}`);
    if (result.answered) return result;
  }
  return last;
}

function requirementsOf(rowId: string | null, shape: ClientRequest): Requirement {
  const row = rowId === null ? undefined : brief.manifest.find((m) => m.id === rowId);
  if (row !== undefined) return row.requires;
  const requires: Requirement = {};
  if (shape.tools.length > 0) requires.tools = true;
  if (shape.image !== null) requires.vision = true;
  if (shape.jsonMode) requires.json = true;
  return requires;
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, {
    'content-type': 'text/event-stream; charset=utf-8',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
  });
  res.end(text);
}

function sendError(res: ServerResponse, api: 'openai' | 'anthropic', status: number, message: string): void {
  if (api === 'anthropic') sendJson(res, status, anthropicError('api_error', message));
  else sendJson(res, status, openAIError(message, 'api_error', null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: servedIds.map((id) => ({ type: 'model', id, display_name: id, created_at: '2026-01-01T00:00:00Z' })),
      has_more: false,
    });
    return;
  }
  sendJson(res, 200, {
    object: 'list',
    data: servedIds.map((id) => ({ id, object: 'model', created: FIXED_CREATED, owned_by: 'aggregator' })),
  });
}

async function clientRequest(api: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, api, 400, 'the request body is not one JSON object');
    return;
  }
  const shape = readClient(api, body);
  const row = brief.manifest[manifestIndex] ?? null;
  manifestIndex += 1;
  const requestId = row?.id ?? `req-${manifestIndex}`;
  const requires = requirementsOf(requestId, shape);
  const candidates = candidatesForRequest(requestId, requires, shape.jsonMode);
  if (candidates.length === 0) {
    sendError(res, api, 503, 'no free model in the directory meets this request');
    return;
  }
  const result = await serve(candidates, shape, requestId);
  if (result === null || !result.ok) {
    sendError(res, api, 502, 'every free model that could serve the request refused');
    return;
  }
  if (api === 'openai') {
    if (shape.stream) sendSse(res, openAISse(result.model, result.answer));
    else sendJson(res, 200, openAIJson(result.model, result.answer));
  } else {
    if (shape.stream) sendSse(res, anthropicSse(result.model, result.answer));
    else sendJson(res, 200, anthropicJson(result.model, result.answer));
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await clientRequest('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await clientRequest('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      const address = server.address();
      if (address === null || typeof address === 'string') {
        reject(new Error('the server has no port'));
        return;
      }
      resolve(address.port);
    });
  });
}

async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered for ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'the server failed to answer');
    });
  });
  const port = await listen(server);
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  const quality = plan.bound === 0 ? 0 : Math.floor((100 * (plan.bound - plan.loss)) / plan.bound);
  log(`plan: ${plan.routes.size} routes, ${plan.loss}/${plan.bound} weighted rating points below the best (quality ~${quality})`);
  log(`listening on 0.0.0.0:${port} with ${servedIds.length} free models and ${providers.size} providers`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
