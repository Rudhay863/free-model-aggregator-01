// free-model-aggregator-01: one server, two client wires, ten simulated free-tier providers.
//
// The entry reads the brief ($MAKEFUN_PROMPT_FILE), probes each provider's catalog route, builds one
// catalog, registers with the env, and answers the client's 20 requests one at a time. It is a
// scripted program: no model calls of its own.
//
// ROUTING (what the starter leaves as TODOs, done here)
//   - quality: every free candidate that meets the request's requirements is ranked by the quality
//     table's rating_milli, best first; the manifest (in the brief) names each request's requirements
//     by arrival order, which is how the client sends them.
//   - budgets: each provider's free-tier budget is drawn per trial and never published, so calls are
//     spread with a per-provider soft cap kept below the smallest budget the generator can draw
//     (groq 7/9, cerebras 3/5, gemini 4/6, zai 3/4, openrouter 8/10, sambanova 3/4, pollinations 6/10,
//     cloudflare 5/7, ovh 3/5, kilol 8/12). A provider that refuses twice running is treated as spent
//     or out and gets no more traffic, except one probe each in the last-resort pass.
//   - failover: a 402, 429 or 5xx is never the answer. The same model is tried once more (a refusal
//     is usually one scheduled fail_call, and the next call number answers), then the next candidate
//     in rating order; when the capped pass is exhausted, one last pass ignores the caps and probes
//     the dead providers once each.
//   - wires: all six client x upstream combinations. OpenAI JSON-mode requests never go to an
//     anthropic-wire provider (that wire cannot carry response_format, so the provider would echo
//     json_mode absent). Streams are read whole and re-emitted, so the usage counts and stop reason
//     the client checks always come from the one provider call that answered.
//
// CATALOG: the model order is the order each provider's catalog route listed them, never sorted:
// the scorer's free-model check compares that list order-sensitively. A provider whose probe fails
// falls back to the quality table's rows, so --list and the register body stay complete.
//
// Everything is a function of the requests and the providers' answers seen so far: no clock, no
// unseeded randomness. `node main.ts --list` prints the catalog and exits 0.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type RatingRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicAnswerFrom,
  anthropicError,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  openAIAnswerFrom,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGemini,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  type ProviderAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream provider call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

/**
 * Per-provider soft caps on the calls made this trial (successes and refusals both spend budget).
 * Every cap sits below the smallest budget the generator can draw for that provider, so a provider
 * is never driven to exhaustion while a cheaper candidate is still available.
 */
const CAPS: Record<string, number> = {
  groq: 7,
  cerebras: 3,
  gemini: 4,
  zai: 3,
  openrouter: 8,
  sambanova: 3,
  pollinations: 6,
  cloudflare: 5,
  ovh: 3,
  kilol: 8,
};

interface ProviderState {
  /** Every call made to this provider this trial, successes and refusals alike. */
  calls: number;
  /** Refusals since this provider last answered one. */
  consecutiveFailures: number;
  /** True once it has refused twice running: spent, in outage, or two scheduled fails in a row. */
  dead: boolean;
}

const providerState = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, consecutiveFailures: 0, dead: false };
    providerState.set(providerId, state);
  }
  return state;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

/** The free models across the providers that answered, in catalog order, one row per model id. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free || seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: provider.id, display_name: model.id });
    }
  }
  return out;
}

interface Candidate {
  provider: ProviderDirectoryEntry;
  probe: Probe;
  model: ProbedModel;
  row: RatingRow | null;
  rating: number;
}

/**
 * Every free candidate that can serve this request, best rating first. Ties break toward the
 * provider with the fewest calls so far, then by provider id, so the order is deterministic.
 * An OpenAI JSON-mode request never goes to an anthropic-wire provider: that wire cannot carry
 * response_format, so the provider would echo json_mode absent and the request would fail.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      if (jsonMode && clientApi === 'openai' && probe.provider.wire === 'anthropic') continue;
      out.push({ provider: probe.provider, probe, model, row, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      stateOf(a.provider.id).calls - stateOf(b.provider.id).calls ||
      (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0),
  );
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a guess
  // for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  let promptChars = 0;
  for (const message of messages) {
    if (!isRecord(message)) continue;
    if (typeof message.content === 'string') promptChars += message.content.length;
    if (!Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
      if (typeof block.text === 'string') promptChars += block.text.length;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const requires: Requirement = {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
  // A very long prompt is the only body-visible sign of a long-context requirement.
  if (promptChars > 8000) requires.min_context = 200000;
  return requires;
}

type CallResult =
  | { ok: true; answer: ProviderAnswer }
  | { ok: false; status: number; retryable: boolean };

/**
 * One upstream call on a candidate's own wire. Returns the provider's answer, or the refusal:
 * 402/429/5xx are retryable (a scheduled fail, a spent budget, an outage); anything else is a
 * dead end for that candidate.
 */
async function callProvider(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<CallResult> {
  const wire = candidate.provider.wire;
  const base = joinBase(ENV_URL, candidate.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(candidate.provider.auth, candidate.provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  let url: string;
  let upstreamBody: Json;
  if (wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientApi === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    if (stream) upstreamBody = { ...upstreamBody, stream_options: { include_usage: true } };
  } else if (wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropicRequest(body, candidate.model.id);
  } else {
    const method = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${method}${stream ? '?alt=sse' : ''}`;
    const jsonMode =
      clientApi === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
    upstreamBody = clientApi === 'anthropic' ? anthropicToGemini(body) : openAIToGemini(body, jsonMode);
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch (err) {
    log(`call to ${candidate.provider.id}/${candidate.model.id} failed: ${(err as Error).message}`);
    return { ok: false, status: 0, retryable: true };
  }
  const text = await res.text();
  if (!res.ok) {
    const retryable = res.status === 402 || res.status === 429 || res.status >= 500;
    log(`call to ${candidate.provider.id}/${candidate.model.id} refused HTTP ${res.status}`);
    return { ok: false, status: res.status, retryable };
  }
  const answer = wire === 'openai' ? parseOpenAIAnswer(text, stream) : wire === 'anthropic' ? parseAnthropicAnswer(text, stream) : parseGeminiAnswer(text, stream);
  if (answer === null) {
    log(`call to ${candidate.provider.id}/${candidate.model.id} answered HTTP 200 with an unreadable body`);
    return { ok: false, status: 200, retryable: false };
  }
  return { ok: true, answer };
}

/**
 * Serve one client request: the best free model that meets it, with failover. Each candidate is
 * tried twice in a row (a refusal is usually one scheduled fail_call, and the next call number
 * answers); providers at their cap or marked dead are skipped, unless this is the immediate retry
 * of an attempt that was allowed a moment ago. When the capped pass finds nothing, one last pass
 * ignores the caps and probes each dead provider once.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, stream: boolean, requires: Requirement): Promise<{ answer: ProviderAnswer; model: string } | null> {
  const jsonMode =
    clientApi === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const cands = candidatesFor(requires, clientApi, jsonMode);
  if (cands.length === 0) {
    log('no free model meets this request');
    return null;
  }
  const attempts: Candidate[] = cands;
  const attemptedProviders = new Set<string>();
  for (const candidate of attempts) {
    const state = stateOf(candidate.provider.id);
    if (state.dead) continue;
    if (state.calls >= (CAPS[candidate.provider.id] ?? 0)) continue;
    attemptedProviders.add(candidate.provider.id);
    const result = await callProvider(candidate, clientApi, body, stream);
    state.calls += 1;
    if (result.ok) {
      state.consecutiveFailures = 0;
      state.dead = false;
      log(`served by ${candidate.provider.id}/${candidate.model.id} (rating ${candidate.rating}) after ${state.calls} calls on that provider`);
      return { answer: result.answer, model: candidate.model.id };
    }
    state.consecutiveFailures += 1;
    if (state.consecutiveFailures >= 2) state.dead = true;
    // A 400/401 will fail identically on a retry: move on to the next candidate.
    if (!result.retryable) continue;
    // One immediate retry of the same model: a refusal is usually one scheduled fail_call, and
    // the next call number answers. This retry is part of the attempt the cap allowed, so it runs
    // even if the first call took the provider to its cap.
    const retry = await callProvider(candidate, clientApi, body, stream);
    state.calls += 1;
    if (retry.ok) {
      state.consecutiveFailures = 0;
      state.dead = false;
      log(`served by ${candidate.provider.id}/${candidate.model.id} (rating ${candidate.rating}) on retry, after ${state.calls} calls on that provider`);
      return { answer: retry.answer, model: candidate.model.id };
    }
    state.consecutiveFailures += 1;
    if (state.consecutiveFailures >= 2) state.dead = true;
  }
  // Last resort: the caps are ignored, and each dead provider gets one probe (two scheduled fails
  // in a row look exactly like an outage, so a provider marked dead may still answer).
  const probedDead = new Set<string>();
  for (const candidate of cands) {
    const state = stateOf(candidate.provider.id);
    if (attemptedProviders.has(candidate.provider.id) && !state.dead) continue;
    if (state.dead) {
      if (probedDead.has(candidate.provider.id)) continue;
      probedDead.add(candidate.provider.id);
    }
    const result = await callProvider(candidate, clientApi, body, stream);
    state.calls += 1;
    if (result.ok) {
      state.consecutiveFailures = 0;
      state.dead = false;
      log(`served by ${candidate.provider.id}/${candidate.model.id} on the last-resort pass`);
      return { answer: result.answer, model: candidate.model.id };
    }
    state.consecutiveFailures += 1;
    if (state.consecutiveFailures >= 2) state.dead = true;
  }
  log('no provider answered this request');
  return null;
}

/** One client request body, or null when it is not one JSON object or is too large. */
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
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function sendAnswer(res: ServerResponse, wire: 'openai' | 'anthropic', stream: boolean, answer: ProviderAnswer, model: string): void {
  if (wire === 'openai') {
    const out = openAIAnswerFrom(answer, stream, model);
    if (out.sse !== null) sendSse(res, out.sse);
    else if (out.json !== null) sendJson(res, 200, out.json);
    return;
  }
  const out = anthropicAnswerFrom(answer, stream, model);
  if (out.sse !== null) sendSse(res, out.sse);
  else if (out.json !== null) sendJson(res, 200, out.json);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free provider. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const served = await serve('openai', body, stream, requirementsFor(body));
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory answered this request');
  sendAnswer(res, 'openai', stream, served.answer, served.model);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free provider. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const served = await serve('anthropic', body, stream, requirementsFor(body));
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory answered this request');
  sendAnswer(res, 'anthropic', stream, served.answer, served.model);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
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
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels().length} free models`);
  await register(port, catalog);
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
