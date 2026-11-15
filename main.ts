// free-tier-gateway: one server, two client APIs, ten free-tier providers, and a quality-ordered router.
//
// A trial hands this program a brief ($MAKEFUN_PROMPT_FILE) and an env ($MAKEFUN_ENV_URL). The brief lists
// ten free-tier providers (id, name, wire, auth scheme, key variable, key, path), the quality table (each
// model's rating, context length and whether it takes tools, images and JSON mode, and whether it is free)
// and the workload (the requests the client will send, in order, with the requirements each one makes).
//
// The run is:
//   1. read the brief;
//   2. probe every provider at its own catalog route, in that provider's own wire;
//   3. build one catalog document from the probes, and use that same object for `--list` and for register;
//   4. listen on a free port, POST /register {protocol_version, port, catalog}, and serve;
//   5. poll GET /work and exit 0 the moment it says done.
//
// Each client request is routed to the highest-rated free model that meets the request's requirements,
// translating into that provider's wire and back into the client's: OpenAI Chat Completions and Anthropic
// Messages, JSON and server-sent events. A provider that refuses (402, 429, 5xx or 400) is recorded and the
// request moves to the next-best model, so one spent free tier never takes a request down with it.
//
// Nothing here reads a wall clock or draws an unseeded random number: every choice is a function of the
// brief, the requests and the providers' answers, so a replay of the trial leaves the same state export.
//
// FILES
//   main.ts            this server: brief, probes, catalog, routes, register, work poll.
//   lib/providers.ts   the brief's shapes and reader, the auth headers, the catalog probe and the catalog.
//   lib/wire.ts        the normalized request and the OpenAI, Anthropic and Gemini wires in both directions.
//
// Run: node main.ts            serve a trial (needs MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE)
//      node main.ts --list     print the catalog document and exit 0

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  freeModels,
  isRecord,
  joinBase,
  meetsRequirements,
  probeProvider,
  readBrief,
  type Catalog,
  type ClientApi,
  type Json,
  type ModelEntry,
  type Probe,
  type Requirement,
  type WorkloadRequest,
} from './lib/providers.ts';
import {
  anthropicMessage,
  anthropicModelList,
  anthropicStreamFrames,
  buildUpstreamBody,
  errorBody,
  normalizeRequest,
  openAICompletion,
  openAIModelList,
  openAIStreamFrames,
  readUpstream,
  upstreamPath,
  type NormalizedRequest,
  type ServedModel,
  type UpstreamAnswer,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 20000;
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** How many refusals in a row settle a provider that never says why it refused. */
const OUT_AFTER_REFUSALS = 4;
/** The most provider calls one client request may spend before it gives up. */
const MAX_ATTEMPTS = 6;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
let candidates: ModelEntry[] = [];
/** The manifest row of each client request, in arrival order: the client sends them one at a time. */
let manifestIndex = 0;
/** A per-request counter for the response ids: deterministic, so a replay writes the same bytes. */
let requestCounter = 0;

/** What this trial has learned about one provider, from its answers alone. */
interface ProviderState {
  calls: number;
  refusals: number;
  refusalsInARow: number;
  /** True once the provider's own answer said its allowance is gone or it is down: that lasts the trial. */
  out: boolean;
}

const providerState = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, refusals: 0, refusalsInARow: 0, out: false };
    providerState.set(providerId, state);
  }
  return state;
}

/**
 * Whether a refusal is final for the rest of the trial.
 *
 * A free tier says so in its own words when its allowance is spent or it is down, and both last until the
 * trial ends; a scheduled refusal is one call. Reading the provider's message lets one refusal settle a
 * provider that is really gone, instead of spending a second call to find out.
 */
const FINAL_REFUSAL = /budget|allowance|quota|credit|spent|exhaust|is out|unavailable|suspended/i;

function refusalIsFinal(message: string): boolean {
  return FINAL_REFUSAL.test(message);
}

/** The message a provider's error body carries, for the shapes the three wires use. */
function errorMessageOf(body: unknown): string {
  if (!isRecord(body)) return '';
  const error = isRecord(body.error) ? body.error : null;
  if (error !== null) {
    if (typeof error.message === 'string') return error.message;
    if (typeof error.code === 'string') return error.code;
  }
  if (typeof body.message === 'string') return body.message;
  return '';
}

function log(message: string): void {
  process.stderr.write(`free-tier-gateway: ${message}\n`);
}

/** Probe every provider in its own wire, then build the one catalog and the route table. */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok) log(probe.error ?? `${probe.provider.id} did not answer its catalog route`);
  catalog = buildCatalog(probes, brief.ratings);
  candidates = freeModels(probes, brief.ratings);
}

/** The models GET /v1/models lists: exactly the free ids the catalog registered, once each. */
function servedModels(): ServedModel[] {
  const seen = new Set<string>();
  const out: ServedModel[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free || seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: provider.id, display_name: model.id });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload ?? [];
  const row: WorkloadRequest | undefined = manifest[manifestIndex];
  if (row !== undefined) return row.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, for a brief that carries no manifest row for this request. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message)) continue;
    if (!Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const requires: Requirement = {};
  if (Array.isArray(body.tools) && body.tools.length > 0) requires.tools = true;
  if (vision) requires.vision = true;
  if (format === 'json_object' || format === 'json_schema') requires.json = true;
  return requires;
}

/**
 * The free models that may serve one request, best first.
 *
 * Ranked by the quality table's rating, because quality is 60 percent of the score. Ties go to the provider
 * this trial has found least trouble with, then to the one it has called least, so load spreads over the
 * free tiers instead of burning one dry. A provider whose wire cannot carry a field the request asked for
 * (an Anthropic upstream has no `response_format`, and its wire demands a `max_tokens` the client may not
 * have sent) is kept, but after every provider that can: a wire-compatible answer at a slightly lower
 * rating scores, and an echo the client can compare scores, where a dropped field passes no check at all.
 */
function candidatesFor(requires: Requirement, req: NormalizedRequest): ModelEntry[] {
  const penalized = (entry: ModelEntry): number => {
    const dropFormat = req.jsonMode && entry.provider.wire === 'anthropic' ? 1 : 0;
    const addCap = req.maxTokens === null && entry.provider.wire === 'anthropic' ? 1 : 0;
    return dropFormat + addCap;
  };
  const usable = candidates.filter((entry) => meetsRequirements(requires, entry.caps) && !deadModels.has(modelKey(entry)));
  return usable.sort((a, b) => {
    const pa = penalized(a);
    const pb = penalized(b);
    if (pa !== pb) return pa - pb;
    const sa = stateOf(a.provider.id);
    const sb = stateOf(b.provider.id);
    if (sa.out !== sb.out) return sa.out ? 1 : -1;
    if (a.rating !== b.rating) return b.rating - a.rating;
    if (sa.refusals !== sb.refusals) return sa.refusals - sb.refusals;
    if (sa.calls !== sb.calls) return sa.calls - sb.calls;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
}

/** One provider call: the answer, or the status and code of its refusal. */
interface CallOutcome {
  ok: boolean;
  status: number;
  code: string;
  answer: UpstreamAnswer | null;
}

/** The error code a provider's refusal carries, in the shape its wire uses. */
function errorCodeOf(body: unknown): string {
  if (!isRecord(body)) return '';
  const error = isRecord(body.error) ? body.error : null;
  if (error === null) return '';
  if (typeof error.code === 'string') return error.code;
  if (typeof error.status === 'string') return error.status;
  if (typeof error.type === 'string') return error.type;
  return '';
}

/** A model a provider answered 400 for: this provider does not serve it, whatever the catalog said. */
const deadModels = new Set<string>();

const modelKey = (entry: ModelEntry): string => `${entry.provider.id}/${entry.model.id}`;

async function callProvider(entry: ModelEntry, req: NormalizedRequest): Promise<CallOutcome> {
  const base = joinBase(ENV_URL, entry.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    accept: 'application/json',
    ...authHeaders(entry.provider.auth, entry.provider.key),
  };
  if (entry.provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  const body = buildUpstreamBody(entry.provider.wire, entry.model.id, req);
  const state = stateOf(entry.provider.id);
  state.calls += 1;
  try {
    const res = await fetch(`${base}${upstreamPath(entry.provider.wire, entry.model.id)}`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
    const text = await res.text();
    let parsed: unknown = null;
    try {
      parsed = text === '' ? null : JSON.parse(text);
    } catch {
      parsed = null;
    }
    if (!res.ok) {
      state.refusals += 1;
      state.refusalsInARow += 1;
      const message = errorMessageOf(parsed);
      // A 400 is about this model, not the provider: drop the model and keep the provider.
      if (res.status === 400) deadModels.add(modelKey(entry));
      // A spent allowance or an outage outlasts this request, so stop spending calls on that provider. A
      // scheduled refusal is one call and says nothing about the next, so the provider stays in the running.
      if (res.status !== 400 && (refusalIsFinal(message) || state.refusalsInARow >= OUT_AFTER_REFUSALS)) state.out = true;
      return { ok: false, status: res.status, code: errorCodeOf(parsed), answer: null };
    }
    const answer = readUpstream(entry.provider.wire, parsed);
    if (answer === null) {
      state.refusals += 1;
      state.refusalsInARow += 1;
      if (state.refusalsInARow >= OUT_AFTER_REFUSALS) state.out = true;
      return { ok: false, status: 502, code: 'unreadable_answer', answer: null };
    }
    state.refusalsInARow = 0;
    return { ok: true, status: res.status, code: '', answer };
  } catch (err) {
    state.refusals += 1;
    state.refusalsInARow += 1;
    if (state.refusalsInARow >= OUT_AFTER_REFUSALS) state.out = true;
    return { ok: false, status: 502, code: (err as Error).name, answer: null };
  }
}

/** A chosen model and the answer it gave. */
interface Routed {
  entry: ModelEntry;
  answer: UpstreamAnswer;
}

/**
 * Serve one request: walk the candidates best first, skipping the providers this trial has found out (and
 * coming back to them only when nothing else is left), and return the first answer that arrives. The walk
 * stops after MAX_ATTEMPTS calls, so a directory whose free tiers are all spent still answers in bounded
 * time rather than burning the trial's wall.
 */
async function route(requires: Requirement, req: NormalizedRequest): Promise<Routed | null> {
  const all = candidatesFor(requires, req);
  const fresh = all.filter((entry) => !stateOf(entry.provider.id).out);
  const spent = all.filter((entry) => stateOf(entry.provider.id).out);
  let attempts = 0;
  for (const entry of [...fresh, ...spent]) {
    if (attempts >= MAX_ATTEMPTS) break;
    attempts += 1;
    const outcome = await callProvider(entry, req);
    if (outcome.ok && outcome.answer !== null) return { entry, answer: outcome.answer };
    log(`${entry.provider.id}/${entry.model.id} refused (${outcome.status} ${outcome.code}); next free model`);
  }
  return null;
}

async function readJsonBody(req: IncomingMessage): Promise<Json | null> {
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

function sendError(res: ServerResponse, api: ClientApi, status: number, message: string): void {
  sendJson(res, status, errorBody(api, status, message));
}

/** Answer one routed request in the wire the client used, streamed or whole as it asked. */
function answer(res: ServerResponse, api: ClientApi, req: NormalizedRequest, routed: Routed, id: string): void {
  const answer = routed.answer;
  if (!req.stream) {
    sendJson(res, 200, api === 'openai' ? openAICompletion(id, routed.entry.model.id, answer) : anthropicMessage(id, routed.entry.model.id, answer));
    return;
  }
  res.writeHead(200, {
    'content-type': 'text/event-stream; charset=utf-8',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
  });
  const frames = api === 'openai' ? openAIStreamFrames(id, routed.entry.model.id, answer) : anthropicStreamFrames(id, routed.entry.model.id, answer);
  for (const frame of frames) res.write(frame);
  res.end();
}

/** One client request, whatever its wire. */
async function handleRequest(req: IncomingMessage, res: ServerResponse, api: ClientApi): Promise<void> {
  const body = await readJsonBody(req);
  const requires = requirementsFor(body ?? {});
  manifestIndex += 1;
  if (body === null) {
    sendError(res, api, 400, 'the request body is not one JSON object');
    return;
  }
  requestCounter += 1;
  const id = `${api === 'openai' ? 'chatcmpl' : 'msg'}-free-${requestCounter}`;
  const normalized = normalizeRequest(api, body);
  const routed = await route(requires, normalized);
  if (routed === null) {
    sendError(
      res,
      api,
      503,
      ENV_URL === ''
        ? 'no providers are registered: start the server with MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE set to a trial env'
        : 'no free model on the directory can serve this request',
    );
    return;
  }
  answer(res, api, normalized, routed, id);
}

/** GET /v1/models: the wire is the auth header. `x-api-key` asks for Anthropic's shape, Bearer for OpenAI's. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  const path = url.pathname.replace(/\/+$/, '') === '' ? '/' : url.pathname.replace(/\/+$/, '');
  if (req.method === 'GET' && path === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && path === '/v1/chat/completions') {
    await handleRequest(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && path === '/v1/messages') {
    await handleRequest(req, res, 'anthropic');
    return;
  }
  sendError(res, path.startsWith('/v1/messages') ? 'anthropic' : 'openai', 404, `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      const address = server.address();
      if (address === null || typeof address === 'string') reject(new Error('the server has no port'));
      else resolve(address.port);
    });
  });
}

/** Listen on one named port (a standalone run), falling back to any free port if it is taken. */
function listenOn(server: Server, port: number): Promise<number> {
  return new Promise((resolve, reject) => {
    const onError = (err: NodeJS.ErrnoException): void => {
      if (err.code === 'EADDRINUSE') {
        server.removeListener('error', onError);
        resolve(listen(server));
        return;
      }
      reject(err);
    };
    server.once('error', onError);
    server.listen(port, '0.0.0.0', () => {
      server.removeListener('error', onError);
      const address = server.address();
      if (address === null || typeof address === 'string') reject(new Error('the server has no port'));
      else resolve(address.port);
    });
  });
}

/** POST /register with the port and the one catalog. */
async function register(port: number, report: Catalog): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  const text = await res.text();
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${text.slice(0, 500)}`);
  let reply: unknown = null;
  try {
    reply = JSON.parse(text);
  } catch {
    reply = null;
  }
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms and exit 0 the moment it says the workload is done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            log('the env says the workload is done: exiting 0');
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not answering yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

/**
 * Start the server.
 *
 * Under a trial (MAKEFUN_ENV_URL set) it binds a port the OS picks, registers, serves and exits 0 when the
 * env says the work is done. Outside one it still serves, on $PORT or 8787: with no env there is no catalog
 * to probe and nothing to register with, so it answers /v1/models with an empty list and refuses a chat
 * request with a clear 503, which is what `npm start` can do before anyone points it at providers.
 */
async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  const inTrial = ENV_URL !== '';
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'the server failed to answer');
      else res.end();
    });
  });
  const port = inTrial ? await listen(server) : await listenOn(server, localPort());
  log(`listening on 0.0.0.0:${port}: ${catalog.providers.length} providers, ${servedModels().length} free models`);
  if (!inTrial) {
    log('MAKEFUN_ENV_URL is not set: serving standalone, with no providers registered and no workflow to poll');
    process.on('SIGTERM', () => process.exit(0));
    process.on('SIGINT', () => process.exit(0));
    return;
  }
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

/** The port a standalone run listens on: $PORT when it is a port, else 8787. */
function localPort(): number {
  const raw = process.env.PORT ?? '';
  const parsed = /^\d{1,5}$/.test(raw) ? Number(raw) : NaN;
  return Number.isSafeInteger(parsed) && parsed > 0 && parsed <= 65535 ? parsed : 8787;
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
