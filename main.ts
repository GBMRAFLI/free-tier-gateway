// Free-tier model aggregator: one server, two client APIs, a directory of providers.
//
// A trial gives this server a brief ($MAKEFUN_PROMPT_FILE) and an env ($MAKEFUN_ENV_URL). It probes
// each provider's catalog, registers one catalog document, serves the client's OpenAI Chat Completions
// and Anthropic Messages requests (JSON and SSE), and exits 0 when GET /work says the workload is done.
// Routing ranks every free model that meets a request's requirements by the quality table's
// rating_milli, fails over on 402/429/5xx (and on 400 to the next model), and speaks the OpenAI,
// Anthropic and Gemini upstream wires. A one-off 429 demotes a provider only among equal ratings.
// `node main.ts --list` prints the same catalog the register body carries, byte for byte.
//
// Node 22 standard library only. Relative imports keep their .ts extension. Choices are a function of
// the brief, the requests and the providers' answers: no wall clock and no unseeded randomness.

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
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGemini,
  passthroughEvents,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 15000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;

/** Providers that have refused with a lasting 402/429 (budget spent or an outage), not a one-off. */
const exhausted = new Set<string>();
/** Providers whose last inference call was a 402/429; a later success clears the demotion. */
const demoted = new Set<string>();
/** Consecutive lasting refusals, used to turn a demotion into exhaustion. */
const strikes = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Route {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

function ratingFor(probe: Probe, model: ProbedModel): number {
  return ratingOfModel(brief.ratings, probe.provider.id, model.id)?.rating_milli ?? 0;
}

/**
 * Whether this request must be served by a wire that can carry JSON mode. An OpenAI client that
 * asked for response_format is echoed as json_mode; an Anthropic-wire provider always echoes false.
 */
function needsJsonWire(body: Json, requires: Requirement): boolean {
  if (requires.json !== true) return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

function wireCarriesJson(wire: string): boolean {
  return wire === 'openai' || wire === 'gemini';
}

function compareRoutes(a: Route, b: Route): number {
  if (b.rating !== a.rating) return b.rating - a.rating;
  const aDemoted = demoted.has(a.probe.provider.id) ? 1 : 0;
  const bDemoted = demoted.has(b.probe.provider.id) ? 1 : 0;
  if (aDemoted !== bDemoted) return aDemoted - bDemoted;
  const providers = a.probe.provider.id.localeCompare(b.probe.provider.id);
  if (providers !== 0) return providers;
  return a.model.id.localeCompare(b.model.id);
}

/**
 * Rank remaining free candidates: highest rating first. A one-off 429 only demotes a provider
 * among equally rated models; it must not hide a higher-rated model that still has budget.
 * Ties break on provider id, then model id, so two catalogs that list the same models in a
 * different order still pick the same route.
 */
function candidatesFor(requires: Requirement, body: Json): Route[] {
  const jsonWire = needsJsonWire(body, requires);
  const out: Route[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (exhausted.has(probe.provider.id)) continue;
    if (jsonWire && !wireCarriesJson(probe.provider.wire)) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: ratingFor(probe, model) });
    }
  }
  out.sort(compareRoutes);
  return out;
}

/** 402 is always a spent budget. 429/5xx may be a one-off until a second strike. */
function isLastingRefusal(status: number, code: string | null): boolean {
  if (code === 'concurrent_requests') return false;
  if (status === 402) return true;
  if (code === 'payment_required' || code === 'insufficient_quota' || code === 'neurons_exhausted') return true;
  return status === 429 || status >= 500 || code === 'upstream_unreachable' || code === 'upstream_error' || code === 'refused';
}

function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
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

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

function refusalCode(wire: string, status: number, text: string): string | null {
  if (status !== 402 && status !== 429 && status < 500) return null;
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      const err = value.error;
      if (typeof err.code === 'string') return err.code;
      if (typeof err.type === 'string') return err.type;
      if (typeof err.status === 'string') return err.status;
    }
  } catch {
    // Not JSON: still a refusal of this status.
  }
  void wire;
  return status >= 500 ? 'upstream_error' : 'refused';
}

function noteRefusal(providerId: string, status: number, code: string | null): void {
  if (code === 'concurrent_requests') return;
  if (!isLastingRefusal(status, code)) return;
  const next = (strikes.get(providerId) ?? 0) + 1;
  strikes.set(providerId, next);
  demoted.add(providerId);
  // A 402 (or an explicit quota code) is the spent-budget signal: do not try this provider again.
  // Two lasting refusals with no success in between is an outage or a spent budget; a single 429
  // may only be that provider's scheduled one-off failure.
  if (status === 402 || code === 'payment_required' || code === 'insufficient_quota' || code === 'neurons_exhausted' || next >= 2) {
    exhausted.add(providerId);
  }
}

function noteSuccess(providerId: string): void {
  strikes.set(providerId, 0);
  demoted.delete(providerId);
}

interface UpstreamRequest {
  url: string;
  headers: Record<string, string>;
  body: Json;
}

function openAIForwardBody(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  if (body.stream === true) {
    const options = isRecord(body.stream_options) ? { ...body.stream_options } : {};
    options.include_usage = true;
    out.stream_options = options;
  }
  return out;
}

function upstreamRequest(route: Route, client: 'openai' | 'anthropic', body: Json): UpstreamRequest {
  const provider = route.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  const stream = body.stream === true;
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    return { url: `${base}/v1/chat/completions`, headers, body: openAIForwardBody(client === 'openai' ? body : anthropicToOpenAI(body, route.model.id), route.model.id) };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const payload = client === 'anthropic' ? { ...body, model: route.model.id } : openAIToAnthropicRequest(body, route.model.id);
    return { url: `${base}/v1/messages`, headers, body: payload };
  }
  const payload = client === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
  const action = stream ? 'streamGenerateContent' : 'generateContent';
  const url = `${base}/v1beta/models/${encodeURIComponent(route.model.id)}:${action}${stream ? '?alt=sse' : ''}`;
  return { url, headers, body: payload };
}

async function writeEvents(res: ServerResponse, events: AsyncIterable<string>): Promise<void> {
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
  for await (const event of events) res.write(event);
  res.end();
}

async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

async function deliver(res: ServerResponse, client: 'openai' | 'anthropic', body: Json, route: Route, upstream: Response): Promise<void> {
  const stream = body.stream === true;
  const providerWire = route.probe.provider.wire;
  const requested = typeof body.model === 'string' && body.model !== '' && body.model !== 'auto' ? body.model : route.model.id;
  if (stream && upstream.body !== null) {
    if (client === 'openai' && providerWire === 'openai') {
      await pipeStream(upstream.body, res);
      return;
    }
    if (client === 'anthropic' && providerWire === 'anthropic') {
      await writeEvents(res, passthroughEvents(upstream.body));
      return;
    }
    if (client === 'openai' && providerWire === 'anthropic') {
      await writeEvents(res, openAIEventsFromAnthropic(upstream.body, requested));
      return;
    }
    if (client === 'anthropic' && providerWire === 'openai') {
      await writeEvents(res, anthropicEventsFromOpenAI(upstream.body, requested));
      return;
    }
    if (client === 'openai' && providerWire === 'gemini') {
      await writeEvents(res, openAIEventsFromGemini(upstream.body, requested));
      return;
    }
    await writeEvents(res, anthropicEventsFromGemini(upstream.body, requested));
    return;
  }
  const answer: unknown = await upstream.json();
  const record = isRecord(answer) ? answer : {};
  if (client === 'openai' && providerWire === 'openai') {
    sendJson(res, 200, record);
    return;
  }
  if (client === 'anthropic' && providerWire === 'anthropic') {
    sendJson(res, 200, record);
    return;
  }
  if (client === 'openai' && providerWire === 'anthropic') {
    sendJson(res, 200, anthropicToOpenAIResponse(record, requested));
    return;
  }
  if (client === 'anthropic' && providerWire === 'openai') {
    sendJson(res, 200, openAIToAnthropic(record, requested));
    return;
  }
  if (client === 'openai' && providerWire === 'gemini') {
    sendJson(res, 200, geminiToOpenAI(record, requested));
    return;
  }
  sendJson(res, 200, geminiToAnthropic(record, requested));
}

async function serveClient(req: IncomingMessage, res: ServerResponse, client: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, client, 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const requires = requirementsFor(body);
  const routes = candidatesFor(requires, body);
  if (routes.length === 0) {
    return sendError(res, client, 503, client === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  let lastStatus = 503;
  let lastMessage = 'every candidate provider refused this request';
  const skipped = new Set<string>();
  for (const route of routes) {
    const providerId = route.probe.provider.id;
    if (skipped.has(providerId) || exhausted.has(providerId)) continue;
    const built = upstreamRequest(route, client, body);
    let upstream: Response;
    try {
      upstream = await fetch(built.url, {
        method: 'POST',
        headers: built.headers,
        body: JSON.stringify(built.body),
        signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
      });
    } catch (err) {
      lastStatus = 502;
      lastMessage = (err as Error).message;
      noteRefusal(providerId, 502, 'upstream_unreachable');
      skipped.add(providerId);
      continue;
    }
    if (upstream.ok) {
      noteSuccess(providerId);
      await deliver(res, client, body, route, upstream);
      return;
    }
    const text = await upstream.text();
    const code = refusalCode(route.probe.provider.wire, upstream.status, text);
    lastStatus = upstream.status;
    lastMessage = `the provider answered HTTP ${upstream.status}: ${text.slice(0, 300)}`;
    noteRefusal(providerId, upstream.status, code);
    // Budget and outage refusals are provider-wide: skip this provider's other models on this request.
    // A 400/401 (unknown or paid model, wrong key) is not: the next candidate may be another of its models.
    if (isLastingRefusal(upstream.status, code) || code === 'concurrent_requests') skipped.add(providerId);
    continue;
  }
  sendError(res, client, lastStatus >= 400 ? lastStatus : 503, client === 'anthropic' ? 'api_error' : 'upstream_refused', lastMessage);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serveClient(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serveClient(req, res, 'anthropic');
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
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
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
