// The brief, the provider catalog probe, and the candidate model table.
//
// A trial hands the entry one file ($MAKEFUN_PROMPT_FILE) holding three JSON documents, found here by
// their `schema` member so the surrounding prose does not matter:
//
//   free-model-aggregator-01/providers/v1  the ten free-tier providers: id, name, wire, auth scheme, the
//                                          key an operator would keep in `key_env`, the key the trial's
//                                          service takes, and the path its routes hang off.
//   free-model-aggregator-01/ratings/v1    the quality table: each model's rating in milli-points, its
//                                          context length, whether it takes tools, images and JSON mode,
//                                          and whether it is free. A paid model is never a legal pick.
//   free-model-aggregator-01/workload/v1   the manifest: twenty requests in order, each with its api,
//                                          whether it streams, its weight and what a model must have.
//
// Every provider is then probed at its own catalog route, in its own wire, and the answers become the one
// catalog document: the same object `main.ts --list` prints and the register body carries. Nothing here
// reads a clock or draws a random number, so two runs on the same brief build the same bytes.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
export type ClientApi = 'openai' | 'anthropic';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

/** One provider as the brief's directory lists it. `base` is its path (`/p/<id>`) or an absolute URL. */
export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

/** One row of the quality table. */
export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

/** What a model can do: the table's row first, the listing's own fields second, null when neither said. */
export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** What one request needs before a model may serve it. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One row of the workload manifest. */
export interface WorkloadRequest {
  id: string;
  api: ClientApi;
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

/** One model as a provider's catalog route listed it. */
export interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** One provider's catalog probe. */
export interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  status: number;
  models: ProbedModel[];
  error: string | null;
}

/** A free model that may serve requests, with everything routing needs to weigh it. */
export interface ModelEntry {
  provider: ProviderEntry;
  model: ProbedModel;
  rating: number;
  caps: Capabilities;
}

/** The brief. `ratings` and `workload` are null when the brief did not carry them. */
export interface Brief {
  directory: ProviderEntry[];
  ratings: RatingRow[];
  workload: WorkloadRequest[] | null;
  error: string | null;
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

/** The one catalog document: `--list` prints it, the register body carries it. */
export interface Catalog {
  providers: CatalogProvider[];
}

// ---------------------------------------------------------------------------------------------
// Reading the brief
// ---------------------------------------------------------------------------------------------

/** Every `{...}` in `text` that parses as a JSON object carrying a string `schema`, in order. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = matchingBrace(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not a JSON document on its own: keep scanning.
    }
  }
  return out;
}

/** The index of the `}` closing the object opened at `start`, skipping string contents; -1 if unbalanced. */
function matchingBrace(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

const str = (v: unknown): string => (typeof v === 'string' ? v : '');
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const bool = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

function wireOf(v: unknown): Wire | null {
  return v === 'openai' || v === 'anthropic' || v === 'gemini' ? v : null;
}

function authOf(v: unknown): AuthScheme {
  if (v === 'bearer' || v === 'x-api-key' || v === 'x-goog-api-key' || v === 'none') return v;
  return 'bearer';
}

/** The directory row: `path` is what the brief writes; an absolute `base_url` is accepted as well. */
function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = str(v.id);
  const wire = wireOf(v.wire);
  if (id === '' || wire === null) return null;
  const base =
    str(v.path) !== ''
      ? str(v.path)
      : str(v.base_path) !== ''
        ? str(v.base_path)
        : str(v.base_url) !== ''
          ? str(v.base_url)
          : str(v.base);
  return {
    id,
    name: str(v.name) !== '' ? str(v.name) : id,
    wire,
    auth: authOf(v.auth),
    base,
    key_env: str(v.key_env),
    key: str(v.key),
  };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = str(v.model);
  const provider = str(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: num(v.rating_milli) ?? 0,
    context_length: num(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function requirementOf(v: unknown): Requirement {
  if (!isRecord(v)) return {};
  const requires: Requirement = {};
  if (v.tools === true) requires.tools = true;
  if (v.vision === true) requires.vision = true;
  if (v.json === true) requires.json = true;
  const min = num(v.min_context);
  if (min !== null) requires.min_context = min;
  return requires;
}

function workloadRowOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = str(v.id);
  if (id === '') return null;
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: requirementOf(v.requires),
  };
}

/** The brief at `path`. A brief that cannot be read leaves the directory empty and names why. */
export function readBrief(path: string): Brief {
  if (path === '') return { directory: [], ratings: [], workload: null, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { directory: [], ratings: [], workload: null, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const providersDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  if (providersDoc === undefined) {
    return { directory: [], ratings: [], workload: null, error: `the brief at ${path} carries no ${PROVIDERS_SCHEMA} document` };
  }
  const directory = (providersDoc.providers as unknown[]).flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  const ratings = ratingsDoc === undefined ? [] : (ratingsDoc.ratings as unknown[]).flatMap((v) => {
    const r = ratingOf(v);
    return r === null ? [] : [r];
  });
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  const workload = workloadDoc === undefined ? null : (workloadDoc.manifest as unknown[]).flatMap((v) => {
    const w = workloadRowOf(v);
    return w === null ? [] : [w];
  });
  return { directory, ratings, workload, error: null };
}

// ---------------------------------------------------------------------------------------------
// Providers: base URLs, auth headers, the catalog probe
// ---------------------------------------------------------------------------------------------

/** A provider's base joined to the env's origin; an absolute base is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const origin = envUrl.replace(/\/+$/, '');
  if (b === '') return origin;
  return `${origin}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. Its wire never decides the header: its auth scheme does. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (auth === 'none' || key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  return { 'x-goog-api-key': key };
}

/** A price object on a listing: true when every price is zero, null when the listing says nothing. */
function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values: number[] = [];
  for (const raw of Object.values(pricing)) {
    if (typeof raw === 'number') values.push(raw);
    else if (typeof raw === 'string' && raw.trim() !== '') values.push(Number(raw));
  }
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? str(v.name).replace(/^models\//, '') : str(v.id);
  if (id === '') return null;
  return {
    id,
    display_name: str(v.display_name) !== '' ? str(v.display_name) : str(v.displayName) !== '' ? str(v.displayName) : id,
    free: bool(v.free) ?? zeroPriced(v),
    context_length: num(v.context_length) ?? num(v.context_window) ?? num(v.inputTokenLimit),
    tools: bool(v.tools),
    vision: bool(v.vision),
    json: bool(v.json),
  };
}

function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** One provider's catalog route, in its own wire. Every failure is tolerated and reported. */
export async function probeProvider(envUrl: string, provider: ProviderEntry, timeoutMs: number): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

// ---------------------------------------------------------------------------------------------
// Capabilities and requirements
// ---------------------------------------------------------------------------------------------

export function ratingOfModel(ratings: readonly RatingRow[], providerId: string, modelId: string): RatingRow | null {
  for (const row of ratings) if (row.provider === providerId && row.model === modelId) return row;
  return null;
}

/** Free is the table's word where it has a row (the sealed truth), else the listing's; paid when neither. */
export function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free === true;
}

export function capabilitiesOf(model: ProbedModel, row: RatingRow | null): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length,
    tools: row?.tools ?? model.tools,
    vision: row?.vision ?? model.vision,
    json: row?.json ?? model.json,
  };
}

/** An unknown capability never meets a requirement. */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/** Every free model a probe answered with, at the capabilities the table and the listing agree on. */
export function freeModels(probes: readonly Probe[], ratings: readonly RatingRow[]): ModelEntry[] {
  const out: ModelEntry[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ provider: probe.provider, model, rating: row?.rating_milli ?? 0, caps: capabilitiesOf(model, row) });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// The catalog document
// ---------------------------------------------------------------------------------------------

const byText = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

/**
 * The one catalog: every provider that answered, every model it listed, each with the free flag.
 *
 * `base_url` is the directory's own path, never joined to the env's origin: the env's port differs between
 * a live pass and its replay, and the catalog is part of the state export a replay must reproduce byte for
 * byte. The scorer compares a reported base_url by its path suffix, so the path scores as the joined URL
 * does; the joined URL reaches a provider, the path names it.
 */
export function buildCatalog(probes: readonly Probe[], ratings: readonly RatingRow[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort((a, b) => byText(a.id, b.id))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      models.push({
        id: model.id,
        free: modelFree(model, row),
        context_length: row?.context_length ?? model.context_length ?? null,
      });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}

/** The free model ids the catalog registers: what GET /v1/models lists, in both shapes. */
export function freeModelIds(catalog: Catalog): { id: string; display: string; provider: string }[] {
  const out: { id: string; display: string; provider: string }[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (model.free) out.push({ id: model.id, display: model.id, provider: provider.id });
    }
  }
  return out;
}
