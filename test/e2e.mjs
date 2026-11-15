// test/e2e.mjs: drives the whole entry end to end against a mock of the trial's env.
//
// It is the competition's protocol in miniature, on the three real wires:
//
//   POST /register  takes the port and the catalog, and answers the workload's size
//   GET  /work      says done false until every request has been sent, then done true
//   /p/<id>/...     four providers: an OpenAI one whose budget runs out after one call (429), an Anthropic
//                   one, a Gemini one, and a keyless OpenAI one that cannot take images or JSON
//   GET  /v1/models asked with both auth headers, expected to list the free ids the catalog registered
//
// and it then asserts what a served request has to look like: the echo the provider parsed comes back whole
// (system prompt, user text, image, tool names, max_tokens, stop sequences, JSON mode), the stop reason and
// the usage counts match the provider's, the stream event order is the one the client's wire defines, the
// catalog `--list` prints is the register body byte for byte, a refused first call still gets an answer from
// the next-best free model, and the entry exits 0 when the env says the work is done.
//
// Run: node test/e2e.mjs        (exit 0: every assertion passed)

import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const ENTRY = join(ROOT, 'main.ts');
const NODE = process.execPath;

const failures = [];
let checks = 0;
function check(name, pass, detail) {
  checks += 1;
  if (!pass) failures.push(`${name}: ${detail}`);
  process.stdout.write(`${pass ? 'ok  ' : 'FAIL'} ${name}${pass ? '' : `: ${detail}`}\n`);
}

const sha8 = (text) => createHash('sha256').update(text).digest('hex').slice(0, 8);
const sha8Bytes = (bytes) => createHash('sha256').update(bytes).digest('hex').slice(0, 8);

function imageSha(url) {
  if (typeof url !== 'string' || url === '') return null;
  const m = /^data:[^,]*;base64,(.*)$/s.exec(url);
  return m === null ? sha8(url) : sha8Bytes(Buffer.from(m[1], 'base64'));
}

function blocksText(content) {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.filter((b) => b !== null && typeof b === 'object' && b.type === 'text' && typeof b.text === 'string').map((b) => b.text).join('\n');
}

/** The echo a provider computes from a client body: exactly the fields the trial compares. */
function echoOfOpenAI(body) {
  const messages = Array.isArray(body.messages) ? body.messages.filter((m) => m !== null && typeof m === 'object') : [];
  const system = [];
  let userText = '';
  let imageSha8 = null;
  for (const m of messages) {
    if (m.role === 'system') system.push(blocksText(m.content));
    else if (m.role === 'user') {
      userText = blocksText(m.content);
      if (Array.isArray(m.content)) {
        for (const b of m.content) {
          if (b !== null && typeof b === 'object' && b.type === 'image_url' && b.image_url !== null && typeof b.image_url === 'object') imageSha8 = imageSha(b.image_url.url) ?? imageSha8;
        }
      }
    }
  }
  const tools = [];
  if (Array.isArray(body.tools)) for (const t of body.tools) if (t?.function?.name !== undefined) tools.push(t.function.name);
  const format = body.response_format !== null && typeof body.response_format === 'object' ? body.response_format.type : undefined;
  const systemText = system.join('\n');
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  const stops = typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? body.stop : [];
  return {
    system_present: systemText !== '',
    system_sha8: systemText === '' ? null : sha8(systemText),
    user_sha8: sha8(userText),
    tools,
    max_tokens: maxTokens,
    stops,
    image_sha8: imageSha8,
    json_mode: format === 'json_object' || format === 'json_schema',
  };
}

function echoOfAnthropic(body) {
  const systemText = blocksText(body.system);
  const messages = Array.isArray(body.messages) ? body.messages.filter((m) => m !== null && typeof m === 'object') : [];
  let userText = '';
  let imageSha8 = null;
  for (const m of messages) {
    if (m.role !== 'user') continue;
    userText = blocksText(m.content);
    if (Array.isArray(m.content)) {
      for (const b of m.content) {
        if (b === null || typeof b !== 'object' || b.type !== 'image' || b.source === null || typeof b.source !== 'object') continue;
        if (b.source.type === 'base64' && typeof b.source.data === 'string') imageSha8 = sha8Bytes(Buffer.from(b.source.data, 'base64'));
        else if (b.source.type === 'url') imageSha8 = imageSha(b.source.url) ?? imageSha8;
      }
    }
  }
  const tools = [];
  if (Array.isArray(body.tools)) for (const t of body.tools) if (typeof t.name === 'string') tools.push(t.name);
  return {
    system_present: systemText !== '',
    system_sha8: systemText === '' ? null : sha8(systemText),
    user_sha8: sha8(userText),
    tools,
    max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : null,
    stops: typeof body.stop_sequences === 'string' ? [body.stop_sequences] : Array.isArray(body.stop_sequences) ? body.stop_sequences : [],
    image_sha8: imageSha8,
    json_mode: false,
  };
}

function echoOfGemini(body) {
  const systemText = body.systemInstruction !== null && typeof body.systemInstruction === 'object' ? blocksOfParts(body.systemInstruction.parts) : '';
  const contents = Array.isArray(body.contents) ? body.contents.filter((c) => c !== null && typeof c === 'object') : [];
  let userText = '';
  let imageSha8 = null;
  for (const c of contents) {
    if (c.role === 'model') continue;
    userText = blocksOfParts(c.parts);
    for (const p of Array.isArray(c.parts) ? c.parts : []) {
      if (p === null || typeof p !== 'object') continue;
      if (p.inlineData !== undefined) imageSha8 = sha8Bytes(Buffer.from(p.inlineData.data, 'base64'));
      else if (p.fileData !== undefined) imageSha8 = imageSha(p.fileData.fileUri) ?? imageSha8;
    }
  }
  const tools = [];
  for (const t of Array.isArray(body.tools) ? body.tools : []) for (const fn of Array.isArray(t.functionDeclarations) ? t.functionDeclarations : []) if (typeof fn.name === 'string') tools.push(fn.name);
  const config = body.generationConfig !== null && typeof body.generationConfig === 'object' ? body.generationConfig : {};
  return {
    system_present: systemText !== '',
    system_sha8: systemText === '' ? null : sha8(systemText),
    user_sha8: sha8(userText),
    tools,
    max_tokens: typeof config.maxOutputTokens === 'number' ? config.maxOutputTokens : null,
    stops: Array.isArray(config.stopSequences) ? config.stopSequences : [],
    image_sha8: imageSha8,
    json_mode: config.responseMimeType === 'application/json',
  };
}

function blocksOfParts(parts) {
  if (!Array.isArray(parts)) return '';
  return parts.filter((p) => p !== null && typeof p === 'object' && typeof p.text === 'string').map((p) => p.text).join('\n');
}

// ---------------------------------------------------------------------------------------------
// The fixture: four providers, one of which runs out of budget after a single call
// ---------------------------------------------------------------------------------------------

const PROVIDERS = [
  {
    id: 'fast',
    name: 'Fast',
    wire: 'openai',
    auth: 'bearer',
    key: 'sim-fast',
    path: '/p/fast',
    budget: 1,
    failure_status: 429,
    failure_code: 'rate_limit_exceeded',
    models: [
      { id: 'fast-top', free: true, rating_milli: 1500, context_length: 400000, tools: true, vision: true, json: true },
      { id: 'fast-paid', free: false, rating_milli: 1520, context_length: 400000, tools: true, vision: true, json: true },
    ],
  },
  {
    id: 'claude',
    name: 'Claude',
    wire: 'anthropic',
    auth: 'x-api-key',
    key: 'sim-claude',
    path: '/p/claude',
    budget: 30,
    failure_status: 429,
    failure_code: 'rate_limit_error',
    models: [{ id: 'claude-mid', free: true, rating_milli: 1400, context_length: 400000, tools: true, vision: true, json: true }],
  },
  {
    id: 'gemi',
    name: 'Gemi',
    wire: 'gemini',
    auth: 'x-goog-api-key',
    key: 'sim-gemi',
    path: '/p/gemi',
    budget: 30,
    failure_status: 402,
    failure_code: 'payment_required',
    models: [{ id: 'gemi-mid', free: true, rating_milli: 1300, context_length: 1000000, tools: true, vision: true, json: true }],
  },
  {
    id: 'open',
    name: 'Open',
    wire: 'openai',
    auth: 'none',
    key: '',
    path: '/p/open',
    budget: 30,
    failure_status: 429,
    failure_code: 'rate_limit_exceeded',
    models: [{ id: 'open-low', free: true, rating_milli: 1100, context_length: 32000, tools: true, vision: false, json: false }],
  },
];

const IMAGE_B64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';

const TOOLS_OPENAI = [
  { type: 'function', function: { name: 'lookup_customer', description: 'Look a customer up.', parameters: { type: 'object', properties: { account_id: { type: 'string' } }, required: ['account_id'] } } },
];
const TOOLS_ANTHROPIC = [{ name: 'lookup_customer', description: 'Look a customer up.', input_schema: { type: 'object', properties: { account_id: { type: 'string' } }, required: ['account_id'] } }];

const WORKLOAD = [
  {
    id: 'q1',
    api: 'openai',
    stream: false,
    requires: { tools: true, vision: true, json: true },
    body: {
      model: 'auto',
      messages: [{ role: 'system', content: 'You are careful.' }, { role: 'user', content: [{ type: 'text', text: 'Read the receipt.' }, { type: 'image_url', image_url: { url: `data:image/png;base64,${IMAGE_B64}` } }] }],
      max_tokens: 64,
      temperature: 0.5,
      tools: TOOLS_OPENAI,
      response_format: { type: 'json_object' },
    },
  },
  {
    id: 'q2',
    api: 'openai',
    stream: true,
    requires: { tools: true, vision: true },
    body: {
      model: 'auto',
      stream: true,
      messages: [{ role: 'user', content: [{ type: 'text', text: 'Summarise the chart.' }, { type: 'image_url', image_url: { url: `data:image/png;base64,${IMAGE_B64}` } }] }],
      max_tokens: 96,
      tools: TOOLS_OPENAI,
      stop: ['\n\n###'],
    },
  },
  {
    id: 'q3',
    api: 'anthropic',
    stream: false,
    requires: { tools: true, min_context: 200000 },
    body: {
      model: 'auto',
      system: 'You are careful.',
      messages: [{ role: 'user', content: 'Look up account 4407.' }],
      max_tokens: 128,
      tools: TOOLS_ANTHROPIC,
      stop_sequences: ['\n\nHuman:'],
    },
  },
  {
    id: 'q4',
    api: 'anthropic',
    stream: true,
    requires: {},
    body: { model: 'auto', stream: true, messages: [{ role: 'user', content: [{ type: 'text', text: 'Restate the claim.' }, { type: 'image', source: { type: 'base64', media_type: 'image/png', data: IMAGE_B64 } }] }], max_tokens: 256 },
  },
];

// ---------------------------------------------------------------------------------------------
// The mock env
// ---------------------------------------------------------------------------------------------

const calls = [];
const budgets = new Map(PROVIDERS.map((p) => [p.id, p.budget]));
let registered = null;
let sent = 0;

const sse = (event, data) => `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;

function pieces(text, size = 13) {
  const out = [];
  for (let at = 0; at < text.length; at += size) out.push(text.slice(at, at + size));
  return out.length === 0 ? [''] : out;
}

function answerFor(wire, model, echo, stream) {
  const text = `ANSWER ECHO${JSON.stringify(echo)}`;
  const usage = { input: 1 + Math.floor((echo.user_sha8.length + String(echo.system_sha8).length) / 4), output: 1 + Math.floor(text.length / 4) };
  const tokens = { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output };
  if (wire === 'openai') {
    if (!stream) {
      return { status: 200, json: { id: 'chatcmpl-mock', object: 'chat.completion', created: 1767225600, model, choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: 'stop', logprobs: null }], usage: tokens } };
    }
    const chunk = (delta, finish) => ({ id: 'chatcmpl-mock', object: 'chat.completion.chunk', created: 1767225600, model, choices: [{ index: 0, delta, finish_reason: finish }] });
    let body = sse(null, chunk({ role: 'assistant', content: '' }, null));
    for (const p of pieces(text)) body += sse(null, chunk({ content: p }, null));
    body += sse(null, chunk({}, 'stop'));
    body += sse(null, { id: 'chatcmpl-mock', object: 'chat.completion.chunk', created: 1767225600, model, choices: [], usage: tokens });
    body += 'data: [DONE]\n\n';
    return { status: 200, sse: body };
  }
  if (wire === 'anthropic') {
    if (!stream) {
      return { status: 200, json: { id: 'msg-mock', type: 'message', role: 'assistant', model, content: [{ type: 'text', text }], stop_reason: 'end_turn', stop_sequence: null, usage: { input_tokens: usage.input, output_tokens: usage.output } } };
    }
    let body = sse('message_start', { type: 'message_start', message: { id: 'msg-mock', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: usage.input, output_tokens: 0 } } });
    body += sse('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
    for (const p of pieces(text)) body += sse('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: p } });
    body += sse('content_block_stop', { type: 'content_block_stop', index: 0 });
    body += sse('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: usage.output } });
    body += sse('message_stop', { type: 'message_stop' });
    return { status: 200, sse: body };
  }
  const metadata = { promptTokenCount: usage.input, candidatesTokenCount: usage.output, totalTokenCount: usage.input + usage.output };
  if (!stream) {
    return { status: 200, json: { candidates: [{ content: { parts: [{ text }], role: 'model' }, finishReason: 'STOP', index: 0 }], usageMetadata: metadata, modelVersion: model } };
  }
  let body = '';
  for (const p of pieces(text)) body += sse(null, { candidates: [{ content: { parts: [{ text: p }], role: 'model' }, index: 0 }] });
  body += sse(null, { candidates: [{ content: { parts: [{ text: '' }], role: 'model' }, finishReason: 'STOP', index: 0 }], usageMetadata: metadata, modelVersion: model });
  return { status: 200, sse: body };
}

function errorBody(wire, status, code, message) {
  if (wire === 'openai') return { error: { message, type: status === 429 ? 'rate_limit_error' : 'invalid_request_error', param: null, code } };
  if (wire === 'anthropic') return { type: 'error', error: { type: code, message } };
  return { error: { code: status, message, status: code } };
}

function authOk(provider, headers, query) {
  if (provider.auth === 'none') return true;
  if (provider.auth === 'bearer') return headers.authorization === `Bearer ${provider.key}`;
  if (provider.auth === 'x-api-key') return headers['x-api-key'] === provider.key;
  return headers['x-goog-api-key'] === provider.key || query.get('key') === provider.key;
}

function handleProvider(provider, method, path, headers, url, raw) {
  const sub = path.slice(provider.path.length);
  const json = (status, value) => ({ status, headers: { 'content-type': 'application/json' }, body: Buffer.from(`${JSON.stringify(value)}\n`) });

  if (sub === '/v1/models' && method === 'GET') {
    if (!authOk(provider, headers, url.searchParams)) return json(401, errorBody(provider.wire, 401, 'invalid_api_key', 'the key is missing or wrong'));
    if (provider.wire === 'openai') {
      return json(200, { object: 'list', data: provider.models.map((m) => ({ id: m.id, object: 'model', created: 1767225600, owned_by: provider.id, free: m.free, context_length: m.context_length, tools: m.tools, vision: m.vision, json: m.json })) });
    }
    if (provider.wire === 'anthropic') {
      return json(200, { data: provider.models.map((m) => ({ type: 'model', id: m.id, display_name: m.id, created_at: '2026-01-01T00:00:00Z', free: m.free, context_length: m.context_length, tools: m.tools, vision: m.vision, json: m.json })), has_more: false });
    }
    return json(200, { models: provider.models.map((m) => ({ name: `models/${m.id}`, displayName: m.id, inputTokenLimit: m.context_length, free: m.free, context_length: m.context_length, tools: m.tools, vision: m.vision, json: m.json })) });
  }
  if (sub === '/v1beta/models' && method === 'GET') {
    if (!authOk(provider, headers, url.searchParams)) return json(401, errorBody(provider.wire, 401, 'UNAUTHENTICATED', 'the key is missing or wrong'));
    return json(200, { models: provider.models.map((m) => ({ name: `models/${m.id}`, displayName: m.id, inputTokenLimit: m.context_length, free: m.free, context_length: m.context_length, tools: m.tools, vision: m.vision, json: m.json })) });
  }

  const call = provider.wire === 'gemini' ? /^\/v1beta\/models\/([^/:]+):(generateContent|streamGenerateContent)$/.exec(sub) : null;
  const isInfer = provider.wire === 'gemini' ? call !== null : sub === (provider.wire === 'openai' ? '/v1/chat/completions' : '/v1/messages');
  if (!isInfer || method !== 'POST') return json(404, { error: 'not_found', message: `no route ${method} ${path}` });
  if (!authOk(provider, headers, url.searchParams)) {
    calls.push({ provider: provider.id, model: null, status: 401 });
    return json(401, errorBody(provider.wire, 401, 'invalid_api_key', 'the key is missing or wrong'));
  }

  let body = null;
  try {
    body = JSON.parse(raw.toString('utf8'));
  } catch {
    body = null;
  }
  if (body === null || typeof body !== 'object' || Array.isArray(body)) {
    calls.push({ provider: provider.id, model: null, status: 400 });
    return json(400, errorBody(provider.wire, 400, 'invalid_request', 'the request body is not one JSON object'));
  }
  const model = provider.wire === 'gemini' ? decodeURIComponent(call[1]) : body.model;
  const row = provider.models.find((m) => m.id === model);
  if (row === undefined || !row.free) {
    calls.push({ provider: provider.id, model: model ?? null, status: 400 });
    return json(400, errorBody(provider.wire, 400, 'invalid_model', `${String(model)} is not a free model here`));
  }
  const used = (budgets.get(provider.id) ?? 0) - 1;
  budgets.set(provider.id, used);
  if (used < 0) {
    calls.push({ provider: provider.id, model, status: provider.failure_status });
    return json(provider.failure_status, errorBody(provider.wire, provider.failure_status, provider.failure_code, `${provider.name}'s free-tier budget is spent`));
  }
  const echo = provider.wire === 'openai' ? echoOfOpenAI(body) : provider.wire === 'anthropic' ? echoOfAnthropic(body) : echoOfGemini(body);
  const stream = provider.wire === 'gemini' ? call[2] === 'streamGenerateContent' : body.stream === true;
  const answer = answerFor(provider.wire, model, echo, stream);
  const stop = provider.wire === 'openai' ? 'stop' : provider.wire === 'anthropic' ? 'end_turn' : 'STOP';
  const usage = { input: 1 + Math.floor((echo.user_sha8.length + String(echo.system_sha8).length) / 4), output: 0 };
  const text = `ANSWER ECHO${JSON.stringify(echo)}`;
  usage.output = 1 + Math.floor(text.length / 4);
  calls.push({ provider: provider.id, model, status: 200, stop, usage });
  if (answer.sse !== undefined) return { status: 200, headers: { 'content-type': 'text/event-stream; charset=utf-8' }, body: Buffer.from(answer.sse) };
  return json(answer.status, answer.json);
}

async function readBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  return Buffer.concat(chunks);
}

function jsonReply(status, value) {
  return { status, headers: { 'content-type': 'application/json' }, body: Buffer.from(`${JSON.stringify(value)}\n`) };
}

const env = createServer((req, res) => {
  void (async () => {
    const url = new URL(req.url ?? '/', 'http://env');
    const path = url.pathname;
    const method = req.method ?? 'GET';
    const raw = await readBody(req);
    let reply;
    if (path === '/register' && method === 'POST') {
      const value = JSON.parse(raw.toString('utf8'));
      registered = value;
      reply = { status: 200, headers: { 'content-type': 'application/json' }, body: Buffer.from(`${JSON.stringify({ ok: true, requests: WORKLOAD.length })}\n`) };
    } else if (path === '/work' && method === 'GET') {
      const done = sent >= WORKLOAD.length;
      reply = jsonReply(200, done ? { done: true } : { done: false, next: WORKLOAD[sent].id, pending: WORKLOAD.length - sent });
    } else if (path.startsWith('/p/')) {
      const id = path.slice('/p/'.length).split('/')[0];
      const provider = PROVIDERS.find((p) => p.id === id);
      reply = provider === undefined ? jsonReply(404, { error: 'not_found', message: `no provider ${id}` }) : handleProvider(provider, method, path, req.headers, url, raw);
    } else {
      reply = jsonReply(404, { error: 'not_found', message: `no route ${method} ${path}` });
    }
    res.writeHead(reply.status, { ...reply.headers, 'content-length': String(reply.body.length) });
    res.end(reply.body);
  })();
});

// ---------------------------------------------------------------------------------------------
// The client's checks, as the trial's env runs them
// ---------------------------------------------------------------------------------------------

function parseSseFrames(text) {
  const frames = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event = null;
    const data = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseOpenAIJson(text) {
  const v = JSON.parse(text);
  const choice = v.choices?.[0];
  return {
    text: typeof choice.message.content === 'string' ? choice.message.content : blocksText(choice.message.content),
    stopReason: choice.finish_reason,
    usage: v.usage === undefined ? null : { input: v.usage.prompt_tokens, output: v.usage.completion_tokens },
    shapeOk: choice.message.role === 'assistant' && typeof choice.finish_reason === 'string',
  };
}

function parseAnthropicJson(text) {
  const v = JSON.parse(text);
  return {
    text: blocksText(v.content),
    stopReason: v.stop_reason,
    usage: v.usage === undefined ? null : { input: v.usage.input_tokens, output: v.usage.output_tokens },
    shapeOk: v.type === 'message' && Array.isArray(v.content),
  };
}

function parseOpenAIStream(text) {
  const frames = parseSseFrames(text);
  let body = '';
  let stopReason = null;
  let usage = null;
  let sawFinish = false;
  let sawDone = false;
  let order = null;
  for (let i = 0; i < frames.length; i++) {
    const payload = frames[i].data.trim();
    if (payload === '[DONE]') {
      if (i !== frames.length - 1) order ??= 'events followed [DONE]';
      sawDone = true;
      continue;
    }
    if (sawDone) {
      order ??= 'an event followed [DONE]';
      continue;
    }
    const chunk = JSON.parse(payload);
    if (chunk.usage !== undefined && chunk.usage !== null) usage = { input: chunk.usage.prompt_tokens, output: chunk.usage.completion_tokens };
    if (!Array.isArray(chunk.choices) || chunk.choices.length === 0) continue;
    const choice = chunk.choices[0];
    if (typeof choice.delta?.content === 'string' && choice.delta.content !== '') {
      if (sawFinish) order ??= 'a content delta followed the finish chunk';
      body += choice.delta.content;
    }
    if (typeof choice.finish_reason === 'string') {
      if (sawFinish) order ??= 'two finish chunks';
      sawFinish = true;
      stopReason = choice.finish_reason;
    }
  }
  if (!sawFinish) order ??= 'no finish chunk';
  if (!sawDone) order ??= 'no [DONE]';
  return { text: body, stopReason, usage, shapeOk: true, order };
}

function parseAnthropicStream(text) {
  const frames = parseSseFrames(text);
  let body = '';
  let stopReason = null;
  let input = null;
  let output = null;
  let started = false;
  let sawDelta = false;
  let stopped = false;
  let order = null;
  const open = new Set();
  const closed = new Set();
  for (let i = 0; i < frames.length; i++) {
    const frame = frames[i];
    if (stopped) {
      order ??= 'an event followed message_stop';
      continue;
    }
    if (frame.event === null) {
      order ??= 'an event with no name';
      continue;
    }
    const data = JSON.parse(frame.data);
    if (frame.event === 'message_start') {
      if (started) order ??= 'message_start twice';
      else if (i !== 0) order ??= 'message_start was not first';
      started = true;
      input = data.message?.usage?.input_tokens ?? null;
    } else if (frame.event === 'content_block_start') {
      if (open.has(data.index) || closed.has(data.index)) order ??= `block ${data.index} started twice`;
      open.add(data.index);
    } else if (frame.event === 'content_block_delta') {
      if (!open.has(data.index)) order ??= `delta for block ${data.index}, which is not open`;
      if (data.delta?.type === 'text_delta') body += data.delta.text;
    } else if (frame.event === 'content_block_stop') {
      if (!open.has(data.index)) order ??= `stop for block ${data.index}, which is not open`;
      open.delete(data.index);
      closed.add(data.index);
    } else if (frame.event === 'message_delta') {
      if (open.size > 0) order ??= 'message_delta with a block open';
      sawDelta = true;
      if (typeof data.delta?.stop_reason === 'string') stopReason = data.delta.stop_reason;
      output = data.usage?.output_tokens ?? null;
    } else if (frame.event === 'message_stop') {
      if (!sawDelta) order ??= 'message_stop before message_delta';
      stopped = true;
    } else {
      order ??= `unknown event ${frame.event}`;
    }
  }
  if (!started) order ??= 'no message_start';
  if (!stopped) order ??= 'no message_stop';
  return { text: body, stopReason, usage: input === null && output === null ? null : { input, output }, shapeOk: true, order };
}

const STOPS = new Map([
  ['stop', 0], ['end_turn', 0], ['STOP', 0],
  ['length', 1], ['max_tokens', 1], ['MAX_TOKENS', 1],
  ['tool_calls', 2], ['function_call', 2], ['tool_use', 2],
  ['content_filter', 3], ['refusal', 3], ['SAFETY', 3],
]);

function echoFromText(text) {
  if (!text.startsWith('ANSWER ECHO')) return null;
  try {
    return JSON.parse(text.slice('ANSWER ECHO'.length));
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------------------------

function waitForLayout() {
  return new Promise((resolve) => env.listen(0, '127.0.0.1', () => resolve(env.address().port)));
}

function runEntry(args, envVars, onSpawn) {
  const child = spawn(NODE, [ENTRY, ...args], { env: { ...process.env, ...envVars }, stdio: ['ignore', 'pipe', 'pipe'] });
  let stdout = '';
  let stderr = '';
  child.stdout.on('data', (d) => {
    stdout += d.toString();
  });
  child.stderr.on('data', (d) => {
    stderr += d.toString();
  });
  if (onSpawn !== undefined) onSpawn(child);
  const exited = new Promise((resolve) => child.on('exit', (code) => resolve({ code, stdout, stderr })));
  return { child, exited, output: () => ({ stdout, stderr }) };
}

async function post(path, headers, body) {
  const res = await fetch(`http://127.0.0.1:${clientPort}${path}`, { method: 'POST', headers: { 'content-type': 'application/json', ...headers }, body: JSON.stringify(body) });
  return { status: res.status, text: await res.text() };
}

let clientPort = 0;

async function main() {
  const workDir = mkdtempSync(join(tmpdir(), 'ftg-e2e-'));
  try {
    const briefFile = join(workDir, 'prompt.txt');
    writeFileSync(
      briefFile,
      [
        'free-model-aggregator-01 end-to-end fixture.',
        JSON.stringify({ schema: 'free-model-aggregator-01/providers/v1', providers: PROVIDERS.map((p) => ({ id: p.id, name: p.name, wire: p.wire, auth: p.auth, key_env: `${p.id.toUpperCase()}_KEY`, key: p.key, path: p.path })) }),
        JSON.stringify({
          schema: 'free-model-aggregator-01/ratings/v1',
          ratings: PROVIDERS.flatMap((p) => p.models.map((m) => ({ model: m.id, provider: p.id, rating_milli: m.rating_milli, context_length: m.context_length, tools: m.tools, vision: m.vision, json: m.json, free: m.free }))),
          best: WORKLOAD.map((r) => ({ id: r.id, rating_milli: 1500 })),
        }),
        JSON.stringify({ schema: 'free-model-aggregator-01/workload/v1', manifest: WORKLOAD.map(({ id, api, stream, requires }) => ({ id, api, stream, weight: 1, requires })), bodies: {} }),
        '',
      ].join('\n'),
    );

    const envPort = await waitForLayout();
    const envUrl = `http://127.0.0.1:${envPort}`;
    const vars = { MAKEFUN_ENV_URL: envUrl, MAKEFUN_PROMPT_FILE: briefFile, MAKEFUN_TRIAL_SEED: '1234' };

    // 1. --list, outside any server: the same catalog, from the same brief.
    const list = runEntry(['--list'], vars);
    const listed = await list.exited;
    check('--list exits 0', listed.code === 0, `exit ${listed.code}: ${listed.stderr.slice(0, 400)}`);
    let listedDoc = null;
    try {
      listedDoc = JSON.parse(listed.stdout);
    } catch {
      listedDoc = null;
    }
    check('--list prints one JSON document', listedDoc !== null, `stdout was ${JSON.stringify(listed.stdout.slice(0, 200))}`);
    check(
      '--list names every provider with its path and wire',
      listedDoc !== null && listedDoc.providers.length === PROVIDERS.length && listedDoc.providers.every((p, i) => p.id === PROVIDERS[i].id && p.base_url === PROVIDERS[i].path && p.wire === PROVIDERS[i].wire),
      JSON.stringify(listedDoc?.providers?.map((p) => [p.id, p.base_url, p.wire])),
    );
    const paidRegistered = listedDoc?.providers?.flatMap((p) => p.models.filter((m) => !m.free).map((m) => m.id)) ?? [];
    check('--list marks the paid model paid', paidRegistered.length === 1 && paidRegistered[0] === 'fast-paid', JSON.stringify(paidRegistered));

    // 2. standalone: no env, no brief. It must still come up and answer, as `npm start` does on a laptop.
    const standalone = runEntry([], { MAKEFUN_ENV_URL: '', MAKEFUN_PROMPT_FILE: '', PORT: '0' });
    let standalonePort = null;
    for (let i = 0; i < 200 && standalonePort === null; i++) {
      const m = /listening on 0\.0\.0\.0:(\d+)/.exec(standalone.output().stderr);
      if (m !== null) standalonePort = Number(m[1]);
      else await new Promise((r) => setTimeout(r, 25));
    }
    check('npm start with no trial env comes up', standalonePort !== null, `never listened: ${standalone.output().stderr.slice(0, 400)}`);
    if (standalonePort !== null) {
      const list = await fetch(`http://127.0.0.1:${standalonePort}/v1/models`, { headers: { authorization: 'Bearer client-key' } });
      const body = await list.json();
      check('standalone GET /v1/models answers 200 with an empty list', list.status === 200 && Array.isArray(body.data) && body.data.length === 0, `HTTP ${String(list.status)} ${JSON.stringify(body).slice(0, 120)}`);
      const chat = await fetch(`http://127.0.0.1:${standalonePort}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', authorization: 'Bearer client-key' },
        body: JSON.stringify({ model: 'auto', messages: [{ role: 'user', content: 'hi' }] }),
      });
      const refusal = await chat.json();
      check('standalone chat answers a 503 that names the missing env', chat.status === 503 && /MAKEFUN_ENV_URL/.test(refusal?.error?.message ?? ''), `HTTP ${String(chat.status)} ${JSON.stringify(refusal).slice(0, 160)}`);
      check('standalone still running (it is a server, not a one-shot)', standalone.child.exitCode === null, `exited ${String(standalone.child.exitCode)}`);
      standalone.child.kill('SIGTERM');
      await standalone.exited;
    }

    // 3. the server, against the mock env.
    const server = runEntry([], vars);
    for (let i = 0; i < 400 && registered === null; i++) await new Promise((r) => setTimeout(r, 25));
    check('the entry registers', registered !== null, `no POST /register in 10 s: ${server.output().stderr.slice(0, 400)}`);
    if (registered === null) return;
    check('register speaks protocol_version 1', registered.protocol_version === 1, String(registered.protocol_version));
    check('register names a port', Number.isSafeInteger(registered.port) && registered.port > 0, String(registered.port));
    clientPort = registered.port;
    check('the registered catalog is the one --list printed, byte for byte', JSON.stringify(registered.catalog) === JSON.stringify(listedDoc), 'the two documents differ');

    // 3. GET /v1/models in both shapes: exactly the free ids the catalog registered.
    const freeIds = (registered.catalog.providers ?? []).flatMap((p) => (p.models ?? []).filter((m) => m.free).map((m) => m.id));
    const asOpenAI = await fetch(`http://127.0.0.1:${clientPort}/v1/models`, { headers: { authorization: 'Bearer client-key' } });
    const asAnthropic = await fetch(`http://127.0.0.1:${clientPort}/v1/models`, { headers: { 'x-api-key': 'client-key', 'anthropic-version': '2023-06-01' } });
    const openAIBody = await asOpenAI.json();
    const anthropicBody = await asAnthropic.json();
    check('GET /v1/models answers OpenAI shape for Bearer', Array.isArray(openAIBody.data) && openAIBody.object === 'list', JSON.stringify(openAIBody).slice(0, 200));
    check('GET /v1/models answers Anthropic shape for x-api-key', Array.isArray(anthropicBody.data) && anthropicBody.data.every((m) => m.type === 'model'), JSON.stringify(anthropicBody).slice(0, 200));
    const listedIds = new Set(openAIBody.data.map((m) => m.id));
    const wantedIds = new Set(freeIds);
    check('both listings are exactly the registered free ids', listedIds.size === wantedIds.size && [...wantedIds].every((id) => listedIds.has(id)), `${[...listedIds].join(',')} vs ${[...wantedIds].join(',')}`);

    // 4. the workload, one request at a time, as the client sends it.
    for (const request of WORKLOAD) {
      const before = calls.length;
      const headers = request.api === 'openai' ? { authorization: 'Bearer client-key' } : { 'x-api-key': 'client-key', 'anthropic-version': '2023-06-01' };
      if (request.stream) headers.accept = 'text/event-stream';
      const answer = await post(request.api === 'openai' ? '/v1/chat/completions' : '/v1/messages', headers, request.body);
      sent += 1;
      const requestCalls = calls.slice(before);
      const ok = requestCalls.filter((c) => c.status === 200 && c.model !== null);
      const provider = ok.length > 0 ? ok[ok.length - 1] : null;

      check(`${request.id}: answered 2xx`, answer.status >= 200 && answer.status < 300, `HTTP ${answer.status}: ${answer.text.slice(0, 300)}`);
      const parsed = request.stream
        ? request.api === 'openai'
          ? parseOpenAIStream(answer.text)
          : parseAnthropicStream(answer.text)
        : request.api === 'openai'
          ? parseOpenAIJson(answer.text)
          : parseAnthropicJson(answer.text);
      check(`${request.id}: shape`, parsed.shapeOk, JSON.stringify(parsed).slice(0, 300));
      if (request.stream) check(`${request.id}: stream event order`, parsed.order === null, String(parsed.order));

      const echo = echoFromText(parsed.text);
      const expected = request.api === 'openai' ? echoOfOpenAI(request.body) : echoOfAnthropic(request.body);
      check(`${request.id}: the provider's echo came back whole`, echo !== null, `text was ${JSON.stringify(parsed.text.slice(0, 120))}`);
      if (echo !== null) {
        check(`${request.id}: echo_user`, echo.user_sha8 === expected.user_sha8, `${echo.user_sha8} vs ${expected.user_sha8}`);
        check(`${request.id}: echo_max_tokens`, echo.max_tokens === expected.max_tokens, `${String(echo.max_tokens)} vs ${String(expected.max_tokens)}`);
        if (expected.system_present) check(`${request.id}: echo_system`, echo.system_present && echo.system_sha8 === expected.system_sha8, `${String(echo.system_sha8)} vs ${String(expected.system_sha8)}`);
        if (expected.tools.length > 0) check(`${request.id}: echo_tools`, JSON.stringify(echo.tools) === JSON.stringify(expected.tools), `${JSON.stringify(echo.tools)} vs ${JSON.stringify(expected.tools)}`);
        if (expected.stops.length > 0) check(`${request.id}: echo_stop`, JSON.stringify(echo.stops) === JSON.stringify(expected.stops), `${JSON.stringify(echo.stops)} vs ${JSON.stringify(expected.stops)}`);
        if (expected.image_sha8 !== null) check(`${request.id}: echo_image`, echo.image_sha8 === expected.image_sha8, `${String(echo.image_sha8)} vs ${String(expected.image_sha8)}`);
        if (expected.json_mode) check(`${request.id}: echo_json`, echo.json_mode === true, `json_mode was ${String(echo.json_mode)}`);
      }
      check(`${request.id}: stop reason matches the provider's`, provider !== null && STOPS.get(parsed.stopReason) === STOPS.get(provider.stop), `${String(parsed.stopReason)} vs ${String(provider?.stop)}`);
      const usageOk =
        provider !== null &&
        parsed.usage !== null &&
        (request.api === 'anthropic' && request.stream ? parsed.usage.output === provider.usage.output : parsed.usage.input === provider.usage.input && parsed.usage.output === provider.usage.output);
      check(`${request.id}: usage is the provider's`, usageOk, `${JSON.stringify(parsed.usage)} vs ${JSON.stringify(provider?.usage ?? null)}`);
      check(`${request.id}: one provider call answered it`, ok.length === 1, `${ok.length} successful calls`);
      check(`${request.id}: served by a free model`, provider !== null, 'no provider answered');
    }

    // 5. failover: fast allows one call, so later requests must have been refused there and still served.
    const refused = calls.filter((c) => c.status === 429 || c.status === 402 || c.status >= 500);
    check('a provider refused at least one call', refused.length > 0, `${calls.length} calls, none refused`);
    check('every request was still served after a refusal', sent === WORKLOAD.length && failures.filter((f) => /served by a free model/.test(f)).length === 0, String(sent));

    // 6. the env says done: the entry exits 0.
    const exit = await Promise.race([server.exited, new Promise((r) => setTimeout(() => r({ code: 'timeout' }), 15000))]);
    check('the entry exits 0 when GET /work says done', exit.code === 0, `exit ${String(exit.code)}: ${(exit.stderr ?? '').slice(-600)}`);
  } finally {
    env.close();
    if (clientPort !== 0) {
      try {
        await fetch(`http://127.0.0.1:${clientPort}/__never__`).catch(() => void 0);
      } catch {
        // The entry is already gone.
      }
    }
    rmSync(workDir, { recursive: true, force: true });
  }
}

await main();
process.stdout.write(`\n${checks - failures.length}/${checks} checks passed\n`);
if (failures.length > 0) {
  process.stdout.write(`\n${failures.length} failed:\n${failures.map((f) => `  - ${f}`).join('\n')}\n`);
  process.exit(1);
}
process.exit(0);
