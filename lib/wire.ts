// The two client wires (OpenAI Chat Completions, Anthropic Messages) and the three upstream wires
// (OpenAI, Anthropic, Gemini), in both directions, JSON and server-sent events.
//
// The shape of every request here is a pure function of its arguments. Nothing reads a clock or draws a
// random number: two runs of a trial build the same bytes, which is what lets a replay leave the same
// state export.
//
// A provider echoes what it parsed of a request back to us, and the env compares that echo with what its
// client sent: the system prompt, the last user turn's text and image, the tool names in order, the output
// cap, and the stop sequences. So every translation below preserves exactly those fields; anything the echo
// does not carry is passed through when the target wire has a place for it and dropped when it has none.

import { isRecord, type ClientApi, type Json, type Wire } from './providers.ts';

/** A fixed epoch second for every reply's `created`: nothing here reads the clock. */
const FIXED_CREATED = 1767225600;
/** A fixed instant for Anthropic's model list, for the same reason. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

/** An output cap for a client that sent none and an upstream wire that demands one. */
const FALLBACK_MAX_TOKENS = 1024;

// ---------------------------------------------------------------------------------------------
// Normalized request
// ---------------------------------------------------------------------------------------------

export type MessagePart =
  | { kind: 'text'; text: string }
  | { kind: 'image'; url: string | null; media: string; data: string | null }
  | { kind: 'tool_use'; id: string; name: string; input: Json }
  | { kind: 'tool_result'; id: string; text: string };

export interface NormalizedMessage {
  role: 'user' | 'assistant';
  parts: MessagePart[];
}

export interface NormalizedTool {
  name: string;
  description: string;
  schema: Json;
}

/** One client request, in the terms every upstream wire can be built from. */
export interface NormalizedRequest {
  api: ClientApi;
  stream: boolean;
  /** The exact system text the provider must echo, or null when the client sent none. */
  systemText: string | null;
  messages: NormalizedMessage[];
  tools: NormalizedTool[];
  maxTokens: number | null;
  stops: string[];
  temperature: number | null;
  topP: number | null;
  jsonMode: boolean;
}

const textOfBlocks = (content: unknown): string => {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  return parts.join('\n');
};

const numberOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** A data URL split into its media type and payload, or null when it is a remote URL. */
function dataUrlOf(url: string): { media: string; data: string } | null {
  const m = /^data:([^;,]*)(?:;[^,]*)?;base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { media: m[1] === '' ? 'image/png' : m[1], data: m[2] };
}

/** One OpenAI content part (text or image_url) as a message part. */
function partOfOpenAI(part: Json): MessagePart | null {
  if (part.type === 'text' && typeof part.text === 'string') return { kind: 'text', text: part.text };
  if (part.type === 'image_url') {
    const image = isRecord(part.image_url) ? part.image_url : {};
    const url = typeof image.url === 'string' ? image.url : '';
    if (url === '') return null;
    const data = dataUrlOf(url);
    return data === null ? { kind: 'image', url, media: 'image/png', data: null } : { kind: 'image', url: null, media: data.media, data: data.data };
  }
  return null;
}

/** One Anthropic content block as a message part. */
function partOfAnthropic(block: Json): MessagePart | null {
  if (block.type === 'text' && typeof block.text === 'string') return { kind: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { kind: 'image', url: null, media: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { kind: 'image', url: source.url, media: 'image/png', data: null };
  }
  if (block.type === 'tool_use') {
    return {
      kind: 'tool_use',
      id: typeof block.id === 'string' ? block.id : '',
      name: typeof block.name === 'string' ? block.name : '',
      input: isRecord(block.input) ? block.input : {},
    };
  }
  if (block.type === 'tool_result') {
    const content = block.content;
    const text = typeof content === 'string' ? content : textOfBlocks(content) || JSON.stringify(content ?? '');
    return { kind: 'tool_result', id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', text };
  }
  return null;
}

function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** An OpenAI Chat Completions request in the normalized form. */
function normalizeOpenAI(body: Json): NormalizedRequest {
  const systemParts: string[] = [];
  const messages: NormalizedMessage[] = [];
  const raw = Array.isArray(body.messages) ? body.messages : [];
  for (const item of raw) {
    if (!isRecord(item)) continue;
    const role = typeof item.role === 'string' ? item.role : 'user';
    if (role === 'system' || role === 'developer') {
      const text = textOfBlocks(item.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'tool' || role === 'function') {
      const text = textOfBlocks(item.content);
      messages.push({ role: 'user', parts: [{ kind: 'tool_result', id: typeof item.tool_call_id === 'string' ? item.tool_call_id : '', text }] });
      continue;
    }
    const parts: MessagePart[] = [];
    if (typeof item.content === 'string') parts.push({ kind: 'text', text: item.content });
    else if (Array.isArray(item.content)) {
      for (const block of item.content) {
        if (!isRecord(block)) continue;
        const part = partOfOpenAI(block);
        if (part !== null) parts.push(part);
      }
    }
    if (role === 'assistant') {
      const calls = Array.isArray(item.tool_calls) ? item.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          kind: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', parts });
    } else {
      messages.push({ role: 'user', parts });
    }
  }
  const tools: NormalizedTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool)) continue;
      const fn = isRecord(tool.function) ? tool.function : tool;
      if (typeof fn.name !== 'string') continue;
      tools.push({
        name: fn.name,
        description: typeof fn.description === 'string' ? fn.description : '',
        schema: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    api: 'openai',
    stream: body.stream === true,
    systemText: systemParts.length === 0 ? null : systemParts.join('\n'),
    messages,
    tools,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    jsonMode: format === 'json_object' || format === 'json_schema',
  };
}

/** An Anthropic Messages request in the normalized form. */
function normalizeAnthropic(body: Json): NormalizedRequest {
  const system = typeof body.system === 'string' ? body.system : textOfBlocks(body.system);
  const messages: NormalizedMessage[] = [];
  const raw = Array.isArray(body.messages) ? body.messages : [];
  for (const item of raw) {
    if (!isRecord(item)) continue;
    const assistant = item.role === 'assistant';
    const parts: MessagePart[] = [];
    if (typeof item.content === 'string') parts.push({ kind: 'text', text: item.content });
    else if (Array.isArray(item.content)) {
      for (const block of item.content) {
        if (!isRecord(block)) continue;
        const part = partOfAnthropic(block);
        if (part !== null) parts.push(part);
      }
    }
    messages.push({ role: assistant ? 'assistant' : 'user', parts });
  }
  const tools: NormalizedTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== 'string') continue;
      tools.push({
        name: tool.name,
        description: typeof tool.description === 'string' ? tool.description : '',
        schema: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  return {
    api: 'anthropic',
    stream: body.stream === true,
    systemText: system === '' ? null : system,
    messages,
    tools,
    maxTokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    jsonMode: false,
  };
}

export function normalizeRequest(api: ClientApi, body: Json): NormalizedRequest {
  return api === 'openai' ? normalizeOpenAI(body) : normalizeAnthropic(body);
}

// ---------------------------------------------------------------------------------------------
// Upstream requests
// ---------------------------------------------------------------------------------------------

function openAIContent(parts: readonly MessagePart[]): { content: unknown; toolCalls: Json[] } {
  const toolCalls: Json[] = [];
  const blocks: Json[] = [];
  for (const part of parts) {
    if (part.kind === 'text') blocks.push({ type: 'text', text: part.text });
    else if (part.kind === 'image') {
      const url = part.url ?? `data:${part.media};base64,${part.data ?? ''}`;
      blocks.push({ type: 'image_url', image_url: { url } });
    } else if (part.kind === 'tool_use') {
      toolCalls.push({ id: part.id, type: 'function', function: { name: part.name, arguments: JSON.stringify(part.input) } });
    }
  }
  return { content: blocks.length === 1 && blocks[0].type === 'text' ? (blocks[0].text as string) : blocks, toolCalls };
}

function buildOpenAI(model: string, req: NormalizedRequest): Json {
  const messages: Json[] = [];
  if (req.systemText !== null) messages.push({ role: 'system', content: req.systemText });
  for (const message of req.messages) {
    const { content, toolCalls } = openAIContent(message.parts);
    if (toolCalls.length > 0) {
      const text = message.parts
        .filter((p): p is { kind: 'text'; text: string } => p.kind === 'text')
        .map((p) => p.text)
        .join('\n');
      messages.push({ role: 'assistant', content: text === '' ? null : text, tool_calls: toolCalls });
      continue;
    }
    const results = message.parts.filter((p): p is { kind: 'tool_result'; id: string; text: string } => p.kind === 'tool_result');
    for (const result of results) messages.push({ role: 'tool', tool_call_id: result.id, content: result.text });
    const rest = message.parts.filter((p) => p.kind === 'text' || p.kind === 'image');
    if (rest.length > 0) messages.push({ role: message.role, content: content });
  }
  const out: Json = { model, messages };
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stops.length > 0) out.stop = req.stops;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.schema } }));
  }
  if (req.api === 'openai' && req.jsonMode) out.response_format = { type: 'json_object' };
  return out;
}

function buildAnthropic(model: string, req: NormalizedRequest): Json {
  const messages: Json[] = [];
  for (const message of req.messages) {
    const blocks: Json[] = [];
    for (const part of message.parts) {
      if (part.kind === 'text') blocks.push({ type: 'text', text: part.text });
      else if (part.kind === 'image') {
        blocks.push({
          type: 'image',
          source: part.url !== null ? { type: 'url', url: part.url } : { type: 'base64', media_type: part.media, data: part.data ?? '' },
        });
      } else if (part.kind === 'tool_use') blocks.push({ type: 'tool_use', id: part.id, name: part.name, input: part.input });
      else blocks.push({ type: 'tool_result', tool_use_id: part.id, content: part.text });
    }
    messages.push({ role: message.role, content: blocks });
  }
  const out: Json = { model, messages };
  if (req.systemText !== null) out.system = req.systemText;
  out.max_tokens = req.maxTokens ?? FALLBACK_MAX_TOKENS;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stops.length > 0) out.stop_sequences = req.stops;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.schema }));
  }
  return out;
}

function buildGemini(model: string, req: NormalizedRequest): Json {
  const contents: Json[] = [];
  for (const message of req.messages) {
    const parts: Json[] = [];
    for (const part of message.parts) {
      if (part.kind === 'text') parts.push({ text: part.text });
      else if (part.kind === 'image') {
        if (part.url !== null) parts.push({ fileData: { fileUri: part.url, mimeType: part.media } });
        else parts.push({ inlineData: { mimeType: part.media, data: part.data ?? '' } });
      } else if (part.kind === 'tool_use') parts.push({ functionCall: { name: part.name, args: part.input } });
      else parts.push({ functionResponse: { name: part.id, response: { result: part.text } } });
    }
    contents.push({ role: message.role === 'assistant' ? 'model' : 'user', parts });
  }
  const config: Json = {};
  if (req.maxTokens !== null) config.maxOutputTokens = req.maxTokens;
  if (req.temperature !== null) config.temperature = req.temperature;
  if (req.topP !== null) config.topP = req.topP;
  if (req.stops.length > 0) config.stopSequences = req.stops;
  if (req.jsonMode) config.responseMimeType = 'application/json';
  const out: Json = { contents };
  if (req.systemText !== null) out.systemInstruction = { parts: [{ text: req.systemText }] };
  if (req.tools.length > 0) out.tools = [{ functionDeclarations: req.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.schema })) }];
  if (Object.keys(config).length > 0) out.generationConfig = config;
  void model;
  return out;
}

/** The upstream body for one provider wire. */
export function buildUpstreamBody(wire: Wire, model: string, req: NormalizedRequest): Json {
  if (wire === 'openai') return buildOpenAI(model, req);
  if (wire === 'anthropic') return buildAnthropic(model, req);
  return buildGemini(model, req);
}

/** The path (relative to the provider's base) an upstream call is made to. */
export function upstreamPath(wire: Wire, model: string): string {
  if (wire === 'openai') return '/v1/chat/completions';
  if (wire === 'anthropic') return '/v1/messages';
  return `/v1beta/models/${encodeURIComponent(model)}:generateContent`;
}

// ---------------------------------------------------------------------------------------------
// Upstream answers
// ---------------------------------------------------------------------------------------------

/** What one provider call answered: its text whole, its usage counts, and the stop family it reported. */
export interface UpstreamAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stopFamily: number;
}

const num = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);

const STOP_FAMILIES: Record<string, number> = {
  stop: 0,
  end_turn: 0,
  STOP: 0,
  length: 1,
  max_tokens: 1,
  MAX_TOKENS: 1,
  tool_calls: 2,
  function_call: 2,
  tool_use: 2,
  content_filter: 3,
  refusal: 3,
  SAFETY: 3,
};

export function stopFamilyOf(reason: unknown): number {
  return typeof reason === 'string' ? (STOP_FAMILIES[reason] ?? 0) : 0;
}

/** One provider's answer body, in its wire, as the text, usage and stop family it carries. */
export function readUpstream(wire: Wire, body: unknown): UpstreamAnswer | null {
  if (!isRecord(body)) return null;
  if (wire === 'openai') {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) return null;
    const message = isRecord(choice.message) ? choice.message : {};
    const text = typeof message.content === 'string' ? message.content : textOfBlocks(message.content);
    const usage = isRecord(body.usage) ? body.usage : {};
    return { text, inputTokens: num(usage.prompt_tokens), outputTokens: num(usage.completion_tokens), stopFamily: stopFamilyOf(choice.finish_reason) };
  }
  if (wire === 'anthropic') {
    const content = Array.isArray(body.content) ? body.content : [];
    const texts: string[] = [];
    for (const block of content) if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
    const usage = isRecord(body.usage) ? body.usage : {};
    return { text: texts.join(''), inputTokens: num(usage.input_tokens), outputTokens: num(usage.output_tokens), stopFamily: stopFamilyOf(body.stop_reason) };
  }
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const texts: string[] = [];
  for (const part of parts) if (isRecord(part) && typeof part.text === 'string') texts.push(part.text);
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: texts.join(''),
    inputTokens: num(usage.promptTokenCount),
    outputTokens: num(usage.candidatesTokenCount),
    stopFamily: stopFamilyOf(candidate.finishReason),
  };
}

// ---------------------------------------------------------------------------------------------
// Client answers
// ---------------------------------------------------------------------------------------------

/** The stop reason name a client wire uses for one family. */
export function stopReasonFor(api: ClientApi, family: number): string {
  if (api === 'openai') return family === 1 ? 'length' : family === 2 ? 'tool_calls' : family === 3 ? 'content_filter' : 'stop';
  return family === 1 ? 'max_tokens' : family === 2 ? 'tool_use' : family === 3 ? 'refusal' : 'end_turn';
}

/** An OpenAI Chat Completions object for one answer. */
export function openAICompletion(id: string, model: string, answer: UpstreamAnswer): Json {
  return {
    id,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: stopReasonFor('openai', answer.stopFamily),
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** An Anthropic Messages object for one answer. */
export function anthropicMessage(id: string, model: string, answer: UpstreamAnswer): Json {
  return {
    id,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: stopReasonFor('anthropic', answer.stopFamily),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** The pieces the assistant's text is cut into on the wire: fixed, so a replay writes the same bytes. */
export function textPieces(text: string, size = 96): string[] {
  if (text === '') return [];
  const out: string[] = [];
  for (let at = 0; at < text.length; at += size) out.push(text.slice(at, at + size));
  return out;
}

/** An OpenAI event stream for one answer: role chunk, text deltas, the finish chunk, usage, [DONE]. */
export function openAIStreamFrames(id: string, model: string, answer: UpstreamAnswer, piece: (t: string) => string[] = textPieces): string[] {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id,
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const part of piece(answer.text)) frames.push(sseEvent(null, chunk({ content: part }, null)));
  frames.push(sseEvent(null, chunk({}, stopReasonFor('openai', answer.stopFamily))));
  frames.push(
    sseEvent(null, {
      id,
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames;
}

/** An Anthropic event stream for one answer, in the order the Messages API defines. */
export function anthropicStreamFrames(id: string, model: string, answer: UpstreamAnswer, piece: (t: string) => string[] = textPieces): string[] {
  const frames = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const part of piece(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: part } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopReasonFor('anthropic', answer.stopFamily), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames;
}

// ---------------------------------------------------------------------------------------------
// Model listings and errors
// ---------------------------------------------------------------------------------------------

export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** GET /v1/models answered in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models answered in Anthropic's list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_CREATED_AT })),
    has_more: false,
    first_id: models.length > 0 ? models[0].id : null,
    last_id: models.length > 0 ? models[models.length - 1].id : null,
  };
}

/** An error in OpenAI's envelope. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, param: null, code } };
}

/** An error in Anthropic's envelope. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

export function errorBody(api: ClientApi, status: number, message: string): Json {
  if (api === 'anthropic') {
    const type = status === 400 ? 'invalid_request_error' : status === 401 ? 'authentication_error' : status === 429 ? 'rate_limit_error' : 'api_error';
    return anthropicError(type, message);
  }
  const type = status === 400 ? 'invalid_request_error' : status === 401 ? 'authentication_error' : status === 429 ? 'rate_limit_error' : 'api_error';
  return openAIError(message, type, null);
}
