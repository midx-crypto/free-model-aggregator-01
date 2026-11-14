// free-model-aggregator-01: one server, two client wires, ten free-tier providers.
//
// The server reads the trial brief (MAKEFUN_PROMPT_FILE), builds the catalog from
// the provider directory and the quality table, registers with the env, and answers
// the client's 20 requests. Each request is routed to the best free model that
// meets its requirements (by the quality table's rating_milli), failing over to
// the next-best model when a provider refuses (402/429/5xx). Requests are sent to
// providers strictly one at a time: a provider serves one call at a time, and a
// second concurrent call is refused.
//
// Wires spoken upstream: OpenAI Chat Completions, Anthropic Messages, Gemini
// generateContent / streamGenerateContent. Wires spoken downstream: OpenAI Chat
// Completions (POST /v1/chat/completions) and Anthropic Messages
// (POST /v1/messages), JSON and SSE, plus GET /v1/models in the shape the auth
// header names (Authorization: Bearer -> OpenAI list, x-api-key -> Anthropic list).
//
// Everything is a pure function of the brief, the requests and the providers'
// answers: no wall clock, no unseeded randomness, so a replay leaves the same
// state export. `node main.ts --list` prints the catalog document byte for byte
// as it is registered.
//
// Node 22+, standard library only.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const UPSTREAM_TIMEOUT_MS = 3000;
const REGISTER_RETRIES = 50;
const REGISTER_RETRY_MS = 200;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Fixed timestamps in GET /v1/models answers: nothing in a reply reads the clock. */
const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const strOf = (v: unknown): string => (typeof v === 'string' ? v : '');
const numOrNull = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null;
const boolOrNull = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------------------
// The brief: three JSON documents embedded in prose, found by their `schema`.
// ---------------------------------------------------------------------------

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

interface ProviderDir {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  keyEnv: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

interface ManifestEntry {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

interface Brief {
  providers: ProviderDir[];
  ratings: RatingRow[];
  manifest: ManifestEntry[];
  error: string | null;
}

/** The index of the `}` closing the object that opens at `start`, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in order. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function providerOf(v: unknown): ProviderDir | null {
  if (!isRecord(v)) return null;
  const id = strOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none'
      ? v.auth
      : 'bearer';
  const base =
    strOf(v.path) !== ''
      ? strOf(v.path)
      : strOf(v.base_path) !== ''
        ? strOf(v.base_path)
        : strOf(v.base_url) !== ''
          ? strOf(v.base_url)
          : strOf(v.base);
  return {
    id,
    name: strOf(v.name) !== '' ? strOf(v.name) : id,
    wire,
    auth,
    base,
    keyEnv: strOf(v.key_env),
    key: strOf(v.key),
  };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = strOf(v.model);
  const provider = strOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    ratingMilli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    contextLength: numOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function manifestOf(v: unknown): ManifestEntry | null {
  if (!isRecord(v)) return null;
  const id = strOf(v.id);
  if (id === '') return null;
  const req = isRecord(v.requires) ? v.requires : {};
  const requires: Requirement = {};
  if (req.tools === true) requires.tools = true;
  if (req.vision === true) requires.vision = true;
  if (req.json === true) requires.json = true;
  if (typeof req.min_context === 'number' && Number.isFinite(req.min_context)) {
    requires.minContext = req.min_context;
  }
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires,
  };
}

function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const dirDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/providers/v1') ?? null;
  const ratingsDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/ratings/v1') ?? null;
  const workloadDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/workload/v1') ?? null;
  const providers: ProviderDir[] =
    dirDoc !== null && Array.isArray(dirDoc.providers)
      ? dirDoc.providers.flatMap((v) => {
          const p = providerOf(v);
          return p === null ? [] : [p];
        })
      : [];
  const ratings: RatingRow[] =
    ratingsDoc !== null && Array.isArray(ratingsDoc.ratings)
      ? ratingsDoc.ratings.flatMap((v) => {
          const r = ratingOf(v);
          return r === null ? [] : [r];
        })
      : [];
  const manifest: ManifestEntry[] =
    workloadDoc !== null && Array.isArray(workloadDoc.manifest)
      ? workloadDoc.manifest.flatMap((v) => {
          const m = manifestOf(v);
          return m === null ? [] : [m];
        })
      : [];
  return { providers, ratings, manifest, error: providers.length === 0 ? 'the brief holds no provider directory' : null };
}

// ---------------------------------------------------------------------------
// The catalog: one entry per provider of the directory, its models from the
// quality table (the sealed truth: every model the provider lists, free flag
// and context length). Built from the brief alone, so `--list` and the
// register body are the same document in every environment.
// ---------------------------------------------------------------------------

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface Catalog {
  providers: CatalogProvider[];
}

function buildCatalog(brief: Brief): Catalog {
  const providers: CatalogProvider[] = [];
  for (const provider of brief.providers) {
    const rows = brief.ratings.filter((r) => r.provider === provider.id);
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const row of [...rows].sort((a, b) => (a.model < b.model ? -1 : a.model > b.model ? 1 : 0))) {
      if (seen.has(row.model)) continue;
      seen.add(row.model);
      models.push({ id: row.model, free: row.free, context_length: row.contextLength });
    }
    providers.push({
      id: provider.id,
      base_url: provider.base,
      key_env: provider.keyEnv,
      wire: provider.wire,
      models,
    });
  }
  return { providers };
}

/** A base path joined to the env's URL; an absolute base is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

// ---------------------------------------------------------------------------
// The canonical request: one shape for both client wires, holding exactly
// what the providers echo back (system text, user text, image bytes, tool
// names, max_tokens, stop sequences, JSON mode), so every translation keeps
// the echo identical to what the client sent.
// ---------------------------------------------------------------------------

type Part =
  | { kind: 'text'; text: string }
  | { kind: 'image'; mime: string; data: string; url: string | null }
  | { kind: 'tool_result'; toolUseId: string; text: string };

interface ToolCall {
  id: string;
  name: string;
  args: Json;
}

interface CanonMessage {
  role: 'user' | 'assistant' | 'tool';
  parts: Part[];
  toolCalls: ToolCall[];
  toolCallId: string;
}

interface CanonTool {
  name: string;
  description: string;
  parameters: Json;
}

interface CanonRequest {
  system: string;
  messages: CanonMessage[];
  tools: CanonTool[];
  toolChoice: unknown;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
}

/** The text of a string or of a block array's text blocks, joined with '\n'. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** A string or string array as a string list (an OpenAI `stop` field). */
function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

function argsOf(value: unknown): Json {
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

/** One OpenAI content part as a canonical part; null for parts we do not carry. */
function openAIPart(block: unknown): Part | null {
  if (!isRecord(block)) return null;
  if (block.type === 'text' && typeof block.text === 'string') return { kind: 'text', text: block.text };
  if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
    const url = block.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { kind: 'image', mime: data[1] !== '' ? data[1] : 'image/png', data: data[2], url: null };
    }
    return { kind: 'image', mime: '', data: '', url };
  }
  return null;
}

/** An OpenAI Chat Completions body as a canonical request. */
function canonFromOpenAI(body: Json): CanonRequest {
  const systemParts: string[] = [];
  const messages: CanonMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const toolCalls: ToolCall[] = [];
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          toolCalls.push({
            id: strOf(call.id),
            name: strOf(call.function.name),
            args: argsOf(call.function.arguments),
          });
        }
      }
      const parts: Part[] = [];
      const text = blocksText(raw.content);
      if (text !== '') parts.push({ kind: 'text', text });
      messages.push({ role: 'assistant', parts, toolCalls, toolCallId: '' });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'tool',
        parts: typeof raw.content === 'string' ? [{ kind: 'text', text: raw.content }] : [],
        toolCalls: [],
        toolCallId: strOf(raw.tool_call_id),
      });
      continue;
    }
    if (raw.role !== 'user') continue;
    const parts: Part[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ kind: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        const part = openAIPart(block);
        if (part !== null) parts.push(part);
      }
    }
    messages.push({ role: 'user', parts, toolCalls: [], toolCallId: '' });
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: strOf(raw.function.description),
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    system: systemParts.join('\n'),
    messages,
    tools,
    toolChoice: body.tool_choice ?? null,
    maxTokens: numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens),
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** One Anthropic content block as a canonical part; null for parts we do not carry. */
function anthropicPart(block: unknown): Part | null {
  if (!isRecord(block)) return null;
  if (block.type === 'text' && typeof block.text === 'string') return { kind: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return {
        kind: 'image',
        mime: typeof source.media_type === 'string' ? source.media_type : 'image/png',
        data: source.data,
        url: null,
      };
    }
    if (source.type === 'url' && typeof source.url === 'string') {
      return { kind: 'image', mime: '', data: '', url: source.url };
    }
  }
  if (block.type === 'tool_result') {
    return {
      kind: 'tool_result',
      toolUseId: strOf(block.tool_use_id),
      text: blocksText(block.content) !== '' ? blocksText(block.content) : '',
    };
  }
  return null;
}

/** An Anthropic Messages body as a canonical request. */
function canonFromAnthropic(body: Json): CanonRequest {
  const messages: CanonMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const parts: Part[] = [];
      const toolCalls: ToolCall[] = [];
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') {
            parts.push({ kind: 'text', text: block.text });
          } else if (block.type === 'tool_use') {
            toolCalls.push({
              id: strOf(block.id),
              name: strOf(block.name),
              args: isRecord(block.input) ? block.input : {},
            });
          }
        }
      } else if (typeof raw.content === 'string' && raw.content !== '') {
        parts.push({ kind: 'text', text: raw.content });
      }
      messages.push({ role: 'assistant', parts, toolCalls, toolCallId: '' });
      continue;
    }
    if (raw.role !== 'user') continue;
    const parts: Part[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ kind: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        const part = anthropicPart(block);
        if (part !== null) parts.push(part);
      }
    }
    messages.push({ role: 'user', parts, toolCalls: [], toolCallId: '' });
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: strOf(raw.description),
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  return {
    system: blocksText(body.system),
    messages,
    tools,
    toolChoice: body.tool_choice ?? null,
    maxTokens: numOrNull(body.max_tokens),
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    stream: body.stream === true,
  };
}

/** The user text the provider echoes: the text parts of the user messages. */
function userTextOf(canon: CanonRequest): string {
  const texts: string[] = [];
  for (const message of canon.messages) {
    if (message.role !== 'user') continue;
    const text = message.parts
      .flatMap((p) => (p.kind === 'text' ? [p.text] : []))
      .join('\n');
    texts.push(text);
  }
  return texts.length === 0 ? '' : texts[texts.length - 1];
}

// ---------------------------------------------------------------------------
// Upstream requests: the canonical request in each provider wire's shape.
// ---------------------------------------------------------------------------

/** The canonical messages as OpenAI chat messages. */
function openAIMessages(canon: CanonRequest): Json[] {
  const messages: Json[] = [];
  if (canon.system !== '') messages.push({ role: 'system', content: canon.system });
  for (const message of canon.messages) {
    if (message.role === 'user') {
      const parts: Json[] = [];
      for (const part of message.parts) {
        if (part.kind === 'text') parts.push({ type: 'text', text: part.text });
        else if (part.kind === 'image') {
          const url =
            part.url !== null ? part.url : `data:${part.mime !== '' ? part.mime : 'image/png'};base64,${part.data}`;
          parts.push({ type: 'image_url', image_url: { url } });
        } else if (part.kind === 'tool_result') {
          messages.push({ role: 'tool', tool_call_id: part.toolUseId, content: part.text });
        }
      }
      if (parts.length === 1 && parts[0].type === 'text') {
        messages.push({ role: 'user', content: (parts[0] as Json).text });
      } else if (parts.length > 0) {
        messages.push({ role: 'user', content: parts });
      }
      continue;
    }
    if (message.role === 'assistant') {
      const text = message.parts
        .flatMap((p) => (p.kind === 'text' ? [p.text] : []))
        .join('\n');
      const out: Json = { role: 'assistant', content: text !== '' ? text : null };
      if (message.toolCalls.length > 0) {
        out.tool_calls = message.toolCalls.map((c) => ({
          id: c.id,
          type: 'function',
          function: { name: c.name, arguments: JSON.stringify(c.args) },
        }));
      }
      messages.push(out);
      continue;
    }
    // role 'tool': a tool message of its own.
    const text = message.parts
      .flatMap((p) => (p.kind === 'text' ? [p.text] : []))
      .join('\n');
    messages.push({ role: 'tool', tool_call_id: message.toolCallId, content: text });
  }
  return messages;
}

function openAITools(canon: CanonRequest): Json[] {
  return canon.tools.map((t) => ({
    type: 'function',
    function: { name: t.name, description: t.description, parameters: t.parameters },
  }));
}

/** `tool_choice` in the OpenAI wire's spelling. */
function openAIToolChoice(choice: unknown): unknown {
  if (typeof choice === 'string') {
    if (choice === 'auto' || choice === 'none' || choice === 'required') return choice;
    return 'auto';
  }
  if (isRecord(choice)) {
    if (choice.type === 'auto') return 'auto';
    if (choice.type === 'any') return 'required';
    if (choice.type === 'none') return 'none';
    if (choice.type === 'tool' && typeof choice.name === 'string') {
      return { type: 'function', function: { name: choice.name } };
    }
    if (choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
      return choice;
    }
  }
  return undefined;
}

/** The canonical request as an OpenAI Chat Completions body naming `model`. */
function openAIUpstreamBody(canon: CanonRequest, model: string): Json {
  const out: Json = { model, messages: openAIMessages(canon) };
  if (canon.maxTokens !== null) out.max_tokens = canon.maxTokens;
  if (canon.temperature !== null) out.temperature = canon.temperature;
  if (canon.topP !== null) out.top_p = canon.topP;
  if (canon.stops.length > 0) out.stop = canon.stops;
  if (canon.tools.length > 0) out.tools = openAITools(canon);
  const choice = openAIToolChoice(canon.toolChoice);
  if (choice !== undefined) out.tool_choice = choice;
  if (canon.jsonMode) out.response_format = { type: 'json_object' };
  if (canon.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The canonical messages as Anthropic messages. */
function anthropicMessages(canon: CanonRequest): Json[] {
  const messages: Json[] = [];
  for (const message of canon.messages) {
    if (message.role === 'user') {
      const blocks: Json[] = [];
      for (const part of message.parts) {
        if (part.kind === 'text') blocks.push({ type: 'text', text: part.text });
        else if (part.kind === 'image') {
          if (part.url !== null) blocks.push({ type: 'image', source: { type: 'url', url: part.url } });
          else {
            blocks.push({
              type: 'image',
              source: { type: 'base64', media_type: part.mime !== '' ? part.mime : 'image/png', data: part.data },
            });
          }
        } else if (part.kind === 'tool_result') {
          blocks.push({ type: 'tool_result', tool_use_id: part.toolUseId, content: part.text });
        }
      }
      if (blocks.length === 1 && blocks[0].type === 'text') {
        messages.push({ role: 'user', content: (blocks[0] as Json).text });
      } else if (blocks.length > 0) {
        messages.push({ role: 'user', content: blocks });
      }
      continue;
    }
    if (message.role === 'assistant') {
      const blocks: Json[] = [];
      for (const part of message.parts) {
        if (part.kind === 'text') blocks.push({ type: 'text', text: part.text });
      }
      for (const call of message.toolCalls) {
        blocks.push({ type: 'tool_use', id: call.id, name: call.name, input: call.args });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    // role 'tool': a tool result on a user message of its own.
    const text = message.parts
      .flatMap((p) => (p.kind === 'text' ? [p.text] : []))
      .join('\n');
    messages.push({
      role: 'user',
      content: [{ type: 'tool_result', tool_use_id: message.toolCallId, content: text }],
    });
  }
  return messages;
}

/** `tool_choice` in the Anthropic wire's spelling. */
function anthropicToolChoice(choice: unknown): unknown {
  if (typeof choice === 'string') {
    if (choice === 'auto') return { type: 'auto' };
    if (choice === 'required') return { type: 'any' };
    if (choice === 'none') return { type: 'none' };
    return undefined;
  }
  if (isRecord(choice)) {
    if (choice.type === 'auto' || choice.type === 'any' || choice.type === 'none') return { type: choice.type };
    if (choice.type === 'tool' && typeof choice.name === 'string') return { type: 'tool', name: choice.name };
    if (choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
      return { type: 'tool', name: choice.function.name };
    }
  }
  return undefined;
}

/** The canonical request as an Anthropic Messages body naming `model`. */
function anthropicUpstreamBody(canon: CanonRequest, model: string): Json {
  const out: Json = { model, messages: anthropicMessages(canon) };
  if (canon.maxTokens !== null) out.max_tokens = canon.maxTokens;
  if (canon.system !== '') out.system = canon.system;
  if (canon.temperature !== null) out.temperature = canon.temperature;
  if (canon.topP !== null) out.top_p = canon.topP;
  if (canon.stops.length > 0) out.stop_sequences = canon.stops;
  if (canon.tools.length > 0) {
    out.tools = canon.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  const choice = anthropicToolChoice(canon.toolChoice);
  if (choice !== undefined) out.tool_choice = choice;
  if (canon.stream) out.stream = true;
  return out;
}

/** The canonical messages as Gemini contents. */
function geminiContents(canon: CanonRequest): Json[] {
  const contents: Json[] = [];
  for (const message of canon.messages) {
    if (message.role === 'user') {
      const parts: Json[] = [];
      for (const part of message.parts) {
        if (part.kind === 'text') parts.push({ text: part.text });
        else if (part.kind === 'image') {
          if (part.url !== null) parts.push({ fileData: { fileUri: part.url } });
          else parts.push({ inlineData: { mimeType: part.mime !== '' ? part.mime : 'image/png', data: part.data } });
        } else if (part.kind === 'tool_result') {
          parts.push({ functionResponse: { name: part.toolUseId, response: { content: part.text } } });
        }
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
      continue;
    }
    if (message.role === 'assistant') {
      const parts: Json[] = [];
      for (const part of message.parts) {
        if (part.kind === 'text') parts.push({ text: part.text });
      }
      for (const call of message.toolCalls) {
        parts.push({ functionCall: { name: call.name, args: call.args } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const text = message.parts
      .flatMap((p) => (p.kind === 'text' ? [p.text] : []))
      .join('\n');
    contents.push({
      role: 'user',
      parts: [{ functionResponse: { name: message.toolCallId, response: { content: text } } }],
    });
  }
  return contents;
}

/** The canonical request as a Gemini generateContent body. */
function geminiUpstreamBody(canon: CanonRequest): Json {
  const out: Json = { contents: geminiContents(canon) };
  if (canon.system !== '') out.systemInstruction = { parts: [{ text: canon.system }] };
  if (canon.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: canon.tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.parameters,
        })),
      },
    ];
  }
  const config: Json = {};
  if (canon.maxTokens !== null) config.maxOutputTokens = canon.maxTokens;
  if (canon.temperature !== null) config.temperature = canon.temperature;
  if (canon.topP !== null) config.topP = canon.topP;
  if (canon.stops.length > 0) config.stopSequences = canon.stops;
  if (canon.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Upstream answers: every provider wire's JSON and SSE shapes into one
// answer: the whole text, the usage counts, and the stop reason's family.
// ---------------------------------------------------------------------------

interface UpstreamAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stopFamily: number;
}

interface UpstreamFailure {
  ok: false;
  status: number;
  code: string;
}

type UpstreamResult = { ok: true; answer: UpstreamAnswer } | UpstreamFailure;

/** A stop reason's family: 0 end, 1 length, 2 tool call, 3 refusal, -1 unknown. */
function stopFamilyOf(reason: unknown): number {
  if (reason === 'stop' || reason === 'end_turn' || reason === 'STOP') return 0;
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return -1;
}

function numOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** The text of an OpenAI message: its string content, or its text parts joined. */
function openAIText(content: unknown): string {
  if (typeof content === 'string') return content;
  return blocksText(content);
}

/** Parse one SSE text into its frames: {event, data} pairs. */
function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function jsonOrNull(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

/** An OpenAI-wire answer (JSON or SSE) as one answer. */
function parseOpenAIAnswer(bodyText: string, stream: boolean): UpstreamAnswer | null {
  if (!stream) {
    const value = jsonOrNull(bodyText);
    if (!isRecord(value) || !Array.isArray(value.choices) || value.choices.length === 0) return null;
    const choice = isRecord(value.choices[0]) ? value.choices[0] : null;
    if (choice === null || !isRecord(choice.message)) return null;
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text: openAIText(choice.message.content),
      inputTokens: numOrZero(usage.prompt_tokens),
      outputTokens: numOrZero(usage.completion_tokens),
      stopFamily: stopFamilyOf(choice.finish_reason),
    };
  }
  let text = '';
  let input = 0;
  let output = 0;
  let family = -1;
  for (const frame of sseFrames(bodyText)) {
    if (frame.data.trim() === '[DONE]') continue;
    const chunk = jsonOrNull(frame.data);
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      input = numOrZero(chunk.usage.prompt_tokens);
      output = numOrZero(chunk.usage.completion_tokens);
    }
    if (!Array.isArray(chunk.choices) || chunk.choices.length === 0) continue;
    const choice = isRecord(chunk.choices[0]) ? chunk.choices[0] : null;
    if (choice === null) continue;
    if (isRecord(choice.delta) && typeof choice.delta.content === 'string') text += choice.delta.content;
    if (typeof choice.finish_reason === 'string') family = stopFamilyOf(choice.finish_reason);
  }
  return { text, inputTokens: input, outputTokens: output, stopFamily: family };
}

/** An Anthropic-wire answer (JSON or SSE) as one answer. */
function parseAnthropicAnswer(bodyText: string, stream: boolean): UpstreamAnswer | null {
  if (!stream) {
    const value = jsonOrNull(bodyText);
    if (!isRecord(value) || !Array.isArray(value.content)) return null;
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text: blocksText(value.content),
      inputTokens: numOrZero(usage.input_tokens),
      outputTokens: numOrZero(usage.output_tokens),
      stopFamily: stopFamilyOf(value.stop_reason),
    };
  }
  let text = '';
  let input = 0;
  let output = 0;
  let family = -1;
  for (const frame of sseFrames(bodyText)) {
    const data = jsonOrNull(frame.data);
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') text += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') {
        family = stopFamilyOf(data.delta.stop_reason);
      }
      if (isRecord(data.usage)) output = numOrZero(data.usage.output_tokens);
    }
  }
  return { text, inputTokens: input, outputTokens: output, stopFamily: family };
}

/** A Gemini-wire answer (JSON or SSE) as one answer. */
function parseGeminiAnswer(bodyText: string, stream: boolean): UpstreamAnswer | null {
  if (!stream) {
    const value = jsonOrNull(bodyText);
    if (!isRecord(value) || !Array.isArray(value.candidates) || value.candidates.length === 0) return null;
    const candidate = isRecord(value.candidates[0]) ? value.candidates[0] : null;
    if (candidate === null) return null;
    const content = isRecord(candidate.content) ? candidate.content : {};
    const text = Array.isArray(content.parts)
      ? content.parts
          .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
          .join('')
      : '';
    const usage = isRecord(value.usageMetadata) ? value.usageMetadata : {};
    return {
      text,
      inputTokens: numOrZero(usage.promptTokenCount),
      outputTokens: numOrZero(usage.candidatesTokenCount),
      stopFamily: stopFamilyOf(candidate.finishReason),
    };
  }
  let text = '';
  let input = 0;
  let output = 0;
  let family = -1;
  for (const frame of sseFrames(bodyText)) {
    const value = jsonOrNull(frame.data);
    if (!isRecord(value) || !Array.isArray(value.candidates)) continue;
    for (const raw of value.candidates) {
      if (!isRecord(raw)) continue;
      const content = isRecord(raw.content) ? raw.content : {};
      if (Array.isArray(content.parts)) {
        for (const p of content.parts) {
          if (isRecord(p) && typeof p.text === 'string') text += p.text;
        }
      }
      if (typeof raw.finishReason === 'string') family = stopFamilyOf(raw.finishReason);
    }
    if (isRecord(value.usageMetadata)) {
      input = numOrZero(value.usageMetadata.promptTokenCount);
      output = numOrZero(value.usageMetadata.candidatesTokenCount);
    }
  }
  return { text, inputTokens: input, outputTokens: output, stopFamily: family };
}

/**
 * One call to one provider, in its own wire, answered as one answer or as a
 * failure. A failure spends the provider's budget only when the provider
 * counted it; the caller decides the next candidate either way.
 */
async function callUpstream(
  provider: ProviderDir,
  model: string,
  canon: CanonRequest,
): Promise<UpstreamResult> {
  const base = joinBase(ENV_URL, provider.base);
  let url: string;
  let body: Json;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    body = openAIUpstreamBody(canon, model);
    if (canon.stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    body = anthropicUpstreamBody(canon, model);
    if (canon.stream) headers.accept = 'text/event-stream';
  } else {
    const action = canon.stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(model)}:${action}${canon.stream ? '?alt=sse' : ''}`;
    body = geminiUpstreamBody(canon);
    if (canon.stream) headers.accept = 'text/event-stream';
  }
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch {
    return { ok: false, status: 0, code: 'unreachable' };
  }
  const text = await res.text();
  if (!res.ok) {
    const value = jsonOrNull(text);
    let code = '';
    if (isRecord(value)) {
      if (isRecord(value.error) && typeof value.error.code === 'string') code = value.error.code;
      else if (typeof value.error === 'string') code = value.error;
      else if (typeof value.code === 'string') code = value.code;
    }
    return { ok: false, status: res.status, code };
  }
  const parsed =
    provider.wire === 'openai'
      ? parseOpenAIAnswer(text, canon.stream)
      : provider.wire === 'anthropic'
        ? parseAnthropicAnswer(text, canon.stream)
        : parseGeminiAnswer(text, canon.stream);
  if (parsed === null) return { ok: false, status: 502, code: 'unparseable_answer' };
  return { ok: true, answer: parsed };
}

// ---------------------------------------------------------------------------
// Client answers: the upstream answer back in the wire the client used.
// The text, the usage and the stop reason's family are the provider's own.
// ---------------------------------------------------------------------------

function openAIStopReason(family: number): string {
  if (family === 1) return 'length';
  if (family === 2) return 'tool_calls';
  if (family === 3) return 'content_filter';
  return 'stop';
}

function anthropicStopReason(family: number): string {
  if (family === 1) return 'max_tokens';
  if (family === 2) return 'tool_use';
  if (family === 3) return 'refusal';
  return 'end_turn';
}

function openAIJsonAnswer(model: string, answer: UpstreamAnswer): Json {
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: openAIStopReason(answer.stopFamily),
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

/** Split text into pieces for streaming deltas. */
function splitText(text: string, size = 64): string[] {
  if (text === '') return [];
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += size) pieces.push(text.slice(at, at + size));
  return pieces;
}

/** The upstream answer as an OpenAI SSE stream: role chunk, content deltas, finish chunk, usage chunk, [DONE]. */
function openAIStreamAnswer(model: string, answer: UpstreamAnswer): string {
  const head = { id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model };
  const frames: string[] = [];
  frames.push(
    `data: ${JSON.stringify({ ...head, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] })}\n\n`,
  );
  for (const piece of splitText(answer.text)) {
    frames.push(
      `data: ${JSON.stringify({ ...head, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] })}\n\n`,
    );
  }
  frames.push(
    `data: ${JSON.stringify({ ...head, choices: [{ index: 0, delta: {}, finish_reason: openAIStopReason(answer.stopFamily) }] })}\n\n`,
  );
  frames.push(
    `data: ${JSON.stringify({
      ...head,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    })}\n\n`,
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

function anthropicJsonAnswer(model: string, answer: UpstreamAnswer): Json {
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: anthropicStopReason(answer.stopFamily),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** The upstream answer as the Anthropic event order: message_start, content_block_start, deltas, content_block_stop, message_delta, message_stop. */
function anthropicStreamAnswer(model: string, answer: UpstreamAnswer): string {
  const event = (name: string, data: unknown): string => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
  const frames: string[] = [];
  frames.push(
    event('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
  );
  frames.push(
    event('content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    }),
  );
  for (const piece of splitText(answer.text)) {
    frames.push(
      event('content_block_delta', {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: piece },
      }),
    );
  }
  frames.push(event('content_block_stop', { type: 'content_block_stop', index: 0 }));
  frames.push(
    event('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopReason(answer.stopFamily), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
  );
  frames.push(event('message_stop', { type: 'message_stop' }));
  return frames.join('');
}

function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

// ---------------------------------------------------------------------------
// Routing: the free models that meet a request's requirements, best rating
// first. A tie goes to the wire the client already speaks (no translation),
// then to the provider and model ids, so the order is deterministic.
// ---------------------------------------------------------------------------

interface Candidate {
  provider: ProviderDir;
  row: RatingRow;
}

function meetsRequirements(row: RatingRow, requires: Requirement): boolean {
  if (requires.tools === true && !row.tools) return false;
  if (requires.vision === true && !row.vision) return false;
  if (requires.json === true && !row.json) return false;
  if (requires.minContext !== undefined && (row.contextLength === null || row.contextLength < requires.minContext)) {
    return false;
  }
  return true;
}

/**
 * The candidates for one request. An OpenAI request in JSON mode cannot go
 * to an Anthropic-wire provider: that wire cannot carry response_format, so
 * the provider would echo JSON mode absent and the request would fail its
 * echo check. One model per provider: a provider's budget and failure state
 * are per provider, not per model, so its second model would answer the same
 * refusal its best just did (and spend another call doing it).
 */
function candidatesFor(
  brief: Brief,
  clientApi: 'openai' | 'anthropic',
  requires: Requirement,
  jsonMode: boolean,
): Candidate[] {
  const out: Candidate[] = [];
  for (const provider of brief.providers) {
    if (clientApi === 'openai' && jsonMode && provider.wire === 'anthropic') continue;
    for (const row of brief.ratings) {
      if (row.provider !== provider.id || !row.free) continue;
      if (!meetsRequirements(row, requires)) continue;
      out.push({ provider, row });
    }
  }
  out.sort((a, b) => {
    if (a.row.ratingMilli !== b.row.ratingMilli) return b.row.ratingMilli - a.row.ratingMilli;
    const aSame = a.provider.wire === clientApi ? 0 : 1;
    const bSame = b.provider.wire === clientApi ? 0 : 1;
    if (aSame !== bSame) return aSame - bSame;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.row.model < b.row.model ? -1 : a.row.model > b.row.model ? 1 : 0;
  });
  const seen = new Set<string>();
  return out.filter((c) => {
    if (seen.has(c.provider.id)) return false;
    seen.add(c.provider.id);
    return true;
  });
}

/** Consecutive failed calls per provider, across requests: two in a row rest it. */
const consecutiveFailures = new Map<string, number>();

/**
 * Serve one canonical request: try the candidates in order, failing over on
 * any refusal. A provider that failed the last two calls rests: it is skipped
 * on the first pass (its budget is almost certainly spent, and another try
 * only costs a call) and retried on the second pass, which runs only when
 * every other candidate refused, so a resting provider still serves a
 * request nothing else can.
 */
async function serve(
  brief: Brief,
  clientApi: 'openai' | 'anthropic',
  requires: Requirement,
  canon: CanonRequest,
): Promise<{ model: string; answer: UpstreamAnswer } | null> {
  const candidates = candidatesFor(brief, clientApi, requires, canon.jsonMode);
  const rested = candidates.filter((c) => (consecutiveFailures.get(c.provider.id) ?? 0) >= 2);
  const fresh = candidates.filter((c) => (consecutiveFailures.get(c.provider.id) ?? 0) < 2);
  for (const pass of [fresh, rested]) {
    for (const candidate of pass) {
      const result = await callUpstream(candidate.provider, candidate.row.model, canon);
      if (result.ok) {
        consecutiveFailures.set(candidate.provider.id, 0);
        return { model: candidate.row.model, answer: result.answer };
      }
      consecutiveFailures.set(
        candidate.provider.id,
        (consecutiveFailures.get(candidate.provider.id) ?? 0) + 1,
      );
      log(`failover: ${candidate.provider.id}/${candidate.row.model} answered ${result.status || 'unreachable'} (${result.code})`);
    }
  }
  log('no candidate served the request');
  return null;
}

// ---------------------------------------------------------------------------
// The HTTP server: GET /v1/models, POST /v1/chat/completions, POST /v1/messages.
// ---------------------------------------------------------------------------

const brief = readBrief(PROMPT_FILE);
const catalog: Catalog = buildCatalog(brief);
/** The free model ids the catalog registers, for GET /v1/models. */
const freeModelIds: string[] = (() => {
  const ids = new Set<string>();
  for (const provider of catalog.providers) {
    for (const model of provider.models) if (model.free) ids.add(model.id);
  }
  return [...ids].sort();
})();
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;

/** The requirements of the request now arriving: the manifest's row, else the body's own. */
function requirementsFor(clientApi: 'openai' | 'anthropic', body: Json): Requirement {
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined && entry.api === clientApi) return entry.requires;
  // Fallback: read what the body itself asks for.
  const canon = clientApi === 'openai' ? canonFromOpenAI(body) : canonFromAnthropic(body);
  const requires: Requirement = {};
  if (canon.tools.length > 0) requires.tools = true;
  if (canon.jsonMode) requires.json = true;
  for (const message of canon.messages) {
    if (message.role !== 'user') continue;
    if (message.parts.some((p) => p.kind === 'image')) requires.vision = true;
  }
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
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    'content-length': Buffer.byteLength(text),
  });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header it got. */
function models(req: IncomingMessage, res: ServerResponse): void {
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: freeModelIds.map((id) => ({ type: 'model', id, display_name: id, created_at: FIXED_CREATED_AT })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: freeModelIds.map((id) => ({ id, object: 'model', created: FIXED_CREATED, owned_by: 'aggregator' })),
    });
  }
}

async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor('openai', body);
  const canon = canonFromOpenAI(body);
  const served = await serve(brief, 'openai', requires, canon);
  if (served === null) {
    return sendError(res, 'openai', 503, 'no_free_model', 'no free model could serve this request');
  }
  if (canon.stream) sendSse(res, openAIStreamAnswer(served.model, served.answer));
  else sendJson(res, 200, openAIJsonAnswer(served.model, served.answer));
}

async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor('anthropic', body);
  const canon = canonFromAnthropic(body);
  const served = await serve(brief, 'anthropic', requires, canon);
  if (served === null) {
    return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model could serve this request');
  }
  if (canon.stream) sendSse(res, anthropicStreamAnswer(served.model, served.answer));
  else sendJson(res, 200, anthropicJsonAnswer(served.model, served.answer));
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

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number): Promise<void> {
  let lastError = '';
  for (let attempt = 0; attempt < REGISTER_RETRIES; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') {
        log(`registered: the workload holds ${reply.requests} requests`);
      }
      return;
    } catch (err) {
      lastError = (err as Error).message;
      await sleep(REGISTER_RETRY_MS);
    }
  }
  throw new Error(`register failed after ${REGISTER_RETRIES} attempts: ${lastError}`);
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
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
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
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${freeModelIds.length} free models`);
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
