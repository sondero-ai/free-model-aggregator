// Translation between the two client wires and the three upstream wires.
//
// A client request is first read into one canonical shape: its system text, its last user text,
// its last image, its tools in order, its output cap, its stop sequences and whether it asked for
// JSON mode. The provider's parser echoes exactly those fields back, and the trial compares that
// echo with the client's body, so the canonical shape is what every upstream builder carries.
//
// Upstream calls are always made without streaming: one JSON answer carries the whole text and its
// usage, which the entry then reports back in the client's wire, as JSON or as the event sequence
// that wire expects. Nothing here reads a clock or draws randomness.

import { isRecord, type Json, type Wire } from './brief.ts';

export interface ToolDef {
  name: string;
  description: string;
  parameters: Json;
}

export interface ClientRequest {
  api: 'openai' | 'anthropic';
  stream: boolean;
  systemText: string;
  userText: string;
  image: string | null;
  tools: ToolDef[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | null;
}

export interface UpstreamRequest {
  path: string;
  body: Json;
}

export interface UpstreamAnswer {
  text: string;
  input: number;
  output: number;
  stop: string | null;
}

const FIXED_CREATED = 1767225600;
const OPENAI_ID = 'chatcmpl-aggregator';
const ANTHROPIC_ID = 'msg_aggregator';

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function numberOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

/** A string's text, or an array of text blocks joined with a newline, the way the env reads it. */
export function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  return parts.join('\n');
}

/** A stop field as the env reads it: one string, or an array of strings, in order. */
function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

function toolsOfOpenAI(body: Json): ToolDef[] {
  if (!Array.isArray(body.tools)) return [];
  const out: ToolDef[] = [];
  for (const tool of body.tools) {
    if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') continue;
    out.push({
      name: tool.function.name,
      description: stringOf(tool.function.description),
      parameters: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
    });
  }
  return out;
}

function toolsOfAnthropic(body: Json): ToolDef[] {
  if (!Array.isArray(body.tools)) return [];
  const out: ToolDef[] = [];
  for (const tool of body.tools) {
    if (!isRecord(tool) || typeof tool.name !== 'string') continue;
    out.push({
      name: tool.name,
      description: stringOf(tool.description),
      parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
    });
  }
  return out;
}

/** An Anthropic image block's src as a data URL or a plain URL. */
function imageOfAnthropic(content: unknown): string | null {
  if (!Array.isArray(content)) return null;
  let image: string | null = null;
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'image' || !isRecord(block.source)) continue;
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      image = `data:${media};base64,${source.data}`;
    } else if (source.type === 'url' && typeof source.url === 'string') {
      image = source.url;
    }
  }
  return image;
}

/** Reads either client body into the one shape whose echo the provider must reproduce. */
export function readClient(api: 'openai' | 'anthropic', body: Json): ClientRequest {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  let systemText = '';
  let userText = '';
  let image: string | null = null;
  if (api === 'openai') {
    const systemParts: string[] = [];
    for (const message of messages) {
      if (message.role === 'system') {
        const text = blocksText(message.content);
        if (text !== '') systemParts.push(text);
      } else if (message.role === 'user') {
        userText = blocksText(message.content);
        if (Array.isArray(message.content)) {
          for (const block of message.content) {
            if (isRecord(block) && block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') image = block.image_url.url;
          }
        }
      }
    }
    systemText = systemParts.join('\n');
    const format = isRecord(body.response_format) ? body.response_format.type : undefined;
    return {
      api,
      stream: body.stream === true,
      systemText,
      userText,
      image,
      tools: toolsOfOpenAI(body),
      maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
      stops: stringList(body.stop),
      jsonMode: format === 'json_object' || format === 'json_schema',
      temperature: numberOrNull(body.temperature),
    };
  }
  systemText = blocksText(body.system);
  for (const message of messages) {
    if (message.role !== 'user') continue;
    userText = blocksText(message.content);
    image = imageOfAnthropic(message.content) ?? image;
  }
  return {
    api,
    stream: body.stream === true,
    systemText,
    userText,
    image,
    tools: toolsOfAnthropic(body),
    maxTokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    temperature: numberOrNull(body.temperature),
  };
}

function imageBlock(dataUrl: string): Json {
  const data = /^data:([^,]*);base64,(.*)$/s.exec(dataUrl);
  if (data !== null) {
    const media = data[1] === '' ? 'image/png' : data[1];
    return { type: 'image', source: { type: 'base64', media_type: media, data: data[2] } };
  }
  return { type: 'image', source: { type: 'url', url: dataUrl } };
}

function userContentOpenAI(req: ClientRequest): unknown {
  if (req.image === null) return req.userText;
  return [
    { type: 'text', text: req.userText },
    { type: 'image_url', image_url: { url: req.image } },
  ];
}

function userContentAnthropic(req: ClientRequest): unknown {
  if (req.image === null) return req.userText;
  return [{ type: 'text', text: req.userText }, imageBlock(req.image)];
}

function geminiParts(req: ClientRequest): Json[] {
  const parts: Json[] = [{ text: req.userText }];
  if (req.image !== null) {
    const data = /^data:([^,]*);base64,(.*)$/s.exec(req.image);
    if (data !== null) parts.push({ inlineData: { mimeType: data[1] === '' ? 'image/png' : data[1], data: data[2] } });
    else parts.push({ fileData: { fileUri: req.image } });
  }
  return parts;
}

/** One canonical request as the provider's own wire expects it. */
export function buildUpstream(wire: Wire, model: string, req: ClientRequest): UpstreamRequest {
  if (wire === 'openai') {
    const messages: Json[] = [];
    if (req.systemText !== '') messages.push({ role: 'system', content: req.systemText });
    messages.push({ role: 'user', content: userContentOpenAI(req) });
    const body: Json = { model, messages, stream: false };
    if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
    if (req.temperature !== null) body.temperature = req.temperature;
    if (req.stops.length > 0) body.stop = req.stops;
    if (req.tools.length > 0) {
      body.tools = req.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
    }
    if (req.jsonMode) body.response_format = { type: 'json_object' };
    return { path: '/v1/chat/completions', body };
  }
  if (wire === 'anthropic') {
    const body: Json = {
      model,
      messages: [{ role: 'user', content: userContentAnthropic(req) }],
      stream: false,
    };
    if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
    if (req.systemText !== '') body.system = req.systemText;
    if (req.temperature !== null) body.temperature = req.temperature;
    if (req.stops.length > 0) body.stop_sequences = req.stops;
    if (req.tools.length > 0) {
      body.tools = req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
    }
    return { path: '/v1/messages', body };
  }
  const generationConfig: Json = {};
  if (req.maxTokens !== null) generationConfig.maxOutputTokens = req.maxTokens;
  if (req.temperature !== null) generationConfig.temperature = req.temperature;
  if (req.stops.length > 0) generationConfig.stopSequences = req.stops;
  if (req.jsonMode) generationConfig.responseMimeType = 'application/json';
  const body: Json = { contents: [{ role: 'user', parts: geminiParts(req) }], generationConfig };
  if (req.systemText !== '') body.systemInstruction = { parts: [{ text: req.systemText }] };
  if (req.tools.length > 0) {
    body.tools = [{ functionDeclarations: req.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  return { path: `/v1beta/models/${encodeURIComponent(model)}:generateContent`, body };
}

/** A provider's JSON answer, read for its text, its usage and its stop reason. */
export function parseAnswer(wire: Wire, value: unknown): UpstreamAnswer | null {
  if (!isRecord(value)) return null;
  if (wire === 'openai') {
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) return null;
    const message = isRecord(choice.message) ? choice.message : {};
    const content = typeof message.content === 'string' ? message.content : blocksText(message.content);
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text: content,
      input: numberOrZero(usage.prompt_tokens),
      output: numberOrZero(usage.completion_tokens),
      stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : null,
    };
  }
  if (wire === 'anthropic') {
    const content = Array.isArray(value.content) ? value.content : [];
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text: content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join(''),
      input: numberOrZero(usage.input_tokens),
      output: numberOrZero(usage.output_tokens),
      stop: typeof value.stop_reason === 'string' ? value.stop_reason : null,
    };
  }
  const candidates = Array.isArray(value.candidates) ? value.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const usage = isRecord(value.usageMetadata) ? value.usageMetadata : {};
  return {
    text: parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join(''),
    input: numberOrZero(usage.promptTokenCount),
    output: numberOrZero(usage.candidatesTokenCount),
    stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : null,
  };
}

const STOP_FAMILY: Record<string, number> = {
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
const OPENAI_STOP = ['stop', 'length', 'tool_calls', 'content_filter'];
const ANTHROPIC_STOP = ['end_turn', 'max_tokens', 'tool_use', 'refusal'];

function stopFamily(stop: string | null): number {
  return stop !== null && STOP_FAMILY[stop] !== undefined ? STOP_FAMILY[stop] : 0;
}

function usageJson(answer: UpstreamAnswer): Json {
  return { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output };
}

export function openAIJson(model: string, answer: UpstreamAnswer): Json {
  return {
    id: OPENAI_ID,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: OPENAI_STOP[stopFamily(answer.stop)], logprobs: null }],
    usage: usageJson(answer),
  };
}

export function openAISse(model: string, answer: UpstreamAnswer): string {
  const frame = (value: Json): string => `data: ${JSON.stringify(value)}\n\n`;
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: OPENAI_ID,
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  return (
    frame(chunk({ role: 'assistant', content: '' }, null)) +
    frame(chunk({ content: answer.text }, null)) +
    frame(chunk({}, OPENAI_STOP[stopFamily(answer.stop)])) +
    frame({ id: OPENAI_ID, object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: usageJson(answer) }) +
    'data: [DONE]\n\n'
  );
}

export function anthropicJson(model: string, answer: UpstreamAnswer): Json {
  return {
    id: ANTHROPIC_ID,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: ANTHROPIC_STOP[stopFamily(answer.stop)],
    stop_sequence: null,
    usage: { input_tokens: answer.input, output_tokens: answer.output },
  };
}

export function anthropicSse(model: string, answer: UpstreamAnswer): string {
  const frame = (event: string, value: Json): string => `event: ${event}\ndata: ${JSON.stringify(value)}\n\n`;
  return (
    frame('message_start', {
      type: 'message_start',
      message: {
        id: ANTHROPIC_ID,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.input, output_tokens: 0 },
      },
    }) +
    frame('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }) +
    frame('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: answer.text } }) +
    frame('content_block_stop', { type: 'content_block_stop', index: 0 }) +
    frame('message_delta', { type: 'message_delta', delta: { stop_reason: ANTHROPIC_STOP[stopFamily(answer.stop)], stop_sequence: null }, usage: { output_tokens: answer.output } }) +
    frame('message_stop', { type: 'message_stop' })
  );
}

export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}
