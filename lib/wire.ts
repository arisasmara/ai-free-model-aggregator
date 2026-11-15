// The client wires and the upstream wires this entry speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This entry forwards each request to a
// provider on that provider's own wire and translates the answer back:
//
//   - the OpenAI path passes the client's body (with the routed model's id) to an openai-wire provider and
//     streams the provider's JSON or SSE straight back, the way an OpenAI-compatible gateway does;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the provider's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events;
//   - the Gemini path rewrites the request to generateContent (systemInstruction, contents, function
//     declarations, generationConfig) and rewrites the provider's answer into the client's wire, in JSON
//     and in streaming (SSE);
//   - the Anthropic upstream path rewrites the request to Messages (system, messages, tools, max_tokens)
//     and rewrites the provider's answer into the client's wire, in JSON and in streaming (SSE).
//
// Every translation is a pure function of its arguments: no clock and no randomness, so the same body and
// the same provider answer give the same bytes on a replay. The two model listings use fixed timestamps
// for the same reason: a `created` that moved between runs would be a nondeterministic document.
//
// The echo rule drives every translation: the provider parses the upstream body and echoes the system
// prompt, the user text, the tools, max_tokens, the stop sequences, the image and the JSON mode, and the
// client compares that echo with what it sent. A field a translation rewrites or drops loses the request.
// So a translation keeps every field the client sent, in the upstream wire's own spelling, and a JSON-mode
// request never goes to a wire that cannot carry it (the Anthropic wire has no response_format).

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** An error in Gemini's shape (the Generative Language API's error envelope). */
export function geminiError(code: number, message: string, status: string): Json {
  return { error: { code, message, status } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
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

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function openAIFinishOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
export function openAIFinishOfGemini(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY') return 'content_filter';
  if (finish === 'TOOL_CALLS') return 'tool_calls';
  return 'stop';
}

/** Gemini's finishReason as Anthropic's stop_reason. */
export function anthropicStopOfGemini(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'SAFETY') return 'refusal';
  if (finish === 'TOOL_CALLS') return 'tool_use';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A data URL's media type and base64 payload, or null when it is not one. */
function dataUrlOf(url: string): { media: string; data: string } | null {
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { media: m[1] !== '' ? m[1] : 'image/png', data: m[2] };
}

/** An OpenAI content part as an Anthropic content block: text or an inline image; null for the rest. */
function anthropicBlockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = dataUrlOf(url);
    if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data.media, data: data.data } };
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`: the system
 * message becomes the system prompt, image_url parts become image blocks, tool messages become
 * tool_result blocks grouped into the user message they answer, tool_calls become tool_use blocks, and
 * the output cap, temperature, top_p, stop sequences and streaming pass through. response_format has no
 * Anthropic spelling and is dropped: never send a JSON-mode request to this wire.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  /** tool_result blocks waiting for the user message they answer. */
  let pendingResults: Json[] = [];
  const flushResults = (): void => {
    if (pendingResults.length === 0) return;
    messages.push({ role: 'user', content: pendingResults });
    pendingResults = [];
  };
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'tool') {
      pendingResults.push({
        type: 'tool_result',
        tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
        content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content),
      });
      continue;
    }
    if (role === 'assistant') {
      flushResults();
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    // A user message: its text and image parts, with any tool results that precede it.
    flushResults();
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      const block = anthropicBlockOf(part);
      if (block !== null) blocks.push(block);
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  flushResults();

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  // Only carry the output cap when the client sent one: the provider echoes max_tokens, and the client
  // compares it with what it sent, so a default here would fail the echo on a body that named none.
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stop = body.stop;
  if (typeof stop === 'string' && stop !== '') out.stop_sequences = [stop];
  else if (Array.isArray(stop) && stop.length > 0) out.stop_sequences = stop.filter((s): s is string => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** An Anthropic Messages answer as a Chat Completions object that names `model`. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = textOf(content);
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    toolCalls.push({
      id: typeof block.id === 'string' ? block.id : 'call_unknown',
      type: 'function',
      function: {
        name: typeof block.name === 'string' ? block.name : '',
        arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
      },
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: typeof body.id === 'string' ? body.id : 'chatcmpl-anthropic',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
        finish_reason: openAIFinishOf(body.stop_reason),
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: numberOrZero(usage.input_tokens),
      completion_tokens: numberOrZero(usage.output_tokens),
      total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens),
    },
  };
}

/** One OpenAI message as Gemini contents, appended to `out`; tool messages become functionResponse parts. */
function openAIContentsInto(body: Json, out: Json[]): string {
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'tool') {
      const part: Json = {
        functionResponse: {
          name: '',
          response: { content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) },
        },
      };
      // Attach the result to the last user content so the provider's echo keeps the user text: a
      // functionResponse part carries no text, so the content's text stays the user's.
      const last = out.length > 0 ? out[out.length - 1] : null;
      if (last !== null && last.role === 'user' && Array.isArray(last.parts)) (last.parts as Json[]).push(part);
      else out.push({ role: 'user', parts: [part] });
      continue;
    }
    if (role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          functionCall: {
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: argumentsOf(call.function.arguments),
          },
        });
      }
      if (parts.length > 0) out.push({ role: 'model', parts });
      continue;
    }
    // A user message: text and image parts.
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const data = dataUrlOf(part.image_url.url);
          if (data !== null) parts.push({ inlineData: { mimeType: data.media, data: data.data } });
          else parts.push({ fileData: { fileUri: part.image_url.url } });
        }
      }
    }
    if (parts.length > 0) out.push({ role: 'user', parts });
  }
  return systemParts.join('\n');
}

/** One Anthropic message as Gemini contents, appended to `out`; tool results become functionResponse parts. */
function anthropicContentsInto(body: Json, out: Json[]): string {
  const system = textOf(body.system);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      out.push({ role: assistant ? 'model' : 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const parts: Json[] = [];
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'tool_use') {
          parts.push({
            functionCall: {
              name: typeof block.name === 'string' ? block.name : '',
              args: isRecord(block.input) ? block.input : {},
            },
          });
        }
      }
      if (parts.length > 0) out.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
      else if (block.type === 'image' && isRecord(block.source)) {
        const source = block.source;
        if (source.type === 'base64' && typeof source.data === 'string') {
          parts.push({ inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } });
        } else if (source.type === 'url' && typeof source.url === 'string') {
          parts.push({ fileData: { fileUri: source.url } });
        }
      } else if (block.type === 'tool_result') {
        parts.push({
          functionResponse: {
            name: '',
            response: { content: toolResultText(block.content) },
          },
        });
      }
    }
    if (parts.length > 0) out.push({ role: 'user', parts });
  }
  return system;
}

/** OpenAI tool definitions as Gemini function declarations. */
function geminiToolsOfOpenAI(body: Json): Json[] | null {
  if (!Array.isArray(body.tools)) return null;
  const declarations = body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'OBJECT', properties: {} },
      },
    ];
  });
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : null;
}

/** Anthropic tool definitions as Gemini function declarations. */
function geminiToolsOfAnthropic(body: Json): Json[] | null {
  if (!Array.isArray(body.tools)) return null;
  const declarations = body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'OBJECT', properties: {} },
      },
    ];
  });
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : null;
}

/** The generationConfig for a request: output cap, temperature, top_p, stop sequences, JSON mode. */
function geminiConfigOf(maxTokens: number | null, body: Json, jsonMode: boolean): Json {
  const config: Json = {};
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.top_p = body.top_p;
  const stops = geminiStopsOf(body);
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  return config;
}

/** A request's stop sequences, in either client's spelling. */
function geminiStopsOf(body: Json): string[] {
  if (typeof body.stop === 'string' && body.stop !== '') return [body.stop];
  if (Array.isArray(body.stop)) return body.stop.filter((s): s is string => typeof s === 'string');
  if (Array.isArray(body.stop_sequences)) return body.stop_sequences.filter((s): s is string => typeof s === 'string');
  return [];
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body: the system message becomes the
 * systemInstruction, messages become contents, tools become functionDeclarations, and the output cap,
 * temperature, top_p, stop sequences and JSON mode become the generationConfig.
 */
export function openAIToGeminiRequest(body: Json, model: string): Json {
  const contents: Json[] = [];
  const system = openAIContentsInto(body, contents);
  const out: Json = { model, contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const tools = geminiToolsOfOpenAI(body);
  if (tools !== null) out.tools = tools;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const config = geminiConfigOf(maxTokens, body, format === 'json_object' || format === 'json_schema');
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body: the system prompt becomes the
 * systemInstruction, messages become contents, tools become functionDeclarations, and the output cap,
 * temperature, top_p and stop sequences become the generationConfig.
 */
export function anthropicToGeminiRequest(body: Json, model: string): Json {
  const contents: Json[] = [];
  const system = anthropicContentsInto(body, contents);
  const out: Json = { model, contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const tools = geminiToolsOfAnthropic(body);
  if (tools !== null) out.tools = tools;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : null;
  const config = geminiConfigOf(maxTokens, body, false);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** The text of a Gemini candidate's parts, and any functionCall parts as OpenAI tool calls. */
function geminiContentOf(body: Json): { text: string; toolCalls: Json[] } {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const texts: string[] = [];
  const toolCalls: Json[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') texts.push(part.text);
    else if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: `call_${toolCalls.length}`,
        type: 'function',
        function: {
          name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
          arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
        },
      });
    }
  }
  return { text: texts.join(''), toolCalls };
}

/** A Gemini candidate's finishReason, or null when the answer names none. */
function geminiFinishOf(body: Json): string | null {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  return typeof candidate.finishReason === 'string' ? candidate.finishReason : null;
}

/** A Gemini answer's usageMetadata as OpenAI usage, or nulls when it names none. */
function geminiUsageOf(body: Json): { prompt_tokens: number; completion_tokens: number; total_tokens: number } {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const input = numberOrZero(meta.promptTokenCount);
  const output = numberOrZero(meta.candidatesTokenCount);
  return { prompt_tokens: input, completion_tokens: output, total_tokens: input + output };
}

/** A Gemini generateContent answer as a Chat Completions object that names `model`. */
export function geminiToOpenAI(body: Json, model: string): Json {
  const { text, toolCalls } = geminiContentOf(body);
  const usage = geminiUsageOf(body);
  return {
    id: `chatcmpl-${typeof body.modelVersion === 'string' ? body.modelVersion : model}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
        finish_reason: openAIFinishOfGemini(geminiFinishOf(body)),
        logprobs: null,
      },
    ],
    usage,
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object that names `model`. */
export function geminiToAnthropic(body: Json, model: string): Json {
  const { text, toolCalls } = geminiContentOf(body);
  const content: Json[] = [];
  if (text !== '') content.push({ type: 'text', text });
  for (const call of toolCalls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = geminiUsageOf(body);
  return {
    id: `msg_${typeof body.modelVersion === 'string' ? body.modelVersion : 'gemini'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStopOfGemini(geminiFinishOf(body)),
    stop_sequence: null,
    usage: { input_tokens: usage.prompt_tokens, output_tokens: usage.completion_tokens },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Read a provider's SSE body frame by frame, yielding each frame's parsed JSON (null when unparsable). */
async function* sseFrames(upstream: ReadableStream<Uint8Array>): AsyncGenerator<unknown> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const line of frame.split('\n')) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          const payload = trimmed.slice(5).trim();
          if (payload === '[DONE]') {
            yield '[DONE]';
            continue;
          }
          try {
            yield JSON.parse(payload);
          } catch {
            yield null;
          }
        }
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      for (const line of buffer.split('\n')) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data:')) continue;
        const payload = trimmed.slice(5).trim();
        if (payload === '[DONE]') yield '[DONE]';
        else {
          try {
            yield JSON.parse(payload);
          } catch {
            yield null;
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

/**
 * A Gemini streamGenerateContent SSE stream as the Anthropic event sequence: the candidate parts' text
 * becomes text deltas, a finishReason frame closes the blocks and carries the usage, and the sequence ends
 * with message_delta and message_stop.
 */
export async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;
  let pendingStop: string | null = null;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  for await (const frame of sseFrames(upstream)) {
    if (frame === '[DONE]') {
      yield* startEvents();
      yield* finishEvents(pendingStop ?? 'end_turn');
      continue;
    }
    if (!isRecord(frame)) continue;
    const usage = geminiUsageOf(frame);
    if (usage.prompt_tokens !== 0) inputTokens = usage.prompt_tokens;
    if (usage.completion_tokens !== 0) outputTokens = usage.completion_tokens;
    const finish = geminiFinishOf(frame);
    if (finish !== null) pendingStop = anthropicStopOfGemini(finish);
    const { text, toolCalls } = geminiContentOf(frame);
    yield* startEvents();
    if (text !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        yield sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } });
      }
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text } });
    }
    for (const call of toolCalls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      const upstreamIndex = toolIndexes.size;
      let index = toolIndexes.get(upstreamIndex);
      if (index === undefined) {
        if (textIndex !== -1) {
          yield sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex });
          textIndex = -1;
        }
        index = nextIndex;
        nextIndex += 1;
        toolIndexes.set(upstreamIndex, index);
        yield sseEvent('content_block_start', {
          type: 'content_block_start',
          index,
          content_block: {
            type: 'tool_use',
            id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
            name: typeof call.function.name === 'string' ? call.function.name : '',
            input: {},
          },
        });
      }
      const args = typeof call.function.arguments === 'string' ? call.function.arguments : '';
      if (args !== '') {
        yield sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: args } });
      }
    }
    if (finish !== null) yield* finishEvents(pendingStop ?? 'end_turn');
  }
  if (!finished) {
    yield* startEvents();
    yield* finishEvents(pendingStop ?? 'end_turn');
  }
}

/**
 * A Gemini streamGenerateContent SSE stream as Chat Completions SSE: the candidate parts' text becomes
 * content deltas, a finishReason frame becomes the finish chunk, the usageMetadata becomes the usage
 * chunk, and the stream ends with [DONE].
 */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const chunk = (delta: Json, finish: string | null): string =>
    sseEvent(null, { id: 'chatcmpl-gemini', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
  let sentRole = false;
  let finished = false;
  for await (const frame of sseFrames(upstream)) {
    if (frame === '[DONE]') break;
    if (!isRecord(frame)) continue;
    const { text, toolCalls } = geminiContentOf(frame);
    const finish = geminiFinishOf(frame);
    const usage = geminiUsageOf(frame);
    if (!sentRole) {
      sentRole = true;
      yield chunk({ role: 'assistant', content: '' }, null);
    }
    if (text !== '') yield chunk({ content: text }, null);
    for (const call of toolCalls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      yield chunk(
        {
          tool_calls: [
            {
              index: 0,
              id: typeof call.id === 'string' ? call.id : '',
              type: 'function',
              function: { name: typeof call.function.name === 'string' ? call.function.name : '', arguments: typeof call.function.arguments === 'string' ? call.function.arguments : '' },
            },
          ],
        },
        null,
      );
    }
    if (finish !== null && !finished) {
      finished = true;
      yield chunk({}, openAIFinishOfGemini(finish));
      if (usage.prompt_tokens !== 0 || usage.completion_tokens !== 0) {
        yield sseEvent(null, { id: 'chatcmpl-gemini', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage });
      }
    }
  }
  if (!finished) {
    if (!sentRole) yield chunk({ role: 'assistant', content: '' }, null);
    yield chunk({}, 'stop');
  }
  yield 'data: [DONE]\n\n';
}

/**
 * An Anthropic Messages SSE stream as Chat Completions SSE: text deltas become content deltas,
 * input_json_delta events accumulate into tool_calls, and message_delta becomes the finish chunk with
 * the usage, ending with [DONE].
 */
export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const chunk = (delta: Json, finish: string | null): string =>
    sseEvent(null, { id: 'chatcmpl-anthropic', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
  let sentRole = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;
  /** tool_use blocks by their content-block index, accumulating their input JSON. */
  const tools = new Map<number, { id: string; name: string; args: string }>();
  const flushTools = function* (): Generator<string> {
    for (const [index, tool] of [...tools.entries()].sort((a, b) => a[0] - b[0])) {
      yield chunk(
        {
          tool_calls: [
            {
              index,
              id: tool.id,
              type: 'function',
              function: { name: tool.name, arguments: tool.args },
            },
          ],
        },
        null,
      );
    }
    tools.clear();
  };
  for await (const frame of sseFrames(upstream)) {
    if (frame === '[DONE]') break;
    if (!isRecord(frame)) continue;
    // Anthropic SSE frames carry their event name in the `event` field only when sseFrames kept it;
    // this reader yields the parsed data, so the event is read from the data's `type` member.
    const type = frame.type;
    if (type === 'message_start' && isRecord(frame.message) && isRecord(frame.message.usage)) {
      inputTokens = numberOrZero(frame.message.usage.input_tokens);
    } else if (type === 'content_block_start' && typeof frame.index === 'number' && isRecord(frame.content_block)) {
      const block = frame.content_block;
      if (block.type === 'tool_use') {
        tools.set(frame.index, {
          id: typeof block.id === 'string' ? block.id : `call_${frame.index}`,
          name: typeof block.name === 'string' ? block.name : '',
          args: '',
        });
      }
    } else if (type === 'content_block_delta' && typeof frame.index === 'number' && isRecord(frame.delta)) {
      const delta = frame.delta;
      if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
        if (!sentRole) {
          sentRole = true;
          yield chunk({ role: 'assistant', content: '' }, null);
        }
        yield chunk({ content: delta.text }, null);
      } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string') {
        const tool = tools.get(frame.index);
        if (tool !== undefined) tool.args += delta.partial_json;
      }
    } else if (type === 'message_delta' && isRecord(frame.delta) && isRecord(frame.usage)) {
      outputTokens = numberOrZero(frame.usage.output_tokens);
      if (!finished) {
        finished = true;
        yield* flushTools();
        yield chunk({}, openAIFinishOf(frame.delta.stop_reason));
        yield sseEvent(null, {
          id: 'chatcmpl-anthropic',
          object: 'chat.completion.chunk',
          created: FIXED_OPENAI_CREATED,
          model,
          choices: [],
          usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
        });
      }
    }
  }
  if (!finished) {
    if (!sentRole) yield chunk({ role: 'assistant', content: '' }, null);
    yield* flushTools();
    yield chunk({}, 'stop');
  }
  yield 'data: [DONE]\n\n';
}
