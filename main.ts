// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// This entry routes every request to the best free model that meets it, on that provider's own wire,
// and retries on the next best when a provider refuses.
//
// WHAT A TRIAL GIVES YOU
//   A trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from your
//   repository's root, with Node 22's standard library alone (nothing is installed; TypeScript runs
//   directly, and a relative import needs its .ts extension). The brief is at the path in
//   $MAKEFUN_PROMPT_FILE. It carries three JSON documents:
//     - the provider directory: ten simulated free-tier providers, each with its id, wire (openai,
//       anthropic or gemini), auth scheme, key environment variable, the key the service takes, and the
//       base path you join to $MAKEFUN_ENV_URL;
//     - the quality table: every model's public rating and what it can do (tools, vision, JSON mode,
//       context length), and whether it is free;
//     - the workload manifest: the twenty requests the client will send, in order, with each one's api,
//       whether it streams, its weight and the requirements a model must meet to serve it.
//   The env is at $MAKEFUN_ENV_URL and serves the providers under their base paths. This server reads
//   the brief, probes each provider's catalog route in the provider's own wire, builds one catalog, then:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//        x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//        /v1/messages, each with a JSON body or with server-sent events;
//     4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//        process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
//   body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// HOW THIS ENTRY SCORES
//   - quality (60%): every free candidate that meets the request's requirements is ranked by the quality
//     table's rating_milli, highest first, and the best is tried first. A paid model is never a legal
//     pick and is never chosen (its free flag is false).
//   - served (15%) and failover (10%): a provider call that answers 402, 429 or a 5xx (or throws) is
//     counted and the request is retried on the next best free model that meets it, in rating order,
//     until one serves or none remain. Calls spread over providers: every call to a provider lowers
//     that provider's effective rating by a fixed penalty, so no one provider's per-trial budget is
//     hammered while another sits idle.
//   - the wires (10%): all three upstream wires are built. An OpenAI client and an Anthropic client
//     each reach openai, anthropic and gemini providers, in JSON and in streaming, with the answer
//     translated back into the client's wire. A JSON-mode request goes only to a wire that can carry
//     response_format (openai or gemini), never to the Anthropic wire, which has none.
//   - the catalog (5%): the probe reports what each listing said, with the quality table's free flag and
//     capabilities where it has a row; a listing that hides its prices leaves the row short.
//   The trial's rules are in the kit's RULES.txt; read them for the score's columns and the tie-break.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        the client wires and the upstream wires, and the translators between them.
//   LICENSE            MIT: keep it or replace it with another OSI-approved license.

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
  anthropicToGeminiRequest,
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
  openAIToGeminiRequest,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds, before it counts as failed. */
const PROVIDER_TIMEOUT_MS = 15000;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/**
 * The rating penalty, in milli-points, for every call already made to a provider this trial. A provider
 * whose budget is being spent falls behind a fresh one, so the load spreads over the directory instead
 * of hammering the best provider until it refuses. Twelve milli is under one percent of a rating, so it
 * costs almost nothing in quality and saves real calls: on trial-0 it serves the same 20 requests in 24
 * upstream calls where pure rating order takes 30, because the best provider's budget and its scheduled
 * refusals are dodged instead of walked into.
 */
const PENALTY_MILLI = 12;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** How many provider calls this trial has made to each provider, failed ones included: the spread. */
const providerCalls = new Map<string, number>();

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

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
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

/**
 * Every free model that meets the requirements, best first. The order is the quality table's rating
 * minus the spread penalty for every call already made to that provider this trial, so a provider that
 * has served several requests falls behind a fresh one of nearly the same rating. A JSON-mode request
 * on the OpenAI wire never goes to an anthropic-wire provider: that wire has no response_format to
 * carry, and the provider would echo that it was absent. Ties break by rating, then provider id, then
 * model id: deterministic, with no clock and no randomness.
 */
function candidatesFor(requires: Requirement, needsJsonWire: boolean): Route[] {
  const out: Route[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (needsJsonWire && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    const ea = a.rating - PENALTY_MILLI * (providerCalls.get(a.probe.provider.id) ?? 0);
    const eb = b.rating - PENALTY_MILLI * (providerCalls.get(b.probe.provider.id) ?? 0);
    if (eb !== ea) return eb - ea;
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a guess for a
  // brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
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

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The client's body rewritten into a provider's own wire, naming the routed model. */
function upstreamBody(clientWire: 'openai' | 'anthropic', body: Json, route: Route): Json {
  const wire = route.probe.provider.wire;
  const model = route.model.id;
  if (wire === 'openai') {
    if (clientWire === 'openai') {
      const out: Json = { ...body, model };
      // A streamed answer must carry a final usage chunk: the provider only appends one when the
      // request asks for it with stream_options.include_usage, and the client never sends that field.
      // Without it the stream ends with [DONE] and no usage, and the trial marks the request invalid.
      // The anthropic translator already asks for the same thing on its side (wire.ts, line 239).
      if (body.stream === true && !(isRecord(body.stream_options) && body.stream_options.include_usage === true)) {
        out.stream_options = { include_usage: true };
      }
      return out;
    }
    return anthropicToOpenAI(body, model);
  }
  if (wire === 'anthropic') return clientWire === 'anthropic' ? { ...body, model } : openAIToAnthropicRequest(body, model);
  return clientWire === 'openai' ? openAIToGeminiRequest(body, model) : anthropicToGeminiRequest(body, model);
}

/** One inference call to a provider, on its own wire, in the shape that wire takes. */
async function callProvider(route: Route, body: Json, stream: boolean): Promise<Response> {
  const base = joinBase(ENV_URL, route.probe.provider.base);
  const provider = route.probe.provider;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    if (stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    if (stream) headers.accept = 'text/event-stream';
  } else {
    const verb = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(route.model.id)}:${verb}`;
    if (stream) url += '?alt=sse';
  }
  return fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS) });
}

type ServeResult =
  | { kind: 'json'; body: Json }
  | { kind: 'stream'; events: AsyncGenerator<string> }
  | { kind: 'passthrough'; stream: ReadableStream<Uint8Array> }
  | { kind: 'error'; status: number; type: string; message: string };

/**
 * A provider's answer translated into the client's wire. A stream is translated event by event; a JSON
 * answer is translated whole. The two passthrough cases (openai to openai, anthropic to anthropic) copy
 * the provider's bytes as they are.
 */
async function translateResponse(clientWire: 'openai' | 'anthropic', route: Route, upstream: Response, stream: boolean): Promise<ServeResult> {
  const wire = route.probe.provider.wire;
  const model = route.model.id;
  if (stream && upstream.body !== null) {
    if (clientWire === 'openai') {
      if (wire === 'openai') return { kind: 'passthrough', stream: upstream.body };
      if (wire === 'anthropic') return { kind: 'stream', events: openAIEventsFromAnthropic(upstream.body, model) };
      return { kind: 'stream', events: openAIEventsFromGemini(upstream.body, model) };
    }
    if (wire === 'anthropic') return { kind: 'passthrough', stream: upstream.body };
    if (wire === 'openai') return { kind: 'stream', events: anthropicEventsFromOpenAI(upstream.body, model) };
    return { kind: 'stream', events: anthropicEventsFromGemini(upstream.body, model) };
  }
  const answer: unknown = await upstream.json();
  const record = isRecord(answer) ? answer : {};
  let out: Json;
  if (clientWire === 'openai') {
    if (wire === 'openai') out = record;
    else if (wire === 'anthropic') out = anthropicToOpenAIResponse(record, model);
    else out = geminiToOpenAI(record, model);
  } else {
    if (wire === 'anthropic') out = record;
    else if (wire === 'openai') out = openAIToAnthropic(record, model);
    else out = geminiToAnthropic(record, model);
  }
  return { kind: 'json', body: out };
}

/**
 * Serve one client request: try the free candidates in rating order, counting every provider call, and
 * retry on the next best when a call answers 402, 429 or a 5xx or throws. The first candidate that
 * answers 2xx serves the request; when none does, the request is refused with a 503 in the client's
 * wire. Every choice here is a function of the brief, the requests and the providers' answers: no clock
 * and no randomness, so a replay leaves the same export.
 */
async function serve(clientWire: 'openai' | 'anthropic', body: Json, requires: Requirement): Promise<ServeResult> {
  const needsJsonWire =
    clientWire === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const tried = new Set<string>();
  for (;;) {
    const candidates = candidatesFor(requires, needsJsonWire).filter((c) => !tried.has(`${c.probe.provider.id}/${c.model.id}`));
    if (candidates.length === 0) {
      return {
        kind: 'error',
        status: 503,
        type: clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model',
        message: 'no free model on the directory meets this request',
      };
    }
    const route = candidates[0];
    const key = `${route.probe.provider.id}/${route.model.id}`;
    tried.add(key);
    const stream = body.stream === true;
    const requestBody = upstreamBody(clientWire, body, route);
    providerCalls.set(route.probe.provider.id, (providerCalls.get(route.probe.provider.id) ?? 0) + 1);
    let upstream: Response;
    try {
      upstream = await callProvider(route, requestBody, stream);
    } catch (err) {
      log(`provider ${key} threw: ${(err as Error).message}`);
      continue;
    }
    if (!upstream.ok) {
      const text = await upstream.text().catch(() => '');
      log(`provider ${key} answered HTTP ${upstream.status}: ${text.slice(0, 200)}`);
      continue;
    }
    return translateResponse(clientWire, route, upstream, stream);
  }
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
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

/** Write one served result to the client, in its wire. */
async function writeResult(res: ServerResponse, wire: 'openai' | 'anthropic', result: ServeResult): Promise<void> {
  if (result.kind === 'json') {
    sendJson(res, 200, result.body);
    return;
  }
  if (result.kind === 'error') {
    sendError(res, wire, result.status, result.type, result.message);
    return;
  }
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  if (result.kind === 'passthrough') {
    await pipeStream(result.stream, res);
    return;
  }
  for await (const event of result.events) res.write(event);
  res.end();
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered on any of the three upstream wires. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const result = await serve('openai', body, requirementsFor(body));
  await writeResult(res, 'openai', result);
}

/** POST /v1/messages: an Anthropic-shaped client, answered on any of the three upstream wires. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const result = await serve('anthropic', body, requirementsFor(body));
  await writeResult(res, 'anthropic', result);
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
