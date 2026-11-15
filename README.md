# free-tier-gateway

One endpoint in front of ten free-tier model providers. It speaks the OpenAI Chat Completions API and the
Anthropic Messages API, in JSON and in server-sent events, and routes every request to the highest-rated
free model on the directory that can serve it. No dependencies, no build step: Node 22 runs the TypeScript
directly.

## What it does

On start it reads the brief it is given, asks every provider for its model list in that provider's own wire
(OpenAI, Anthropic or Gemini), and builds one catalog document from the answers. It then listens on a free
port, registers that catalog with the env, and serves:

| Route | What it answers |
| --- | --- |
| `POST /v1/chat/completions` | an OpenAI Chat Completions request, JSON or `stream: true` SSE |
| `POST /v1/messages` | an Anthropic Messages request, JSON or `stream: true` SSE |
| `GET /v1/models` | OpenAI's list shape for `Authorization: Bearer`, Anthropic's for `x-api-key` |

Each request goes to the free model with the best rating on the quality table that meets what the request
asked for (tools, images, JSON mode, a context floor). The server translates the request into that
provider's wire, calls it, and translates the answer back into the wire the client used: the system prompt,
the user's text and images, tool definitions and their results, the output cap, the stop sequences, JSON
mode, the stop reason and the usage counts all make the round trip.

A provider that refuses (402, 429, 5xx, or a 400 for that model) is recorded and the request moves on to the
next-best free model, so one spent free tier never takes a request down with it. A refusal whose own message
says the allowance is gone or the service is out marks that provider done for the trial; a one-call refusal
does not.

## Run it

```sh
npm start               # or: node main.ts
node main.ts --list     # print the catalog document and exit 0
```

Under a trial (`MAKEFUN_ENV_URL` and `MAKEFUN_PROMPT_FILE` set) it binds a port the OS picks, registers its
catalog with the env, serves, and exits 0 when the env says the work is done.

Run with no trial environment it still serves, on `$PORT` (or 8787): there is no brief to read and no env to
register with, so the model list is empty and a chat request gets a 503 saying so. That is enough to start it
by hand and see that the routes answer; `--list` in the same state prints `{"providers":[]}` and exits 0.

The catalog `--list` prints is the same object the register body carries, byte for byte: it is built once and
used for both.

## Test

```sh
node test/e2e.mjs
```

The test stands up a mock env with four providers on the three real wires (one of which allows a single call
before refusing with 429), runs the entry against it, and asserts what a served request has to look like: the
catalog `--list` prints equals the register body, `GET /v1/models` lists exactly the registered free ids in
both shapes, every request's provider echo comes back whole (system prompt, user text, image, tool names,
`max_tokens`, stop sequences, JSON mode), the stop reason and usage match the provider's, the SSE event order
is the one each client wire defines, a refused first call still ends in an answer from the next-best model,
and the process exits 0 when the env says the work is done. It prints `63/63 checks passed` and exits 0.

## Layout

```
main.ts             the server: brief, probes, catalog, routes, register, work poll
lib/providers.ts    the brief's shapes and reader, auth headers, the catalog probe and the catalog
lib/wire.ts         the normalized request, and the OpenAI / Anthropic / Gemini wires in both directions
test/e2e.mjs        the end-to-end check above
```

## Determinism

Nothing reads a wall clock or draws an unseeded random number. Response ids come from a request counter, the
`created` fields are fixed constants, and the catalog's `base_url` is the provider's path rather than the
env's address (which changes between a live pass and its replay). Every choice is a function of the brief,
the requests and the providers' answers, so a replay of a trial leaves the same bytes.

## License

MIT. See [LICENSE](LICENSE).
