# free-tier-gateway

One Node 22 server speaking the OpenAI Chat Completions and Anthropic Messages APIs (JSON and SSE) over ten
free-tier providers, routing each request to the highest-rated free model that meets it, with failover.
Node's standard library only; TypeScript runs directly.

## Status

Updated 2026-11-15.

- `node test/e2e.mjs` passes 67/67 (standalone start, then a mock env with four providers on the real three
  wires, all response checks).
- `grade-local --instances 3 --worktree` scores **99 on all three practice trials**: catalog/served/wire/
  failover 100, quality 99, ~26.7 upstream calls, replay matched byte for byte.
- Quality 99 is the arithmetic ceiling: the bound ignores per-trial budgets, and the 1475 tier has 10-12
  calls against 20 requests that want it; optimal in-budget assignment also scores 99.
- Open: `transcripts_present` fails `makefun check` until the user bundles and uploads sessions.

## Run

`npm start` (or `node main.ts`) serves: a trial when `MAKEFUN_ENV_URL` and `MAKEFUN_PROMPT_FILE` are set, else
standalone on `$PORT` or 8787 with an empty model list. `node main.ts --list` prints the catalog and exits 0;
`npm run test:e2e` is the end-to-end check.

## Map

- `main.ts` — probe fan-out, catalog, routes, register, `/work` poll. `candidatesFor` ranks, `route` walks
  the failover, `callProvider` is the only place a provider is dialled.
- `lib/providers.ts` — brief documents found by `schema`, parsers, auth headers, `probeProvider`,
  `buildCatalog`, `meetsRequirements`.
- `lib/wire.ts` — `normalizeRequest`, then `buildUpstreamBody` per wire and `readUpstream` /
  `openAICompletion` / `anthropicMessage` / `*StreamFrames` back out.

## Decisions

- Catalog `base_url` is the provider's path, never joined to the env origin: the port changes between a live
  pass and its replay, and the catalog sits inside the replayed state export. The scorer matches the suffix.
- No clock, no randomness: response ids from a counter, `created` a constant.
- Standalone (no env) is a served mode, not an error: `npm start` with nothing set binds `$PORT` or 8787 and
  answers (200 empty `/v1/models`, 503 naming the missing env on chat). Exiting 1 there broke the hand-off.
- The env's echo comparison is the whole game: every echoed field must survive into the provider's wire
  unchanged, so a wire that cannot carry one ranks *after* one that can, whatever the rating.
- A 400 drops that model only; a refusal matching `FINAL_REFUSAL`, or three in a row, marks a provider out.
- Spreading load inside a rating band was tried and made calls worse (29.3 vs 26.7); strict rating order
  with refusals-then-calls tie-breaks is kept.
