# Free-tier model gateway

One server that speaks the OpenAI Chat Completions API and the Anthropic Messages API, in JSON and in server-sent events, over a directory of free-tier providers.

It reads the provider directory, quality table and workload from a brief, probes each provider's catalog in that provider's own wire (OpenAI, Anthropic or Gemini), registers the catalog it found, and routes every request to the highest-rated free model that meets the request's requirements. When a provider answers 402 or 429, it fails over to the next candidate. A 400 from one model can still fall through to another model on the same provider. Paid models are never chosen.

## Run

Requires Node 22 or later. Nothing is installed at runtime; TypeScript runs directly.

```sh
# Print the catalog document (no brief and no env: an empty provider list)
node main.ts --list

# Serve a trial
MAKEFUN_ENV_URL=... MAKEFUN_PROMPT_FILE=... node main.ts
```

`GET /v1/models` answers in the wire the request's auth header names: `Authorization: Bearer` for the OpenAI list, `x-api-key` for the Anthropic list. `POST /v1/chat/completions` and `POST /v1/messages` accept JSON bodies and `stream: true` for server-sent events.

## License

MIT. See [LICENSE](LICENSE).
