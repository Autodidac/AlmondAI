# Configuring External Teacher Backends

AlmondAI can fan-in answers from a variety of chat-style large language models so
training runs are not limited to the built-in student or the default GPT
integration. This document lists the supported backends, the environment
variables required to connect to them, and how to annotate incoming supervision
so the curator keeps multiple viewpoints from the same prompt.

## Selecting a backend

Set the following environment variables before launching the runtime (or the
console harness under `AlmondShell/examples/AlmondAIRuntime`):

| Variable | Purpose |
| --- | --- |
| `ALMONDAI_CHAT_KIND` | Backend identifier. Supported values: `openai`, `openrouter`, `huggingface`, `togetherai`, `deepinfra`, `h2o`, `rasa`, `botpress`, `deeppavlov`, `lmstudio`. |
| `ALMONDAI_ENDPOINT` | HTTPS endpoint for the chosen provider. |
| `ALMONDAI_MODEL` | Model identifier requested from the provider (where applicable). |
| `ALMONDAI_API_KEY` | Bearer token or API key required by the provider (if needed). |

When a backend is configured, teacher requests (`train.step`/`ingest.step` without
an explicit `teacher_output`) call the remote model first, falling back to the
local student or deterministic responses if the request fails.

## Annotating supervision with `teacher_source`

`train.step` and `ingest.step` now accept an optional `teacher_source` field. The
value is carried into the curated sample provenance so AlmondAI can retain
multiple answers for the same prompt—one per contributing LLM. Examples:

```json
{
  "method": "train.step",
  "params": {
    "prompt": "Draft a project kickoff email",
    "teacher_output": "...",
    "teacher_source": "claude-3",
    "prompt_hash": "custom:email"
  }
}
```

If `teacher_source` is omitted the runtime assigns one automatically:

- `remote_teacher` or the configured backend name when a remote LLM replied.
- `local_student` when the student model produced the supervision.
- `fallback_teacher` when only canned fallback content was available.
- `external_teacher` for manually supplied answers without a label.

The curator deduplicates on `(prompt_hash, teacher_source, teacher_output)`, so a
single prompt can accumulate feedback from several providers as long as the
`teacher_source` strings differ or the answers are unique.

## Logging and diagnostics

Each training step now logs the resolved `teacher_source` alongside the usual
statistics in `data/training_log.txt`. Inspecting the log reveals which backend
contributed a sample and whether the retrieval index supplied supporting
context.

Combine this information with the per-backend environment variables above to
experiment with ensembles or fallback hierarchies tailored to your deployment.

## LM Studio quick start

LM Studio exposes an OpenAI-compatible server on `http://127.0.0.1:1234` by
default. Launch the server inside LM Studio, then either:

- export `ALMONDAI_CHAT_KIND=lmstudio` (the endpoint defaults to
  `http://127.0.0.1:1234/v1/chat/completions` and the model defaults to
  `lmstudio`), or
- run `chat use lmstudio` from the AlmondAI console to apply the same defaults
  interactively.

While LM Studio remains active every `generate` request will also trigger an
automatic `train.step`, allowing the student model to learn from the remote
teacher in real time. The console prints a short status message after each
generation summarizing whether the update succeeded.

You can still override the endpoint/model manually if you host LM Studio on a
different port or want to address a specific model ID. Supplying an API key is
optional because the local server does not require authentication.

Need a quick way to verify LM Studio is reachable? Open
`docs/lmstudio_client.html` in a browser and issue prompts directly against the
server using the same OpenAI-compatible payloads that AlmondAI produces.
