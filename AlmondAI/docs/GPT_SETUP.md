# Configuring GPT Teacher Integration

AlmondAI can call an external GPT-compatible API whenever a training or ingestion
request does not supply a `teacher_output`. The bridge uses `curl` so the feature
works anywhere the binary can execute shell commands.

## Required environment variables

Set these before launching `almondai_app`:

| Variable | Purpose | Example |
| --- | --- | --- |
| `ALMONDAI_GPT_API_KEY` | Bearer token used for the HTTPS request. | `export ALMONDAI_GPT_API_KEY="sk-..."` |
| `ALMONDAI_GPT_ENDPOINT` | Optional override for the chat completions endpoint. Defaults to `https://api.openai.com/v1/chat/completions`. | `export ALMONDAI_GPT_ENDPOINT="https://api.openai.com/v1/chat/completions"` |
| `ALMONDAI_GPT_MODEL` | Optional model name sent to the API. Defaults to `gpt-4o-mini`. | `export ALMONDAI_GPT_MODEL="gpt-4o-mini"` |

The runtime treats `ALMONDAI_GPT_ENDPOINT`, `ALMONDAI_GPT_MODEL`, and
`ALMONDAI_GPT_API_KEY` as fallbacks for `ALMONDAI_ENDPOINT`,
`ALMONDAI_MODEL`, and `ALMONDAI_API_KEY`. You can export only the GPT-specific
variables and the console will still come up with the OpenAI-compatible backend
enabled.

If the key is missing the runtime falls back to a placeholder teacher response and
continues to accept manually provided `teacher_output` values. When supplying
answers from other LLMs you can add a `teacher_source` string to the
`train.step`/`ingest.step` payload so AlmondAI records which provider supplied
the supervision; see [`TEACHER_BACKENDS.md`](TEACHER_BACKENDS.md) for the list of
supported integrations.

## Request contract

When the service receives `train.step` or `ingest.step` without a teacher answer it
sends the following JSON body to the configured endpoint:

```json
{
  "model": "gpt-4o-mini",
  "temperature": 0.2,
  "messages": [
    {
      "role": "system",
      "content": "You are AlmondAI's teacher model. Provide thorough, safe answers suitable for fine-tuning."
    },
    {
      "role": "user",
      "content": "<prompt>\n\nConstraints:\n<serialized constraints>"
    }
  ]
}
```

Responses are expected to follow the OpenAI Chat Completions or Responses schemas.
The bridge extracts the first message content (`choices[0].message.content`) or, if
provided, the aggregated `output_text`. Custom providers should mimic one of these
shapes.

## Using LM Studio

LM Studio bundles an OpenAI-compatible HTTP server. After enabling it inside the
application:

1. Start the server for the model you want to supervise with.
2. Either export `ALMONDAI_CHAT_KIND=lmstudio` before launching AlmondAI (the
   endpoint defaults to `http://127.0.0.1:1234/v1/chat/completions` and the model
   defaults to `lmstudio`) or run `chat use lmstudio` inside the console to apply
   the same defaults interactively (see [`CONSOLE_HELP.md`](CONSOLE_HELP.md) for a
   quick command refresher). The runtime now waits up to 60 seconds for LM Studio
   to answer. If you need a different deadline, export
   `ALMONDAI_HTTP_TIMEOUT_MS=<milliseconds>` before starting the console.
3. Issue `generate` commands and the runtime will route requests through LM
   Studio, automatically curating and training on each remote reply while
   reporting the training outcome in the console.

If LM Studio listens on another port or you want to address a specific model
name, pass explicit values to `chat use lmstudio <endpoint> <model>`. The local
server does not require an API key, so the header is omitted automatically.

For ad-hoc testing without the console, open `docs/lmstudio_client.html` in a
browser. The standalone page exposes the same endpoint/model defaults and lets
you send prompts directly to an LM Studio server using the OpenAI-compatible
chat completions format.

LM Studio's newer OpenAI-compatible builds stream replies as structured
`content` arrays. AlmondAI and the bundled HTML harness now collapse those
segments automatically so messages render in the UI and are attributed to the
remote teacher instead of falling back to the local model.

## Seeding and persistence

* `data/training_seed.jsonl` ships with three starter conversations. On first run the
  learner copies them into the newline-delimited `data/training_data.jsonl`, updates
  the vocabulary, and ingests them into the retrieval index.
* Each successful training step appends the curated sample to
  `data/training_data.jsonl`, logs metrics in `data/training_log.txt`, and writes the
  student weights to `data/student_weights.json`.
* Refer to `docs/DATA_FORMATS.md` for schema details covering all persisted JSON and
  JSONL assets.
* Restarting the runtime reloads the vocabulary, training samples, and weights so the
  learner resumes where it left off.

## Troubleshooting

* Ensure `curl` is available in the runtime `PATH`.
* Inspect `data/training_data.jsonl` to confirm new teacher examples are being
  captured.
* Delete `data/student_weights.json` if you need to reset the student model to a
  freshly initialized state.
