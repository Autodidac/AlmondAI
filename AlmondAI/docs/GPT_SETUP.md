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

If the key is missing the runtime falls back to a placeholder teacher response and
continues to accept manually provided `teacher_output` values.

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

## Seeding and persistence

* `data/training_seed.jsonl` ships with three starter conversations. On first run the
  learner copies them into `data/training_data.jsonl`, updates the vocabulary, and
  ingests them into the retrieval index.
* Each successful training step appends the curated sample to
  `data/training_data.jsonl`, logs metrics in `data/training_log.txt`, and writes the
  student weights to `data/student_weights.json`.
* Restarting the runtime reloads the vocabulary, training samples, and weights so the
  learner resumes where it left off.

## Troubleshooting

* Ensure `curl` is available in the runtime `PATH`.
* Inspect `data/training_data.jsonl` to confirm new teacher examples are being
  captured.
* Delete `data/student_weights.json` if you need to reset the student model to a
  freshly initialized state.
