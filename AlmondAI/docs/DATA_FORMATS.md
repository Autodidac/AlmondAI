# AlmondAI Data Formats

AlmondAI persists its long-term state under `data/` using a mix of newline-delimited
JSON (JSONL), standard JSON documents, and plain text. This reference documents the
expected format for every file the runtime touches so you can prepare compatible
artifacts and automate validations.

## Summary by Artifact

| File | Format | Schema / Notes |
| --- | --- | --- |
| `data/training_seed.jsonl` | **JSONL** – one JSON object per line | `prompt` + `teacher_output` pairs following [`training_sample.schema.json`](data-schemas/training_sample.schema.json) |
| `data/training_data.jsonl` | **JSONL** – same shape as seeds | Appended samples validated by the `DataCurator`; see the same schema |
| `data/training_log.txt` | Plain text | Human-readable metrics emitted by `ContinuousLearner::log_stats` |
| `data/student_weights.json` | JSON document | Decoder weights saved by `BaseDecoder::save_weights`; see [`student_weights.schema.json`](data-schemas/student_weights.schema.json) |
| `data/retrieval_index.json` | JSON document | Metadata exported by `RetrievalIndex::save_metadata`; see [`retrieval_index.schema.json`](data-schemas/retrieval_index.schema.json) |
| `data/vocab.txt` | Plain text | UTF-8 tokens serialised with `std::quoted` in the order expected by the streaming tokenizer |
| `data/seed.txt` | Plain text | Free-form bootstrap prompt text consumed when regenerating seed samples |

Any additional files created by local experiments should live outside of `data/` or
follow the same conventions if you expect the runtime to read them.

## JSONL Sample Shape

`training_seed.jsonl` and `training_data.jsonl` store newline-delimited JSON records.
Each line must be a standalone JSON object so the runtime can stream the files
incrementally. The canonical fields are:

```jsonc
{
  "prompt": "Explain how AlmondAI keeps training.",
  "teacher_output": "AlmondAI appends curated JSONL samples and refreshes adapters between runs.",
  "constraints": {},
  "provenance": {
    "source": "seed",
    "prompt_hash": "seed::continuous_learning",
    "teacher_hash": "8840530102558217964"
  },
  "semantic_tags": [
    "source:seed",
    "prompt:explain"
  ]
}
```

The fields map directly to `CuratedSample` inside `ingest.hpp`. Only `prompt` and
`teacher_output` are required; every other property is optional. When missing,
`ContinuousLearner` fills `constraints` and `provenance` with empty objects before
persisting new lines. The JSON schema is provided in
[`data-schemas/training_sample.schema.json`](data-schemas/training_sample.schema.json).

### Validating JSONL Data

Because JSONL files contain one JSON document per line, tools such as `jq`, `python`,
or `jsonschema` should read them line-by-line. Example validation using Python:

```bash
python - <<'PY'
import json, jsonschema, pathlib
schema = json.load(open('AlmondAI/docs/data-schemas/training_sample.schema.json'))
for path in ('data/training_seed.jsonl', 'data/training_data.jsonl'):
    p = pathlib.Path(path)
    if not p.exists():
        continue
    for i, line in enumerate(p.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        jsonschema.validate(json.loads(line), schema)
print('ok')
PY
```

## Retrieval Metadata

`data/retrieval_index.json` captures a complete snapshot of the TF-IDF index used by
`RetrievalIndex`. The structure mirrors the in-memory representation:

- `stats.query_count` / `stats.hit_count` record aggregate usage counters.
- `documents` is an array of objects keyed by `id` with optional `tags` and cached
  `tokens` used to accelerate reconstruction.

The document conforms to [`data-schemas/retrieval_index.schema.json`](data-schemas/retrieval_index.schema.json).

## Student Weights

`data/student_weights.json` stores the decoder configuration and tensors emitted by
`BaseDecoder::save_weights`. The JSON document contains:

- `config` – vocabulary size, hidden size, number of layers, context length, and the
  most recent learning rate.
- `weights` – an array of tensors where each entry has a `shape` (list of dimension
  sizes) and `data` (flattened float values).

Refer to [`data-schemas/student_weights.schema.json`](data-schemas/student_weights.schema.json)
if you need to validate or generate compatible checkpoints.

## Plain-Text Assets

- `data/training_log.txt` is purely informational and never parsed back into the
  runtime.
- `data/vocab.txt` is written by the streaming `WordTokenizer`. Each line is a
  `std::quoted` UTF-8 token, allowing whitespace, emoji, and other multi-byte
  characters to survive round-trips without loss.
- `data/seed.txt` stores the default greeting curriculum; edit it to customise the
  generated seed JSONL samples.

The learner refreshes `data/vocab.txt` by calling
`ContinuousLearner::consume_training_data_for_vocab`, which streams the JSONL
training corpus, tokenises each prompt/response pair on the fly, deduplicates
tokens, and resizes the student weights whenever the vocabulary grows.

Keep these files UTF-8 encoded. The runtime ignores blank lines where appropriate.
