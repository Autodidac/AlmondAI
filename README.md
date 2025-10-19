# AlmondAI

**AlmondAI** is a modular C++20 research runtime for iterating on lightweight
language-model experiments. The library exposes tokenizer tooling, adapter-aware
decoder primitives, retrieval utilities, and an MCP-facing service loop so
experiments can move from notebooks to long-running services without rebuilding
the scaffolding each time.

## Capabilities

- 🚀 **Vocabulary & Tokenization** – `almondai::WordTokenizer` builds,
  persists, and reloads vocabularies from plain text so datasets can be prepared
  quickly across runs.
- 🧠 **Adapter-aware Student Model** – `almondai::StudentModel` combines a
  configurable decoder with `almondai::AdapterManager` so adapters can be
  registered, promoted, or rolled back on the fly.
- 🔁 **Continuous Learning Loop** – `almondai::ContinuousLearner` coordinates
  ingestion, retrieval, evaluation, and policy governance to keep interactive
  services fresh while logging statistics to disk.
- 📚 **Retrieval Augmentation** – `almondai::RetrievalIndex` tracks curated
  samples and exposes TF-IDF search so inference requests can consult the most
  relevant history.
- 🌐 **Model Context Protocol Bridge** – `almondai::Service` pairs the learner
  with an `almondai::MCPBridge`, handling `train.step`, `ingest.step`, and
  `gpt.generate` requests over standard I/O.
- 🤝 **External Teachers** – Chat backends defined in `almondai::chat` let the
  runtime call OpenAI-compatible APIs, OpenRouter, Together AI, or other REST
  providers when a request omits `teacher_output`.

## Repository Layout

```
.
├── AlmondAI/                     # Cross-platform library (headers + sources)
│   ├── include/almondai/         # Public headers for runtime components
│   └── src/                      # Library implementation
├── AlmondAI.sln                  # Visual Studio solution (ships sample runtime)
├── AlmondShell/                  # MSVC project files and console demo entrypoint
├── CMakeLists.txt                # Builds the static lib with CMake
├── data/                         # Persistent state (vocab, seeds, logs, weights)
├── Changes                       # Project change log
├── LICENSE                       # LicenseRef-MIT-NoSell terms
└── README.md                     # Project overview (this file)
```

## Building the Library (CMake)

AlmondAI depends on a C++20 toolchain and libcurl. To produce the static library:

```bash
git clone https://github.com/Autodidac/AlmondAI.git
cd AlmondAI
cmake -B build -S . -G Ninja   # or any generator you prefer
cmake --build build            # emits libalmondai.a / almondai.lib
```

The library installs headers under `AlmondAI/include` and can be linked into
another application that provides an MCP host or CLI.

## Visual Studio Console Runtime

Windows developers can open `AlmondAI.sln` and build the
**AlmondAIEngine** (static library) followed by **AlmondAIRuntime**. The runtime
project hosts the interactive console located at
`AlmondShell/examples/AlmondAIRuntime/main.cpp`, wiring together:

- vocabulary bootstrap from `data/vocab.txt`
- the default `ContinuousLearner` instance
- optional external chat backends (see below)

Launch the resulting binary from the repository root so it can access the
`data/` directory.

## Runtime Data & Persistence

The learner keeps several artifacts in `data/`:

- `training_seed.jsonl` – starter prompts copied into
  `training_data.jsonl` on first run
- `training_data.jsonl` – curated samples ingested during operation
- `training_log.txt` – human-readable metrics per training/evaluation step
- `student_weights.json` – serialized decoder weights
- `vocab.txt` – tokenizer vocabulary
- `seed.txt` – description prompt used when seeding the teacher

These files are created automatically if they do not exist.

## Teacher & Chat Backends

When a `train.step` or `ingest.step` request omits `teacher_output`, the service
can call an external model. Configure the backend with environment variables
before launching the runtime:

- `ALMONDAI_CHAT_KIND` – selects the provider (`OpenAICompat`, `OpenRouter`,
  `TogetherAI`, `DeepInfra`, `HuggingFace`, etc.)
- `ALMONDAI_ENDPOINT` – provider-specific HTTP endpoint
- `ALMONDAI_MODEL` – remote model identifier
- `ALMONDAI_API_KEY` – credential used by the chat backend

For OpenAI-compatible teachers the helper in
[`AlmondAI/docs/GPT_SETUP.md`](AlmondAI/docs/GPT_SETUP.md) also honors:

- `ALMONDAI_GPT_API_KEY`
- `ALMONDAI_GPT_ENDPOINT` (defaults to `https://api.openai.com/v1/chat/completions`)
- `ALMONDAI_GPT_MODEL` (defaults to `gpt-4o-mini`)

See [`AlmondAI/docs/TEACHER_BACKENDS.md`](AlmondAI/docs/TEACHER_BACKENDS.md) for
notes on supported providers and response formats.

## Contributing

AlmondAI is a source-available commercial project and is not accepting external
pull requests. Please open an issue to discuss substantial changes.

## License

The project is distributed under `LicenseRef-MIT-NoSell`. Refer to the
[`LICENSE`](LICENSE) file for the complete, non-commercial usage terms.
