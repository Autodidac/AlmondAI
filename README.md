# AlmondAI

**AlmondAI** is a modular C++23 research runtime (migrating its previous C++20
codebase toward full C++26 compliance) for iterating on lightweight
language-model experiments. The project is evolving into a self-replicating,
lock-step software engine paired with an "AI next door" co-processor that will
eventually integrate directly into the runtime. The goal is to operate a
self-updating LLM that analyses its own sources, siphons curated telemetry and
external LLM transcripts, and applies scripted testing plus compiler debug
feedback to continually improve.

## Long-term Vision

- 🔄 **Self-Improving Engine** – Continuous compiler-driven diagnostics and
  scripted regression suites feed back into the runtime so it can iteratively
  refine generated source while staying in sync with neighbouring AI services.
- 🤖 **Integrated Companion AI** – A dedicated assistant process (the "AI next
  door") will graduate into a first-class runtime component, sharing datasets,
  training signals, and roadmap orchestration with AlmondAI.
- 🧬 **Source-Aware Learning** – The learner aims to replicate and extend
  high-quality codebases by analysing repositories end-to-end, harvesting
  lessons from other LLM outputs, and promoting successful diffs automatically.
- 🚀 **Standards Upgrade Path** – Ongoing modernization embraces C++23 features
  today with a clear plan to unlock upcoming C++26 facilities as toolchains
  mature.

## Capabilities

- 🚀 **Vocabulary & Tokenization** – `almondai::WordTokenizer` builds,
  persists, and reloads vocabularies from plain text so datasets can be prepared
  quickly across runs.
- 🧠 **Adapter-aware Student Model** – `almondai::StudentModel` combines a
  configurable decoder with `almondai::AdapterManager` so adapters can be
  registered, promoted, or rolled back on the fly.
- 🔁 **Continuous Learning Loop** – `almondai::ContinuousLearner` coordinates
  ingestion, retrieval, evaluation, and policy governance to keep interactive
  services fresh while logging statistics to disk, including auto-training on
  high-quality remote transcripts when a teacher is trusted.
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

### Console commands

Inside the runtime console you can type `help` at any time to surface the
interactive command list. The commands below are available in every session,
along with short usage notes and examples to clarify how they interact with the
continuous learner:

- **`help`** – redisplays the built-in cheat sheet. Handy when exploring the
  console for the first time or when switching between machines.
- **`generate <prompt>`** – produces a reply using the currently active route
  (local student or remote teacher). The console echoes the route and backend so
  you can confirm where the response came from. Example:
  ```text
  generate Summarize the last training run.
  ```
- **`retrieve <query>`** – inspects the retrieval index for matching samples.
  This is useful for debugging or verifying that curated data landed in the
  store. Example:
  ```text
  retrieve adversarial prompt mitigation
  ```
- **`train <file> [epochs=1] [batch=32]`** – performs supervised updates on a
  JSONL dataset. The runtime reports per-epoch metrics to the console and
  appends them to `data/training_log.txt`. Example:
  ```text
  train data/tutorial.jsonl 3 16
  ```
- **`hot-swap [name]`** – promotes a named adapter into the student model
  without restarting the process. If a `name` is supplied the learner loads and
  activates that adapter; if it is omitted the command rolls back to the default
  adapter stack. This is an easy way to A/B test new weights in a live service:
  ```text
  hot-swap nightly_adapter
  hot-swap                # roll back to the previously active adapter
  ```
- **`chat use <kind> [endpoint] [model] [key]`** – switches the teacher backend
  while the console is running. The `lmstudio` kind auto-fills
  `http://127.0.0.1:1234/v1/chat/completions` and `lmstudio` for the endpoint
  and model. Other kinds (for example `openrouter`, `togetherai`, or
  `openai`) fall back to the `ALMONDAI_*` environment variables when arguments
  are omitted. When LM Studio is active, remote responses are automatically fed
  back into the student for continual learning.
- **`chat clear`** – returns to the local student model without tearing down the
  console.
- **`exit` / `quit`** – cleanly stops the runtime.

`generate`, `train`, and `hot-swap` all stream progress messages as they run so
you can watch the learner state evolve. After each remote response routed
through `chat use`, the runtime records whether the teacher was involved,
validates it against the `PolicyGovernor`, and, when allowed, trains the student
on the new transcript while reporting loss/accuracy inline.

### Connecting an external LLM

You can connect a teacher model either by exporting environment variables before
launch or by issuing `chat use` commands once the console is open.

**Environment variables (recommended for automation)**

```bash
export ALMONDAI_CHAT_KIND=openai
export ALMONDAI_ENDPOINT=https://api.openai.com/v1/chat/completions
export ALMONDAI_MODEL=gpt-4o-mini
export ALMONDAI_API_KEY=sk-...
./build/AlmondShell/examples/AlmondAIRuntime/AlmondAIRuntime
```

The runtime will boot with the OpenAI-compatible backend already active and
announce that `gpt-4o-mini` is being used. The same variables work with other
providers—set `ALMONDAI_CHAT_KIND=openrouter`, `togetherai`, `deepinfra`, or any
other supported provider and adjust the endpoint/model as needed. The
`ALMONDAI_GPT_*` variables remain supported as fallbacks for OpenAI-compatible
deployments.

**Interactive switching (handy during experiments)**

```text
chat use lmstudio
generate Draft a customer welcome email.
chat clear
generate What adapters are currently loaded?
```

The first command hot-swaps the teacher route to LM Studio using the default
loopback server settings, the `generate` calls confirm where responses come
from, and `chat clear` returns to the local student without restarting the
process.

## LM Studio Integration

AlmondAI ships first-class defaults for LM Studio so you can keep experiments
local while still exercising the continuous learner:

- `chat use lmstudio` (or setting `ALMONDAI_CHAT_KIND=lmstudio`) wires the
  OpenAI-compatible endpoint `http://127.0.0.1:1234/v1/chat/completions` and
  default model name `lmstudio` without any extra arguments.
- The runtime enables "remote reply" auto-training when LM Studio is the active
  backend. Each completion is tagged with its source and streamed into the
  learner so the student weights evolve alongside LM Studio feedback.
- [`AlmondAI/docs/TEACHER_BACKENDS.md`](AlmondAI/docs/TEACHER_BACKENDS.md) and
  [`AlmondAI/docs/GPT_SETUP.md`](AlmondAI/docs/GPT_SETUP.md) include quick-start
  notes plus an HTML harness you can open in a browser to sanity-check LM Studio
  connectivity before using the console.

## Self-Learning Workflow

Behind the scenes the runtime keeps a persistent loop alive so the student model
can improve safely over time:

- Every remote transcript or curated sample flows through the
  `PolicyGovernor`, which enforces JSON-schema style constraints and blocklists
  before anything reaches the learner.
- Approved samples are indexed for retrieval, appended to `data/training_data.jsonl`,
  and logged to `data/training_log.txt` alongside loss/accuracy, adapter norm,
  and retrieval hit-rate metrics so you can audit outcomes offline.
- Automatic training is invoked after successful LM Studio calls and also exposed
  via the `train.step` MCP method and CLI `train` command, letting you replay
  entire corpora or stream new observations into the same continuous learner.
- Canary evaluations run through `evaluate_canary()` so you can spot regressions
  before promoting adapters or exporting weights.

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

`lmstudio` is now accepted as an `ALMONDAI_CHAT_KIND` alias. When selected (or
when issuing `chat use lmstudio` inside the console) AlmondAI defaults the
endpoint to `http://127.0.0.1:1234/v1/chat/completions` and the model name to
`lmstudio`, matching the OpenAI-compatible server bundled with LM Studio. Make
sure LM Studio has its HTTP server enabled before connecting.

For quick smoke tests you can also open
[`AlmondAI/docs/lmstudio_client.html`](AlmondAI/docs/lmstudio_client.html) in a
browser. The page provides a lightweight form that posts prompts directly to an
LM Studio server using the same OpenAI-compatible contract AlmondAI expects.

For OpenAI-compatible teachers the helper in
[`AlmondAI/docs/GPT_SETUP.md`](AlmondAI/docs/GPT_SETUP.md) also honors:

- `ALMONDAI_GPT_API_KEY`
- `ALMONDAI_GPT_ENDPOINT` (defaults to `https://api.openai.com/v1/chat/completions`)
- `ALMONDAI_GPT_MODEL` (defaults to `gpt-4o-mini`)

These `ALMONDAI_GPT_*` variables are also recognized as fallbacks for
`ALMONDAI_ENDPOINT`, `ALMONDAI_MODEL`, and `ALMONDAI_API_KEY`, so exporting only
the GPT-oriented names is enough to boot the console with an OpenAI-compatible
backend already active.

See [`AlmondAI/docs/TEACHER_BACKENDS.md`](AlmondAI/docs/TEACHER_BACKENDS.md) for
notes on supported providers and response formats.

## Contributing

AlmondAI is a source-available commercial project and is not accepting external
pull requests. Please open an issue to discuss substantial changes.

## License

The project is distributed under `LicenseRef-MIT-NoSell`. Refer to the
[`LICENSE`](LICENSE) file for the complete, non-commercial usage terms.
