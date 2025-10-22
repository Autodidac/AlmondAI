# AlmondAI

**AlmondAI** is a self-evolving C++23 AI engine runtime engineered to spawn and
wire up games, tools, and service bots that remain in lock-step with an
embedded, AI-augmented co-processor. The codebase—actively migrating from its
previous C++20 heritage toward full C++26 compliance—treats language-model
reasoning as a first-class runtime dependency. The runtime studies its own
sources, composes fresh injectors, and clones working subsystems into new
targets while it siphons curated telemetry, external LLM transcripts, and
compiler feedback loops to keep every replica improving between releases.

## Long-term Vision

- 🧩 **Composable Replicators** – Every subsystem (rendering, simulation,
  scripting, policy governance) can be exported as an injector bundle that the
  runtime redeploys into fresh workspaces, seeding playable prototypes or
  service bots with the same guardrails as the original host.
- 🤖 **Resident Co-Processor** – The "AI next door" migrates from sidecar to
  sovereign component, sharing context memory, roadmap cues, and mutation
  pipelines while keeping transcripts synchronised across replicas.
- 🧬 **Source-Aware Learning** – The learner continuously audits its own
  artefacts, upstream dependencies, and captured LLM conversations to promote
  working diffs, roll back regressions, and annotate the knowledge graph that
  powers future injections.
- 🚀 **Forward-Looking Standards** – The engine pushes into C++23/26 territory
  with concepts, coroutines, and metaprogramming hooks that make the replicator
  graph scriptable without sacrificing deterministic builds.

## Capabilities

- 🧪 **Injector Recipes** – `almondai::WordTokenizer`, dataset normalisers, and
  curriculum scripts package into reproducible recipes that new replicas can
  replay before they spawn their own services.
- 🧠 **Adapter-Savvy Student Model** – `almondai::StudentModel` and
  `almondai::AdapterManager` co-manage decoder weights so replicators can test,
  promote, or retire behaviours without dropping the control plane.
- 🔁 **Self-Calibration Loop** – `almondai::ContinuousLearner` orchestrates
  ingestion, retrieval, evaluation, and policy governance with telemetry hooks
  that feed back into the mutation planner driving the next wave of injections.
- 🏷️ **Explainable Learning Tags** – `ContinuousLearner` now emits
  `[learn::…]` traces for every ingestion, training, and evaluation step so
  operators can audit autonomous updates in real time or replay them from the
  training log.
- 📚 **Retrieval-Backed Memory** – `almondai::RetrievalIndex` curates history
  with TF-IDF and semantic tags, letting each replica spin up with the most
  relevant narratives and operating procedures.
- 🌐 **Model Context Protocol Bridge** – `almondai::Service` fronts the MCP
  interface so automation can issue `train.step`, `ingest.step`, and
  `gpt.generate` commands while the co-processor streams checkpoints and
  mutation notes.
- 🤝 **Teacher Federation** – Providers in `almondai::chat` keep the runtime
  multilingual and multi-backend, routing through OpenAI-compatible APIs,
  OpenRouter, Together AI, and other REST teachers while tagging transcripts for
  downstream replicas.

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

### Console help at a glance

Type `help` inside the console at any time to open the built-in cheat sheet.
For a printable version—including command summaries, quick recipes, and
troubleshooting tips—see [`docs/CONSOLE_HELP.md`](AlmondAI/docs/CONSOLE_HELP.md).
Need a precise breakdown of arguments, behaviours, and side effects? Check the
command index in [`docs/CONSOLE_COMMANDS.md`](AlmondAI/docs/CONSOLE_COMMANDS.md).

#### Everyday navigation

| Command | What it does | Try it |
| --- | --- | --- |
| `help` | Reprints the concise command overview. Helpful when you are new to the console or returning after a break. | `help` |
| `exit`, `quit` | Closes the runtime cleanly. | `quit` |

#### Get answers and inspect memory

| Command | What it does | Try it |
| --- | --- | --- |
| `generate <prompt>` | Sends a prompt through the active route (local student or remote teacher) and shows which backend replied. | `generate Summarize the last training run.` |
| `retrieve <query>` | Searches the retrieval index so you can confirm that curated samples or seed data are available. | `retrieve greeting curriculum` |

#### Train or swap adapters

| Command | What it does | Try it |
| --- | --- | --- |
| `directory [training]` | Prints the absolute paths for the training dataset and logs under `data/` so you can open them in an editor. | `directory` |
| `train <file> [epochs=1] [batch=32]` | Runs supervised updates on a JSONL dataset, streaming metrics to the console and appending them to `data/training_log.txt`. | `train data/tutorial.jsonl 3 16` |
| `hot-swap [name]` | Loads the named adapter without restarting. Run with no arguments to roll back to the default stack. | `hot-swap nightly_adapter` |

#### Manage chat backends

| Command | What it does | Try it |
| --- | --- | --- |
| `chat use <kind> [endpoint] [model] [key]` | Switches the teacher backend while the console is running. `lmstudio` auto-fills sensible defaults. | `chat use lmstudio` |
| `chat clear` | Returns to the local student after testing a remote teacher. | `chat clear` |

`generate`, `train`, and `hot-swap` stream progress messages so you can watch the
learner state change in real time. When a remote response arrives via `chat use`
the runtime tags the transcript with the teacher name, validates it against the
`PolicyGovernor`, and (when approved) trains the student immediately.

### Bootstrapped greeting curriculum

On the first launch the runtime now guarantees a rich seed curriculum of
English greetings, acknowledgements, and friendly farewells. The
`ContinuousLearner` writes the prompts into `data/training_seed.jsonl` with
provenance tags such as `seed::greeting::hello`, covering salutations like
"Hiya", "Sup?", "How's it going?", "Welcome back", and "Long time no see". Each
seed pairs a conversationally natural teacher reply with the prompt so the local
student can field everyday introductions before any external data arrives. The
same samples flow into the retrieval index and evaluation set, letting the
learner demonstrate conversational competence immediately.

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

### Fast LM Studio bootstrap

When you want the local student to absorb LM Studio answers as quickly as
possible, opt into the compact curriculum and higher learning rates:

- `ALMONDAI_SEED_PROFILE=compact` swaps the verbose seed set for a concise
  English primer covering greetings, clarifying questions, Markdown context
  formatting, and friendly farewells so the model trains on immediately useful
  phrasing.
- `ALMONDAI_FAST_LEARNING=1` bumps the decoder learning rate to `5e-3` unless
  you provide a custom rate.
- `ALMONDAI_LEARNING_RATE=<value>` pins an explicit rate (e.g. `0.01`) and takes
  precedence over the fast-learning toggle—ideal when you're tuning against a
  specific LM Studio model.

The console prints which overrides are active at launch, letting you confirm the
student is in "fast follow" mode before you start streaming prompts.

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

- `training_seed.jsonl` – newline-delimited JSON seed prompts copied into
  `training_data.jsonl` on first run so the learner always has an onboarding
  curriculum
- `training_data.jsonl` – newline-delimited JSON records appended as curated
  samples during operation
- `training_log.txt` – human-readable metrics per training/evaluation step
- `student_weights.json` – serialized decoder weights stored as a single JSON
  document
- `retrieval_index.json` – JSON metadata snapshot for the retrieval index
- `vocab.txt` – tokenizer vocabulary (one token per line)
- `seed.txt` – description prompt used when seeding the teacher

See [`AlmondAI/docs/DATA_FORMATS.md`](AlmondAI/docs/DATA_FORMATS.md) for full
schemas and validation tips covering the JSON and JSONL artifacts managed by
the runtime.

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
