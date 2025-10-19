# AlmondAI

**AlmondAI** is a modular C++20 research runtime for serving and iterating on
lightweight language models. It bundles tokenizer utilities, adapter-aware model
primitives, streaming inference, and online learning helpers so experiments can
move from notebooks to production-style services without rebuilding the
infrastructure every time.

## Highlights

- 🚀 **Tokenizer and Vocabulary Tooling** – `almondai::WordTokenizer` can build,
  persist, and reload vocabularies from plain text so datasets can be prepared
  in minutes.
- 🧠 **Adapter-Aware Models** – `almondai::BaseDecoder` and
  `almondai::AdapterManager` make it easy to register LoRA-style adapters,
  switch them on the fly, and keep per-adapter statistics up to date.
- 🔁 **Continuous Learning Loop** – `almondai::ContinuousLearner` coordinates the
  student model, adapter manager, tokenizer, and safety governor to keep
  long-running services fresh without sacrificing guardrails.
- 🌐 **MCP Bridge Integration** – `almondai::Service` combines the learner with
  an `almondai::MCPBridge` so inference requests can flow over the Model Context
  Protocol using the same parsing utilities that power the CLI sample.
- 🧩 **Composable Utilities** – Headers in `include/almondai/` cover tensor
  helpers, retrieval primitives, training utilities, policy governance, and
  JSON/MCP helpers. Each component is standalone and can be mixed into existing
  projects.

## Repository Layout

```
.
├── AlmondAI.sln                      # Visual Studio solution (Windows)
├── AlmondShell/                      # MSVC projects and legacy engine assets
│   └── examples/
│       ├── AlmondAIEngine/          # Static library wrapper for shared items
│       └── AlmondAIRuntime/         # Console sample linked against the engine
├── CMakeLists.txt                    # Cross-platform build script
├── include/almondai/                # Public headers
├── src/                              # Library implementation and sample entry
├── data/                             # Runtime data such as vocabularies
├── Images/                           # Ancillary artwork (unused by default)
├── LICENSE                           # LicenseRef-MIT-NoSell terms
└── README.md                         # Project overview (this file)
```

Legacy engine headers and scripts remain under `AlmondShell/` for developers who
need the shared MSBuild items; the primary AlmondAI runtime lives in
`include/almondai/` and `src/`.

## Building

### Cross-platform (CMake + Ninja/MSBuild)

```bash
git clone https://github.com/Autodidac/AlmondAI.git
cd AlmondAI
cmake -B build -S . -G Ninja  # or "Visual Studio 17 2022"
cmake --build build
```

The default target builds the console sample defined in `src/main.cpp`, which
runs the MCP-enabled service loop.

### Visual Studio 2022

1. Open `AlmondAI.sln`.
2. Build the **AlmondAIEngine** static library target.
3. Build and run **AlmondAIRuntime** to launch the console harness.

Both projects import the shared `Engine.vcxitems` items, and the runtime target
links against the static library output located in `$(SolutionDir)\x64\$(Configuration)`.

## Running the Sample Service

The runtime expects a vocabulary file at `data/vocab.txt`. On first launch the
sample will generate the file automatically using `WordTokenizer::save_vocab`.
Subsequent runs reload the vocabulary, configure a `StudentModel`, register a
`default` adapter, and expose the inference loop via `Service::run` over standard
I/O.

### Teacher Integration and Seeding

`data/training_seed.jsonl` ships with three example prompts so the learner has
initial vocabulary coverage and retrieval documents. When `almondai_app` starts it
loads those seeds, copies them into `data/training_data.jsonl`, and reloads any
previously saved weights from `data/student_weights.json`.

If you omit `teacher_output` from `train.step` or `ingest.step` requests the
runtime calls an external GPT endpoint using the credentials described in
[`docs/GPT_SETUP.md`](docs/GPT_SETUP.md). To plug in alternative providers (for
example OpenRouter, Hugging Face Inference, or a self-hosted Rasa instance) set
the environment variables documented in
[`docs/TEACHER_BACKENDS.md`](docs/TEACHER_BACKENDS.md) and, optionally, tag
manually-supplied supervision with a `teacher_source` label. Successful training
steps append the curated example to `data/training_data.jsonl` and persist the
updated weights so subsequent runs resume from the latest state.

## Contributing

"We Are Not Accepting PRs At This Time" as AlmondAI is a source-available
commercial project. For substantial changes please open an issue first.

## License

AlmondAI is distributed under the `LicenseRef-MIT-NoSell` license. See
[`LICENSE`](LICENSE) for the complete terms, including the non-commercial usage
requirements.
