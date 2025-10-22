# AlmondAI Language Model Overview

Welcome to the guided tour of AlmondAI's language model stack. This document is designed
as a map for engineers, researchers, and operations teams who need to understand how the
pieces fit together without spelunking through every source file.

---

## Architectural Snapshot

The runtime lives primarily in `AlmondAI/src` with public headers under
`AlmondAI/include/almondai`. The core modules are grouped by responsibility so you can
quickly locate the code you need:

### Core Model Components
- **BaseDecoder** (`model.cpp`, `model.hpp`)
  - Compact decoder with embedding tables, configurable hidden layers, and a projection
    layer back to the vocabulary.
  - Constructor seeds weights from a normal distribution and exposes `forward`,
    `apply_gradients`, and `resize_vocab`.
- **StudentModel** (`model.cpp`)
  - Wraps `BaseDecoder`, owns optimization hyper-parameters, and provides `forward` / `update`
    helpers so callers never manipulate the base network directly.
- **Tensor helpers** (`tensor.cpp`, `tensor.hpp`)
  - Lightweight tensor math utilities shared by the decoder and adapters.

### Adaptation & Personalization
- **AdapterManager** and **Adapter** (`adapter.cpp`, `adapter.hpp`)
  - Register LoRA-style adapters, swap them in and out, and keep per-adapter gradient
    statistics used in health checks.
- **Adapter lifecycle utilities** (`serve.cpp`)
  - `promote_adapter` / `rollback_adapter` keep the retrieval index aware of active adapter
    changes so generations stay consistent.

### Tokenization & Data Handling
- **WordTokenizer** (`tokenizer_word.cpp`, `tokenizer_word.hpp`)
  - Streams UTF-8 code points directly from training prompts and replies,
    deduplicates tokens on the fly, and persists the quoted vocabulary to
    `data/vocab.txt`.
  - Signals `StudentModel` when the projection matrix must be resized and saves
    refreshed weights whenever the vocabulary grows.
- **DataCurator** (`ingest.cpp`, `ingest.hpp`)
  - Filters prompts, deduplicates by `(prompt_hash, teacher_source, teacher_output)`, records
    preference pairs, and enforces guardrails before training data is accepted.
- **RetrievalIndex** (`retrieval.cpp`, `retrieval.hpp`)
  - Stores curated samples for retrieval-augmented generation.
  - Returns scored hits and tracks a hit rate that feeds into training telemetry.
- **ContinuousLearner::consume_training_data_for_vocab** (`train.cpp`)
  - Scans `data/training_data.jsonl` during startup to rebuild the tokenizer
    vocabulary, deduplicating tokens and resizing student weights before
    loading samples into memory.

### Evaluation & Governance
- **Evaluator** (`eval.cpp`, `eval.hpp`)
  - Replays a canary set to produce loss and accuracy signals that catch regressions during
    long runs.
- **PolicyGovernor** (`governor.cpp`, `governor.hpp`)
  - Double-checks curated examples against safety policies before they are committed to disk.

---

## Service Surface

Runtime orchestration is handled by the `Service` class (`serve.cpp`, `serve.hpp`).
It wires the learner into the Model Context Protocol bridge and exposes a small collection
of JSON-RPC-style endpoints:

- **`model.generate`**
  - Retrieval-augmented generation. Computes a prompt hash, looks up retrieval matches,
    samples from the decoder using the configured decode settings, and returns the generated
    text with a context summary.
- **`ingest.step`** & **`train.step`**
  - Enroll new supervision. Delegates to `ContinuousLearner::ingest` and `train_step`,
    auto-invoking the GPT teacher via `MCPBridge` when no `teacher_output` is supplied.
- **`trainer.fit`**
  - Streams batches from `data/training_data.jsonl` for offline fine-tuning runs, emitting
    progress over the same bridge. (See `docs/DATA_FORMATS.md` for JSONL schema details.)
- **`eval.canary`**
  - Reports held-out evaluation metrics to confirm the student has not regressed.
- **Utility calls** — `retrieval.query`, `compiler.build`, `admin.hot_swap`, `gpt.generate`
  - Surface helper capabilities implemented in `retrieval.cpp`, `buildparse.cpp`, adapter
    management, and the teacher bridge.

`MCPBridge` (`mcp.cpp`, `mcp.hpp`) handles JSON serialization, message routing, and optional
delegation to external chat backends (`chat/backend.cpp`). If neither the local model nor a
remote teacher can answer, `fallback.cpp` provides deterministic canned replies.

---

## Continuous Learning Loop

Each training cycle keeps the model fresh while maintaining safety guarantees:

1. **Ingestion**
   - `ContinuousLearner::ingest` forwards prompts to `DataCurator`, refreshes the streaming
    tokenizer vocabulary via `ingest_training_pair`, stores curated samples in training/eval
    buffers, updates the retrieval index, and writes examples to `data/training_data.jsonl`
    (newline-delimited JSON).
2. **Training**
   - `train_step` tokenizes the prompt/teacher pair, runs a forward pass, forms a target
     distribution from the teacher tokens, applies cross-entropy against the student's
     softmax, backpropagates through the projection layer (and any active adapter), records
     statistics, persists weights to `data/student_weights.json`, and mirrors the student's
     reply into the curator's preference buffer.
3. **Evaluation**
   - `evaluate_canary` reuses the evaluator and retrieval hit rate to produce health metrics
     logged alongside training stats.
4. **Adapter lifecycle**
   - `promote_adapter` / `rollback_adapter` switch active adapters and notify the retrieval
     index so generations remain consistent.

---

## Quick Start Checklist

- Need to debug a generation? Inspect `Service::model_generate` in `serve.cpp` and follow the
  call into `RetrievalIndex::query` and `StudentModel::forward`.
- Tracing ingestion issues? Start with `ContinuousLearner::ingest` and the guardrail logic in
  `DataCurator`.
- Evaluating regressions? Run the canary pipeline via `eval.canary` and read metrics emitted
  by `Evaluator`.
- Planning adapter experiments? Review `AdapterManager` for registration details and the
  `promote_adapter` / `rollback_adapter` helpers in `serve.cpp`.

Keep this overview handy as a starting point—deep dives are still best done in the source, but
now you know exactly where to look.
