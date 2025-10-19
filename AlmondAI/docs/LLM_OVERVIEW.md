\# AlmondAI Language Model Overview



This document summarizes the language-model stack implemented in `AlmondAI/src` and `AlmondAI/include/almondai`.  

It explains the core runtime, how the background services cooperate, and where persistent state lives so you can reason about changes without opening every source file.



---



\## Subsystem Atlas



\- \*\*BaseDecoder\*\* (`model.cpp` / `model.hpp`)  

&nbsp; Compact decoder with embeddings, configurable hidden layers, and a projection layer back to the vocabulary.  

&nbsp; Constructor initializes weights from a normal distribution. Exposes `forward`, `apply\_gradients`, and `resize\_vocab`.



\- \*\*StudentModel\*\* (`model.cpp`)  

&nbsp; Wraps the decoder, owns optimization hyper-parameters, and provides `forward` / `update` helpers so callers never manipulate the base network directly.



\- \*\*AdapterManager\*\* and \*\*Adapter\*\* (`adapter.cpp` / `adapter.hpp`)  

&nbsp; Register LoRA-style adapters, swap them in and out, and keep per-adapter gradient statistics used in health checks.



\- \*\*WordTokenizer\*\* (`tokenizer\_word.cpp` / `tokenizer\_word.hpp`)  

&nbsp; Whitespace tokenization, vocabulary growth, and persistence to `data/vocab.txt`.  

&nbsp; When the vocabulary grows it signals the student to resize its projection matrix.



\- \*\*DataCurator\*\* (`ingest.cpp` / `ingest.hpp`)  

&nbsp; Filters prompts, deduplicates by `(prompt\_hash, teacher\_source, teacher\_output)`, records preference pairs, and enforces guard-rails before training data is accepted.



\- \*\*RetrievalIndex\*\* (`retrieval.cpp` / `retrieval.hpp`)  

&nbsp; Stores curated samples for retrieval-augmented generation.  

&nbsp; Queries return scored hits and track a hit rate that feeds into training telemetry.



\- \*\*Evaluator\*\* (`eval.cpp` / `eval.hpp`)  

&nbsp; Replays a canary set to produce loss and accuracy signals that catch regressions during long runs.



\- \*\*PolicyGovernor\*\* (`governor.cpp` / `governor.hpp`)  

&nbsp; Double-checks curated examples against safety policies before they are committed to disk.



\- \*\*Tensor helpers\*\* (`tensor.cpp` / `tensor.hpp`)  

&nbsp; Tiny tensor ops used by the decoder and adapters.



---



\## Service Surface



Runtime orchestration happens in the `Service` class (`serve.cpp` / `serve.hpp`).  

It wires the learner into a Model Context Protocol bridge and exposes JSON-RPC-style endpoints:



\- \*\*`model.generate`\*\*  

&nbsp; Retrieval-augmented generation. Computes a prompt hash, looks up retrieval matches, samples from the decoder using the configured decode settings, and returns the generated text with a context summary.



\- \*\*`ingest.step`\*\* and \*\*`train.step`\*\*  

&nbsp; Enroll new supervision. Call `ContinuousLearner::ingest` and `train\_step` respectively, auto-invoking the GPT teacher via `MCPBridge` when no `teacher\_output` is supplied.



\- \*\*`trainer.fit`\*\*  

&nbsp; Streams batches from `data/training\_data.jsonl` for offline fine-tuning runs, emitting progress over the same bridge.



\- \*\*`eval.canary`\*\*  

&nbsp; Reports held-out evaluation metrics to confirm the student has not regressed.



\- \*\*Utility calls\*\* — `retrieval.query`, `compiler.build`, `admin.hot\_swap`, `gpt.generate`  

&nbsp; Surface helper capabilities implemented in `retrieval.cpp`, `buildparse.cpp`, adapter management, and the teacher bridge.



`MCPBridge` (`mcp.cpp` / `mcp.hpp`) handles JSON serialization, message routing, and optional delegation to external chat backends (`chat/backend.cpp`),  

while `fallback.cpp` provides deterministic canned replies if neither the local model nor a remote teacher can answer.



---



\## Continuous Learning Loop



1\. \*\*Ingestion\*\*  

&nbsp;  `ContinuousLearner::ingest` forwards prompts to the `DataCurator`, refreshes the tokenizer vocabulary, stores the curated sample in training/eval buffers, updates the retrieval index, and writes the example to `data/training\_data.jsonl`.



2\. \*\*Training\*\*  

&nbsp;  `train\_step` tokenizes the prompt/teacher pair, runs a forward pass, computes a squared-error loss, applies gradients to the student (and active adapter), records statistics, and persists the student weights to `data/student\_weights.json`.



3\. \*\*Evaluation\*\*  

&nbsp;  `evaluate\_canary` reuses the evaluator and retrieval hit rate to produce health metrics logged alongside training stats.



4\. \*\*Adapter lifecycle\*\*  

&nbsp;  `promote\_adapter` / `rollback\_adapter` switch active adapters and keep the retrieval index aware of the swap so generations remain consistent.



---



\## Persistence and Data Files



\- `data/training\_data.jsonl` — curated samples appended during ingestion  

\- `data/training\_seed.jsonl` — starter conversations loaded on first run  

\- `data/student\_weights.json` — decoder weights persisted after every training step  

\- `data/vocab.txt` — serialized tokenizer vocabulary  

\- `data/training\_log.txt` — human-readable training, evaluation, and retrieval metrics  



&nbsp; Example entry:

&nbsp; ```text

&nbsp; Step 42 | loss=0.1234 | accuracy=1 | adapter\_norm=3.1415 | retrieval\_hit\_rate=0.75


## Extending the Stack

- **Implement new chat providers**  
  Subclass `chat::Backend` and register them via `chat::make_backend`.

- **Add telemetry hooks**  
  Extend `ContinuousLearner::log_stats` or the retrieval index to ship richer analytics.

- **Introduce structured decoding strategies**  
  Modify the sampling utilities in `serve.cpp` to support techniques like top-k, nucleus, or temperature sampling.

---

With these pieces in mind, you can trace how requests flow from the MCP bridge to the learner,  
how supervision is persisted, and which files to modify when introducing new features.

