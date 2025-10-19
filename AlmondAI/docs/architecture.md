# AlmondAI Architecture

AlmondAI operates as a self-evolving C++23 AI engine runtime composed of tightly
coupled subsystems that monitor, regenerate, and evaluate their own behaviour.
The sections below describe how the major components collaborate to keep the
runtime adaptive without sacrificing deterministic builds.

## Core Feedback Loop

1. **Source-Aware Introspection** – The learner watches compiler diagnostics,
   static analysis results, and repository diffs to detect opportunities for
   self-repair.
2. **Mutation Planning** – A planning layer drafts injector updates, adapter
   swaps, and data curation steps before promoting them into the live runtime.
3. **Self-Rebuild Execution** – The runtime executes build scripts, validates
   binaries, and reloads hot-swappable modules so improvements land without
   downtime.

## Learning Surfaces

- **ContinuousLearner** orchestrates ingestion, retrieval, evaluation, and
  policy governance, persisting each decision so the runtime can replay and
  audit its own behaviour.
- **AdapterManager** maintains multiple adapter stacks, allowing rapid
  promotion or rollback of new behaviours while keeping a stable base model.
- **RetrievalIndex** supplies grounded context from curated telemetry, recent
  transcripts, and documentation snapshots so generations remain anchored to the
  latest system state.

## Governance and Safety

- **PolicyGovernor** enforces guardrails on every generation, scanning for
  secrets, unsafe directives, and policy violations before approving a response.
- **Telemetry Ledger** records training stats, adapter health, and teacher
  provenance to maintain traceability across self-directed updates.

## Runtime Surfaces

- The **CLI console** in `AlmondShell/examples/AlmondAIRuntime` exposes
  generation, training, retrieval, and adapter controls for operators.
- Service integrations communicate through the Model Context Protocol bridge in
  `almondai::Service`, enabling remote automation while the engine continues its
  self-analysis loop.

AlmondAI continuously feeds lessons from each subsystem back into the
self-evolution pipeline, ensuring the engine grows alongside the software it
maintains.
