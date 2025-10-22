# AlmondAI Console Help

This guide reformats the built-in `help` screen into an easy-to-scan reference.
Keep it handy when you are learning the runtime or showing it to someone new.

## Quick start

1. Launch the console runtime (for example, build and run
   `AlmondShell/examples/AlmondAIRuntime`).
2. Type `help` to print the on-screen cheat sheet.
3. Use the tables below when you need plain-language reminders about what each
   command does.

## Everyday navigation

| Command | Purpose | Example |
| --- | --- | --- |
| `help` | Reprints the short command list in the console. | `help` |
| `exit`, `quit` | Closes the runtime without leaving stray processes running. | `quit` |

## Get answers and browse memory

| Command | Purpose | Example |
| --- | --- | --- |
| `generate <prompt>` | Sends your question through the active model (local student or remote teacher) and shows who answered. | `generate Summarize the roadmap highlights.` |
| `retrieve <query>` | Looks up stored samples so you can confirm the learner has context for a topic. | `retrieve greeting` |

## Train or change adapters

| Command | Purpose | Example |
| --- | --- | --- |
| `directory [training]` | Prints absolute paths for the training dataset and log files under `data/`. | `directory` |
| `train <file> [epochs=1] [batch=32]` | Fine-tunes the student on a JSONL dataset and streams loss/accuracy to the console. | `train data/tutorial.jsonl 2 32` |
| `hot-swap [name]` | Activates a saved adapter without restarting. Run with no name to fall back to the default adapter stack. | `hot-swap nightly_adapter` |

## Manage chat backends

| Command | Purpose | Example |
| --- | --- | --- |
| `chat use <kind> [endpoint] [model] [key]` | Switches to a remote teacher. Missing endpoint/model/key arguments fall back to `ALMONDAI_*` environment variables and the `lmstudio` kind fills in its own defaults. | `chat use lmstudio` |
| `chat clear` | Returns to the local student after testing a remote backend. | `chat clear` |

## Quick recipes

### Get your first reply

1. Launch the console.
2. Type `generate Hello there!`.
3. The console prints the answer and labels which backend responded.

### Learn from a file of examples

1. Save your prompts and replies to a JSONL file such as `data/tutorial.jsonl`.
2. Run `train data/tutorial.jsonl 3 16` to train for three epochs with a batch
   size of 16.
3. Watch the console for streaming metrics; they are also saved to
   `data/training_log.txt`.

### Try a remote teacher then return to local mode

1. Run `chat use lmstudio` (or another supported `kind`) to enable the remote
   backend.
2. Use `generate <prompt>` to see replies and automatic training updates from the
   teacher.
3. Enter `chat clear` to switch back to the local student.

## Troubleshooting

- If a command is unfamiliar, type `help` for the built-in cheat sheet or skim
  this document.
- When `generate` fails, confirm that the active backend has network access or
  that required environment variables are set (see `docs/TEACHER_BACKENDS.md`).
- When `train` reports file errors, double-check that the path and filename are
  correct and that the file contains valid JSONL rows.
- If an adapter will not load, list the files under `data/adapters` (or your
  custom adapter directory) and confirm the expected name is present before
  calling `hot-swap`.

## Related documentation

- [`README.md`](../README.md) – Highlights the most common commands inside the
  "Console help at a glance" section.
- [`docs/TEACHER_BACKENDS.md`](TEACHER_BACKENDS.md) – Explains each supported
  remote teacher.
- [`docs/GPT_SETUP.md`](GPT_SETUP.md) – Walks through configuring OpenAI-style
  backends.
