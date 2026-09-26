# rig2 (attempt a)

rig2 is a Rust library for building with AI models: completion, embeddings,
images, audio, reranking, vision, agents, tools, retrieval and record/replay.
It is designed from the ground up around one idea: every model call is a
`Model<T>` of some `Task` `T`, and everything else (tracing, retries,
recording, agents, the Bevy integration) works for any task.

This repository is one of several independent attempts at the same design.
It is not published to crates.io. See [`REPORT.md`](REPORT.md) for the
design decisions, status and measurements.

## License

MIT
