# AGENTS.md

Rules for anyone, human or agent, changing this repository.

## Design principles

1. One way to do each thing. No second spelling, alias or convenience layer.
2. Types make wrong states unrepresentable.
3. Data and behaviour are separate. Serialized types never hold I/O.
4. Sans-IO at the core. Provider mapping, stream parsing and the agent loop
   are pure code over values. I/O sits at the edges.
5. Pay for what you use. Each provider, store and integration is a crate.
6. Extensible by traits users can implement, with the same power as the
   built-in implementations.
7. WASM and native are both first-class.
8. Small. An abstraction must delete more than it adds.

## Code rules

- No `unwrap`, `expect`, `panic!`, `todo!`, `unimplemented!` or `dbg!`
  outside tests. Workspace clippy lints enforce this.
- No `String` as an error type. Fallible APIs return `rig2_core::Error`, or a
  crate-specific error that converts into it.
- Use `MaybeSend` and `MaybeSync` from `rig2-core`, never raw `Send` and
  `Sync` bounds on public APIs, so wasm32 keeps working. Use `BoxFuture` and
  `BoxStream` from `rig2-core` for boxed futures and streams.
- Test modules are sibling files: write `#[cfg(test)] mod tests;` and put the
  body in `foo/tests.rs` (for `foo.rs`) or `tests.rs` beside `lib.rs`.
- `unsafe` is denied workspace-wide. A crate that needs it (FFI) allows it on
  the one item, with a `// SAFETY:` comment.
- Every crate sets `publish = false`. Nothing is published.
- Dependencies are the latest stable release. Manifests state the full
  latest version (`"1.12.1"`, not `"1"`), and a new dependency is checked on
  crates.io before it is added.
- Never commit secrets. Recordings and cassettes are scrubbed as they are
  written, and `cargo xtask scan` checks every fixture. Run it before pushing.

## Clippy

`clippy::pedantic` is on, with these exceptions, allowed in the workspace
manifest: `module_name_repetitions`, `must_use_candidate`,
`missing_errors_doc`, `missing_panics_doc`, `return_self_not_must_use`,
`similar_names`, `too_many_lines`, `doc_markdown`, `single_match_else`, the four numeric `cast_*` lints,
`items_after_statements` and `struct_field_names`. Errors are documented in
prose where they are not obvious.

## Documentation style

- Module docs state the module's purpose in at most three paragraphs.
- Item docs state the contract, inputs, outputs and errors. Short sentences.
- One example per public entry point. Examples compile; use `no_run` when
  they need credentials or a network.
- Every public item is documented (`missing_docs`).

## Checks

Before every push:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo test --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
cargo xtask scan
```

Tests run in replay mode: no network and no credentials. Live runs are opt-in
(`RIG2_LIVE=1`) and record fixtures.
