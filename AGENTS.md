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

## Lints

The workspace manifest is the source of truth. The policy, taken from rig's
and koh's lint sets:

- `forbid` is the default for every correctness and panic-freedom lint:
  `unwrap_used`, `panic`, `unreachable`, `todo`, `unimplemented`,
  `dbg_macro`, `indexing_slicing`, `string_slice`, `get_unwrap`,
  `panic_in_result_fn`, `exit`, `await_holding_lock`, `unused_result_ok`,
  `map_err_ignore`, the lossy `cast_*` lints, `allow_attributes` and more.
  `clippy.toml` relaxes the panic lints inside tests only.
- `deny` only where a dependency's macro expands to an `allow` that `forbid`
  would reject: `expect_used` and `unwrap_in_result` (`#[tokio::test]`), the
  `suspicious` group (`proptest!`) and `rust_2018_idioms` (serde's derives).
  `unsafe_code` is `deny` for one FFI item in rig2-sqlite.
- `clippy::pedantic` and `clippy::nursery` are on as warnings (errors under
  `-D warnings`), minus a short allow list at the end of the manifest:
  naming and documentation lints, `cast_precision_loss`,
  `redundant_pub_crate` (it contradicts rustc's `unreachable_pub`) and
  nursery lints with known false positives.
- Suppress a lint locally with `#[expect(lint, reason = "...")]`, never
  `#[allow]`.

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
