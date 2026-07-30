# Docs

Executed code lives in scripts under `docs/snippets/`, included by `pymdownx.snippets`
(`--8<-- "docs/snippets/<path>.py:name"`, delimited by `# --8<-- [start:name]`). Use
`source="material-block"`, and `html="1"` for figures. One script per session, each
runnable standalone from the repo root.

Three traps, none visible from the sources:

- Build with `--clean --strict` (what `build_docs` does). Plain `zensical build` exits 0
  and reports "No issues found" even when a code block raised, and serves cached HTML.
- Each page gets a fresh markdown-exec session, so a page cannot use a variable bound on
  another. Hence the silent `reopen_*` sections atop getting-started pages 2 and 3.
- Print tables with the `print_table` helper and `html="1"`, never `.to_markdown()` —
  block-level Markdown is not run over markdown-exec output, so a pipe table stays literal
  `|---|`. The helper also strips pandas' `class`/`border` attributes, which every theme
  table rule is gated against (`table:not([class])`).
